from common_header import *
from torch.utils.data import DataLoader, TensorDataset

# Helper Function for Circular Tuning
def Tune(x):
    y = []
    for r in range(0,len(x)):
        v = 0
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y

def loss_function_dyn(model_f, X_train, y_train, config):
    device = next(model_f.parameters()).device
    X_train_clone = X_train.float().to(device)
    y_train_clone = y_train.float().to(device)
    # MSE Loss
    loss_fn = nn.MSELoss()
    loss_MSE = loss_fn(model_f(X_train_clone),y_train_clone)
    # Regularization
    l2_norm = sum(param.pow(2).sum() for layer in model_f.layers_f for param in layer.parameters())
    # Hyperparameters
    DECAY_MSE = config["hyperparameters"]["decay_mse"]
    DECAY_L2 = config["hyperparameters"]["decay_l2_f"]
    # Loss
    loss_mse = DECAY_MSE*loss_MSE
    loss_reg = DECAY_L2 * l2_norm
    return loss_mse #, loss_reg

def loss_function_domain(model_v, model_b, model_f, input_domain, config):
    device = next(model_v.parameters()).device
    model_b = model_b.to(device)
    model_f = model_f.to(device)
    # For the domain
    input_domain = input_domain.float().to(device)
    #Lyapunov Losses
    # Zero set
    x_0 = torch.zeros([1, 2]).to(device)
    V_0 = model_v(x_0)
    # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
    V_value = model_v(input_domain_clone)
    f_value = model_f(input_domain_clone)
    grad_lyap = torch.autograd.grad(
                    torch.sum(V_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_lyap = torch.sum(grad_lyap * f_value, dim=1)
    #Lie derivative of barrier 
    B_value = model_b(input_domain_clone)
    abs_B = torch.abs(B_value)
    lie_tol = config["hyperparameters"]["lie_tol"]
    mask = abs_B <= lie_tol
    B_value = B_value[mask] 
    grad_bar = torch.autograd.grad(
                    torch.sum(B_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_barr = torch.sum(grad_bar * f_value, dim=1)
    #SKIP circular tuning for now, add it if needed
    # Add Regularization for barrier and Lyapunov
    l2_norm_lyap = sum(param.pow(2).sum() for layer in model_v.layers_v for param in layer.parameters()) 
    l2_norm_barr = sum(param.pow(2).sum() for layer in model_b.layers_b for param in layer.parameters())
    #Getting the hyperparameters
    tol = config["hyperparameters"]["tol"]
    DECAY_V0 = config["hyperparameters"]["decay_v0"]
    DECAY_VPOS = config["hyperparameters"]["decay_vpos"]
    DECAY_LV = config["hyperparameters"]["decay_lv"]
    DECAY_L2_V = config["hyperparameters"]["decay_l2_V"]
    DECAY_LB = config["hyperparameters"]["decay_lb"]
    DECAY_L2_B = config["hyperparameters"]["decay_l2_B"]
    alpha = config["hyperparameters"]["alpha"] #Parameter for leaky relu
    #Invidual weighted losses
    loss_zero = DECAY_V0 * (V_0).pow(2)
    loss_lie  = DECAY_LV * (F.leaky_relu(lie_lyap + tol, alpha)).mean() + DECAY_LB * (F.leaky_relu(lie_barr + tol, alpha)).mean() 
    loss_vpos = DECAY_VPOS * (F.leaky_relu(tol + 0.001 - V_value, alpha)).mean()
    loss_reg = DECAY_L2_V * l2_norm_lyap + DECAY_L2_B * l2_norm_barr
    #total loss
    return loss_lie + loss_vpos + loss_zero #+ loss_vpos #+ loss_reg

def loss_function_init(model_b, input_init, config):
    device = next(model_b.parameters()).device
    input_init = input_init.float().to(device)
    # For the init set
    B_init = model_b(input_init)
    # Finding the Lie Derivative for entire Domain
    # Hyperparameters
    tol = config["hyperparameters"]["tol"] 
    DECAY_INIT = config["hyperparameters"]["decay_init"]
    alpha = config["hyperparameters"]["alpha"]
    # Total Loss
    loss_init = DECAY_INIT *(F.leaky_relu(B_init + tol, alpha)).mean()
    return  loss_init

def loss_function_unsafe(model_b, input_unsafe, config):
    device = next(model_b.parameters()).device
    input_unsafe = input_unsafe.float().to(device)
    # For the unsafe set
    B_unsafe = model_b(input_unsafe)
    #hyperparameters
    tol = config["hyperparameters"]["tol"] 
    DECAY_UNSAFE = config["hyperparameters"]["decay_unsafe"]
    alpha = config["hyperparameters"]["alpha"]
    #losses
    loss_unsafe =  DECAY_UNSAFE * (F.leaky_relu(- B_unsafe + 0.001 + tol, alpha)).mean()
    return loss_unsafe
        
def lyapunovVerify(model_v, x_domain, iter, config, DOMAIN):
    device = next(model_v.parameters()).device
    eta = config["hyperparameters"]["eta"]
    N_CE = config["hyperparameters"]["n_counter_examples"]
    flag = True
    countVio = 0
    batch_size = config["model_v"]["batch_size"] 
    epsilon = config["hyperparameters"]["epsilon_v"]
    
    # Generate x and y ranges
    x = torch.linspace(DOMAIN[0][0], DOMAIN[0][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0])/eta))
    y = torch.linspace(DOMAIN[1][0], DOMAIN[1][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0])/eta))
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Convert X and Y to torch tensors
    X_tensor = X.float()
    Y_tensor = Y.float()

    # Concatenate X and Y to create input data tensor
    total_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)

    # Create a TensorDataset and DataLoader
    domain_dataset = TensorDataset(total_data)
    domain_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    alloted = N_CE
    # Iterate through the DataLoader
    for batch in domain_dataloader:
        input_data = batch[0].to(device)  # Unpack the input data from the batch
        # Lyapunov Function
        torch.cuda.empty_cache()
        input_domain_clone = input_data.clone().requires_grad_().to(device)
        V_domain, F_domain = model_v(input_domain_clone)
        gradient_boundary = torch.autograd.grad(
                            torch.sum(V_domain),
                            input_domain_clone,
                            grad_outputs=None,
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=True)[0]
        L_V = torch.sum(gradient_boundary * F_domain, dim=1)
        # Check conditions
        Vneg = V_domain.cpu().detach().numpy() < epsilon
        Lvpos = L_V.cpu().detach().numpy() > -epsilon
        input_data = input_data.cpu()
        # Adds counterexamples for Vneg violations
        vio = np.nonzero(Vneg)
        if alloted <=0:
            break
        elif len(vio[0]) > 0:
            if flag:
                print_warning("Violation of Lyapunov in Positive Definiteness")
            countVio += len(vio[0])
            # Add random elements violating Vneg to x_domain
            allotment = min(alloted, len(vio[0]))
            for i in np.random.randint(0, len(vio[0]), size=allotment):
                x_domain = torch.cat((x_domain, input_data[vio[0][i]].unsqueeze(0)), dim=0)
            flag = False
            alloted -= allotment
        
        # Adds counterexamples for Lvpos violations
        vio = np.nonzero(Lvpos)
        if alloted<=0:
            break
        elif len(vio[0]) > 0:
            countVio += len(vio[0])
            if flag:
                print_warning("Violation of Lyapunov in Lie Derivative")
            # Add random elements violating Lvpos to x_domain
            allotment = min(alloted, len(vio[0]))
            for i in np.random.randint(0, len(vio[0]), size=allotment):
                x_domain = torch.cat((x_domain, input_data[vio[0][i]].unsqueeze(0)), dim=0)
            flag = False
            alloted -= allotment

    # Print status
    if flag:
        print_success(f"Epoch {iter} : Lyapunov is certified with eta: {eta}")
    else:
        print_warning(f"Epoch {iter} : Total No of Violations in Lyapunov: {countVio}")

    return x_domain, flag

def barrierVerify(model_v, model_b, x_domain, x_unsafe, x_init, initial_set_centre, config, DOMAIN):
    device = next(model_v.parameters()).device
    eta = config["hyperparameters"]["eta"]
    N_CE = config["hyperparameters"]["n_counter_examples"]
    batch_size = config["model_b"]["batch_size"]
    flag = True
    countVio = 0
    def process_data(x_range, y_range, condition_fn, violation_message, storage_tensor):
        # Generate x and y ranges based on eta
        x = torch.linspace(x_range[0], x_range[1], math.ceil((x_range[1] - x_range[0]) / eta))
        y = torch.linspace(y_range[0], y_range[1], math.ceil((y_range[1] - y_range[0]) / eta))
        X, Y = torch.meshgrid(x, y, indexing='ij')
        alloted = N_CE//3
        # Convert X and Y to torch tensors
        X_tensor = X.float()
        Y_tensor = Y.float()
        input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
        countVio = 0
        # Create DataLoader with all data in batches of a suitable size
        dataset = TensorDataset(input_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        # Process data using DataLoader
        for batch in data_loader:
            data_batch = batch[0].to(device)
            with torch.no_grad():
                B_out = model_b(data_batch)
                violation_indices = condition_fn(B_out)
                if alloted <=0:
                    break
                elif len(violation_indices[0]) > 0:
                    print_warning(violation_message)
                    allotment = min(alloted, len(violation_indices[0]))
                    data_batch = data_batch.cpu()
                    for i in np.random.randint(0, len(violation_indices[0]), size=allotment):
                        storage_tensor = torch.cat((storage_tensor, data_batch[violation_indices[0][i]].unsqueeze(0)), dim=0)
                    alloted -= allotment
                    countVio += allotment
                    return storage_tensor, False, countVio 
            del data_batch
            torch.cuda.empty_cache() 
        return storage_tensor, True, countVio 
    
    # Adding counter-examples to initial domain
    init_radius = config["init"]["radius"]
    INIT = [[c - init_radius, c + init_radius] for c in initial_set_centre]
    x_init, flag_init, count = process_data(INIT[0], INIT[1], 
                                     lambda B_out: np.nonzero(B_out.cpu().detach().numpy() > 0),
                                     "Violation of Barrier in Initial Domain",
                                     x_init)
    flag = flag and flag_init
    countVio += count
    
    # Adding counter-examples to unsafe domain UNSAFE_1
    unsafe_radius = config["unsafe"]["radius"]
    unsafe_set_centre = config["unsafe"]["centre"]
    UNSAFE = [[c - unsafe_radius, c + unsafe_radius] for c in unsafe_set_centre]
    x_unsafe, flag_unsafe, count = process_data(UNSAFE[0], UNSAFE[1],
                                          lambda B_out: np.nonzero(B_out.cpu().detach().numpy() < 0),
                                          "Violation of Barrier in Unsafe Domain",
                                          x_unsafe)
    flag = flag and flag_unsafe
    countVio += count
    
    # Processing boundary region
    x = torch.linspace(DOMAIN[0][0], DOMAIN[0][1], math.ceil((DOMAIN[0][1] - DOMAIN[0][0]) / eta))
    y = torch.linspace(DOMAIN[1][0], DOMAIN[1][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0]) / eta))
    X, Y = torch.meshgrid(x, y, indexing='ij')
        
    # Convert X and Y to torch tensors
    X_tensor = X.float()
    Y_tensor = Y.float()
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2)
        
    # Create DataLoader with all data in batches of a suitable size
    dataset = TensorDataset(input_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    alloted = N_CE//3
    # Process data using DataLoader
    for batch in data_loader:
        data_batch = batch[0].to(device)
        input_domain_clone = torch.clone(data_batch).requires_grad_().to(device)
        B_domain = model_b(input_domain_clone)
        F_domain = model_v(input_domain_clone)[1]
        gradient_boundary = torch.autograd.grad(
                        torch.sum(B_domain),
                        input_domain_clone,
                        grad_outputs=None,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=True)[0]
        L_B = torch.sum(gradient_boundary * F_domain, dim=1)
        epsilon = config["hyperparameters"]["epsilon_b"]
        LBpos = L_B.cpu().detach().numpy() > -epsilon
        # Adds 10 random elements
        vio = np.nonzero(LBpos)
        if alloted <=0:
            break
        elif len(vio[0]) > 0:
            print_warning("Violation of Barrier Derivative Condition")
            flag_boundary = False
            input_data = input_data.cpu()
            allotment = min(alloted, len(vio[0]))
            for i in np.random.randint(0, len(vio[0]), size=allotment):
                x_domain = torch.cat((x_domain, input_data[vio[0][i]].unsqueeze(0)),dim=0)     
            alloted -= allotment
            countVio += allotment
        # if len(input_boundary) > 0:
        #     input_boundary.requires_grad = True
        #     B_boundary = model_b(input_boundary)
        #     F_boundary = model_v(input_boundary)[1]
        #     gradient_boundary = torch.autograd.grad(
        #             torch.sum(B_boundary),
        #             input_boundary,
        #             grad_outputs=None,
        #             create_graph=True,
        #             only_inputs=True,
        #             allow_unused=True)[0]
        #     L_B = torch.sum(gradient_boundary * F_boundary, dim=1)
        #     epsilon = config["hyperparameters"]["epsilon_b"]
        #     LBpos = L_B.cpu().detach().numpy() > -epsilon 
        #     # Adds 10 random elements
        #     vio = np.nonzero(LBpos)
        #     if alloted <=0:
        #         break
        #     elif len(vio[0]) > 0:
        #         print_warning("Violation of Barrier in Boundary Region")
        #         flag_boundary = False
        #         allotment = min(alloted, len(vio[0]))
        #         for i in np.random.randint(0, len(vio[0]), size=allotment):
        #             x_domain = torch.cat((x_domain, input_data[vio[0][i]].unsqueeze(0)),dim=0)     
        #         alloted -= allotment
        # del data_batch, input_boundary
        torch.cuda.empty_cache()
    flag = flag and flag_boundary
    # If no violations are found
    if flag:
        print_success(f"Barrier is certified with eta: {eta}")
    else:
        print_warning(f"Total No of Violations in Barrier: {countVio}")
    print_info("#####################################################")
    return x_init, x_unsafe, x_domain, flag

