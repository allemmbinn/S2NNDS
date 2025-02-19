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

# Defining the Overall Loss Function for Lyapunov and the Dynamics Function
def loss_function_v(model_v, input_domain, X_train, y_train, iter, config):
    device = next(model_v.parameters()).device
    # For the domain
    input_domain = input_domain.float().to(device)
    # Zero set
    x_0 = torch.zeros([1, 2]).to(device)
    X0 = model_v(x_0)[0]
    ## LOSS
    # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
    V_domain, F_domain = model_v(input_domain_clone)
    gradient_boundary = torch.autograd.grad(
                    torch.sum(V_domain),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    L_V = torch.sum(gradient_boundary * F_domain, dim=1)
    # Circular Tuning
    Circle_Tuning = Tune(input_domain)
    Circle_Tuning = Circle_Tuning.to(device)
    # MSE Loss
    F_mse = model_v(X_train)[1]
    # Epixy
    epsilon = config["hyperparameters"]["epsilon_v"]
    alpha = config["hyperparameters"]["alpha"]
    epxi = (torch.norm(input_domain, p=2, dim=1)**2 * epsilon).clone().detach().to(device)
    # Regularization
    l2_norm = sum(param.pow(2).sum() for layer in model_v.layers_v for param in layer.parameters())
    # l2_norm = sum(p.pow(2).sum() for p in model_v.parameters())
    # Hyperparameters
    DECAY_VPOS = config["hyperparameters"]["decay_vpos"]
    DECAY_LV = config["hyperparameters"]["decay_lv"]
    DECAY_TUNE = config["hyperparameters"]["decay_tune"]
    DECAY_L0 = config["hyperparameters"]["decay_l0"]
    DECAY_MSE = config["hyperparameters"]["decay_mse"]
    DECAY_L2 = config["hyperparameters"]["decay_l2"]
    # Loss
    loss_fn = nn.MSELoss()
    loss_MSE = DECAY_MSE * loss_fn(F_mse,y_train)
    loss_zero = DECAY_L0 * (X0).pow(2) 
    loss_lie  = DECAY_LV * (F.leaky_relu(L_V + epxi, alpha)).mean()
    loss_vpos = DECAY_VPOS * F.leaky_relu(epxi - V_domain, alpha).mean()
    loss_tune = DECAY_TUNE * ((Circle_Tuning-V_domain).pow(2)).mean()
    loss_reg  = DECAY_L2 * l2_norm
    Lyapunov_risk = loss_vpos + loss_lie  + loss_tune + loss_zero + loss_MSE + loss_reg
    vio_pos = np.sum(V_domain.cpu().detach().numpy() < 0)
    vio_lie = np.sum(L_V.cpu().detach().numpy() > 0)
    # Calcuation of Accuracy
    acc_pos = 1 - vio_pos/len(V_domain)
    acc_L = 1 - vio_lie/len(L_V)
    # print_info(f"{iter}) Lyapunov LOSS = {Lyapunov_risk.item():.5E}, MSE = {loss_MSE.item():.5E}, V_0_loss = {loss_vpos.item():.5E}, V_pos_loss = {loss_vpos.item():.5E}, Lv_loss = {loss_lie.item():.5E},\
    #     Circular Tuning Loss = {loss_tune.item():.5E}, Lie Violations = {vio_lie}, Positive Violations = {vio_pos}")
    wandb.log({"Lyapunov Risk": Lyapunov_risk.item(), "MSE Loss": loss_MSE.item(), "V_pos accuracy: ":acc_pos, "V_lie accuracy: ":acc_L})

    return Lyapunov_risk

# Defining the Barrier Loss Function
def loss_function_b(model_b, model_v, input_init, input_unsafe, input_domain, X_train, iter, config):
    device = next(model_v.parameters()).device
    # For the domain
    input_domain = input_domain.float().to(device)
    # For the unsafe set
    input_unsafe = input_unsafe.float().to(device)
    B_unsafe = model_b(input_unsafe)
    # For the init set
    input_init = input_init.float().to(device)
    B_init = model_b(input_init)
    ## LOSS
    # # MSE Loss
    # B_mse = model_b(X_train)
    # Hyperparameters
    DECAY_INIT = config["hyperparameters"]["decay_init"]
    DECAY_UNSAFE = config["hyperparameters"]["decay_unsafe"]
    DECAY_CUT = config["hyperparameters"]["decay_cut"]
    DECAY_LB = config["hyperparameters"]["decay_lb"]
    DECAY_L2 = config["hyperparameters"]["decay_l2"]
    epsilon = config["hyperparameters"]["epsilon_b"]
    alpha = config["hyperparameters"]["alpha"]
    epi_bound = config["hyperparameters"]["epi_bound"]
    # Finding the Boundary Points for Lie Derivative
    # with torch.no_grad():
    #     B_domain = model_b(input_domain)
    #     boundary_index = ((B_domain[:,0] >= -epi_bound) & (B_domain[:,0] <= epi_bound)).nonzero()
    #     input_boundary = torch.index_select(input_domain, 0, boundary_index[:, 0])
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
    # else:
    #     L_B = torch.tensor([0.0])
    # Finding the Lie Derivative for entire Domain
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
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
    # Regularization
    l2_norm = sum(p.pow(2).sum() for p in model_b.parameters())
    # Total Loss
    # loss_cut = DECAY_CUT * F.relu(B_mse).mean()
    loss_cut = 0
    loss_init = DECAY_INIT * (F.leaky_relu(B_init + epsilon, alpha)).mean()
    loss_unsafe = DECAY_UNSAFE * (F.leaky_relu(- B_unsafe + epsilon, alpha)).mean()
    loss_lieb = DECAY_LB * F.relu(L_B + epsilon).mean()
    loss_reg = DECAY_L2 * l2_norm
    Barrier_risk = loss_init + loss_unsafe + loss_lieb + loss_cut + loss_reg
    # Violations
    vio_unsafe = np.sum(B_unsafe.cpu().detach().numpy() < 0)
    vio_init = np.sum(B_init.cpu().detach().numpy() > 0)
    vio_lie = np.sum(L_B.cpu().detach().numpy() > 0)
    # Accuracy
    acc_unsafe = 1- vio_unsafe/len(B_unsafe)
    acc_init = 1-  vio_init/len(B_init)
    acc_lie = 1-vio_lie/len(L_B)
    # Logging on Wandb
    wandb.log({"Barrier Risk": Barrier_risk.item(), "Unsafe Set accuracy: ":acc_unsafe, "Init Set accuracy: ":acc_init, "Boundary Set accuracy: ":acc_lie})
    # print(f"{iter}) Barrier LOSS = {Barrier_risk.item():.5E}, loss_init = {loss_init.item():.5E}, loss_unsafe = {loss_unsafe.item():.5E},\
    # loss_lieb = {loss_lieb.item():.5E}, loss_cut = {loss_cut.item():.5E}, Init Violations = {vio_init}, Unsafe Violations = {vio_unsafe}, Lie Violations = {vio_lie}")
    return Barrier_risk

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

