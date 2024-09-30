from common_header import *
from torch.utils.data import DataLoader, TensorDataset

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)

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
def loss_function_v(model_v, input_domain, X_train, y_train, iter):
    device = config["device"]
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
    loss_fn = nn.MSELoss()
    loss_MSE = loss_fn(F_mse,y_train)
    # Epixy
    epsilon = config["hyperparameters"]["epsilon_v"]
    alpha = config["hyperparameters"]["alpha"]
    epxi = (torch.norm(input_domain, p=2, dim=1)**2 * epsilon).clone().detach().to(device)
    # Hyperparameters
    DECAY_VPOS = config["hyperparameters"]["decay_vpos"]
    DECAY_LV = config["hyperparameters"]["decay_lv"]
    DECAY_TUNE = config["hyperparameters"]["decay_tune"]
    DECAY_L0 = config["hyperparameters"]["decay_l0"]
    DECAY_MSE = config["hyperparameters"]["decay_mse"]
    # Loss
    loss_zero = (X0).pow(2) 
    loss_lie = (F.leaky_relu(L_V + epxi, alpha)).mean()
    loss_vpos = F.leaky_relu(epxi - V_domain, alpha).mean()
    loss_tune = ((Circle_Tuning-V_domain).pow(2)).mean()
    Lyapunov_risk = DECAY_VPOS * loss_vpos + DECAY_LV * loss_lie  + DECAY_TUNE * loss_tune + DECAY_L0 * loss_zero + DECAY_MSE * loss_MSE
    vio_pos = np.sum(V_domain.cpu().detach().numpy() < 0)
    vio_lie = np.sum(L_V.cpu().detach().numpy() > 0)
    # Calcuation of Accuracy
    acc_pos = 1 - vio_pos/len(V_domain)
    acc_L = 1 - vio_lie/len(L_V)
    wandb.log({"Lyapunov Risk": Lyapunov_risk.item(), "MSE Loss": loss_MSE.item(), "V_pos accuracy: ":acc_pos, "V_lie accuracy: ":acc_L})
    # print_info(f"{iter}) Lyapunov LOSS = {Lyapunov_risk.item():.5E}, MSE = {loss_MSE.item():.5E}, V_0_loss = {loss_vpos.item():.5E}, V_pos_loss = {loss_vpos.item():.5E}, Lv_loss = {loss_lie.item():.5E},\
    #     Circular Tuning Loss = {loss_tune.item():.5E}, Lie Violations = {vio_lie}, Positive Violations = {vio_pos}")
    return Lyapunov_risk

# Defining the Barrier Loss Function
def loss_function_b(model_b, model_v, input_init, input_unsafe, input_domain, X_train, iter):
    device = config["device"]
    # For the domain
    input_domain = input_domain.float().to(device)
    # For the unsafe set
    input_unsafe = input_unsafe.float().to(device)
    B_unsafe = model_b(input_unsafe)
    # For the init set
    input_init = input_init.float().to(device)
    B_init = model_b(input_init)
    ## LOSS
    # MSE Loss
    B_mse = model_b(X_train)
    # Hyperparameters
    DECAY_INIT = config["hyperparameters"]["decay_init"]
    DECAY_UNSAFE = config["hyperparameters"]["decay_unsafe"]
    DECAY_CUT = config["hyperparameters"]["decay_cut"]
    DECAY_LB = config["hyperparameters"]["decay_lb"]
    # Loss to push to right solution
    loss_cut = DECAY_CUT * F.relu(B_mse).mean()
    # Total Loss
    epsilon = config["hyperparameters"]["epsilon_b"]
    alpha = config["hyperparameters"]["alpha"]
    epi_bound = config["hyperparameters"]["epi_bound"]
    loss_init = DECAY_INIT * (F.leaky_relu(B_init + epsilon, alpha)).mean()
    loss_unsafe = DECAY_UNSAFE * (F.leaky_relu(- B_unsafe + epsilon, alpha)).mean()
    #### NEED TO CHECK BOUNDARY STUFF FOR BARRIER
    with torch.no_grad():
        B_domain = model_b(input_domain)
        boundary_index = ((B_domain[:,0] >= -epi_bound) & (B_domain[:,0] <= epi_bound)).nonzero()
        input_boundary = torch.index_select(input_domain, 0, boundary_index[:, 0])
    if len(input_boundary) > 0:
        input_boundary.requires_grad = True
        B_boundary = model_b(input_boundary)
        F_boundary = model_v(input_boundary)[1]
        gradient_boundary = torch.autograd.grad(
                torch.sum(B_boundary),
                input_boundary,
                grad_outputs=None,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        L_B = torch.sum(gradient_boundary * F_boundary, dim=1)
    else:
        L_B = torch.tensor([0.0])
    loss_lieb = DECAY_LB * F.relu(L_B + epsilon).mean()
    Barrier_risk = loss_init + loss_unsafe + loss_lieb + loss_cut
    # Violations
    vio_unsafe = np.sum(B_unsafe.cpu().detach().numpy() < 0)
    vio_init = np.sum(B_init.cpu().detach().numpy() > 0)
    vio_lie = np.sum(L_B.cpu().detach().numpy() > 0)
    # Accuracy
    acc_unsafe = 1- vio_unsafe/len(B_unsafe)
    acc_init = 1-  vio_init/len(B_init)
    acc_lie = 1-vio_lie/len(L_B)
    wandb.log({"Barrier Risk": Barrier_risk.item(), "Unsafe Set accuracy: ":acc_unsafe, "Init Set accuracy: ":acc_init, "Boundary Set accuracy: ":acc_lie})
    # print(f"{iter}) Barrier LOSS = {Barrier_risk.item():.5E}, loss_init = {loss_init.item():.5E}, loss_unsafe = {loss_unsafe.item():.5E},\
    # loss_lieb = {loss_lieb.item():.5E}, loss_cut = {loss_cut.item():.5E}, Init Violations = {vio_init}, Unsafe Violations = {vio_unsafe}, Lie Violations = {vio_lie}")
    return Barrier_risk

def lyapunovVerify(model_v, x_domain, iter):
    device = config["device"]
    eta = config["hyperparameters"]["eta"]
    N_CE = config["hyperparameters"]["n_counter_examples"]
    flag = True
    countVio = 0
    batch_size = config["model_v"]["batch_size"] 
    DOMAIN = config["domain"]["range"]
    epsilon = config["hyperparameters"]["epsilon_v"]
    
    # Generate x and y ranges
    x = np.linspace(DOMAIN[0][0], DOMAIN[0][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0])/eta))
    y = np.linspace(DOMAIN[1][0], DOMAIN[1][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0])/eta))
    X, Y = np.meshgrid(x, y)

    # Convert X and Y to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    # Concatenate X and Y to create input data tensor
    tot_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(tot_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    alloted = N_CE

    # Iterate through the DataLoader
    for batch in data_loader:
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
        print_warning(f"Epoch {iter} : Total No of Violations: {countVio}")

    return x_domain, flag

def barrierVerify(model_v, model_b, x_domain, x_unsafe, x_init, initial_set_centre):
    device = config["device"]
    eta = config["hyperparameters"]["eta"]
    N_CE = config["hyperparameters"]["n_counter_examples"]
    batch_size = config["model_b"]["batch_size"]
    flag = True
    
    def process_data(x_range, y_range, condition_fn, violation_message, storage_tensor):
        # Generate x and y ranges based on eta
        x = np.linspace(x_range[0], x_range[1], math.ceil((x_range[1] - x_range[0]) / eta))
        y = np.linspace(y_range[0], y_range[1], math.ceil((y_range[1] - y_range[0]) / eta))
        X, Y = np.meshgrid(x, y)
        alloted = N_CE//3
        # Convert X and Y to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
        
        # Create DataLoader with all data in batches of a suitable size
        dataset = TensorDataset(input_data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
                    storage_tensor = storage_tensor.to(device)
                    for i in np.random.randint(0, len(violation_indices[0]), size=allotment):
                        storage_tensor = torch.cat((storage_tensor, data_batch[violation_indices[0][i]].unsqueeze(0)), dim=0)
                    alloted -= allotment
                    return storage_tensor, False  
            del data_batch
            torch.cuda.empty_cache() 
        return storage_tensor, True
    
    # Adding counter-examples to initial domain
    init_radius = config["init"]["radius"]
    INIT = [[c - init_radius, c + init_radius] for c in initial_set_centre]
    x_init, flag_init = process_data(INIT[0], INIT[1], 
                                     lambda B_out: np.nonzero(B_out.cpu().detach().numpy() > 0),
                                     "Violation of Barrier in Initial Domain",
                                     x_init)
    flag = flag and flag_init
    
    # Adding counter-examples to unsafe domain UNSAFE_1
    unsafe_radius = config["unsafe"]["radius"]
    unsafe_set_centre = config["unsafe"]["centre"]
    UNSAFE = [[c - unsafe_radius, c + unsafe_radius] for c in unsafe_set_centre]
    x_unsafe, flag_unsafe = process_data(UNSAFE[0], UNSAFE[1],
                                          lambda B_out: np.nonzero(B_out.cpu().detach().numpy() < 0),
                                          "Violation of Barrier in Unsafe Domain",
                                          x_unsafe)
    flag = flag and flag_unsafe
    
    # Processing boundary region
    DOMAIN = config["domain"]["range"]
    x = np.linspace(DOMAIN[0][0], DOMAIN[0][1], math.ceil((DOMAIN[0][1] - DOMAIN[0][0]) / eta))
    y = np.linspace(DOMAIN[1][0], DOMAIN[1][1], math.ceil((DOMAIN[1][1] - DOMAIN[1][0]) / eta))
    X, Y = np.meshgrid(x, y)
        
    # Convert X and Y to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
        
    # Create DataLoader with all data in batches of a suitable size
    dataset = TensorDataset(input_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    alloted = N_CE//3
    # Process data using DataLoader
    for batch in data_loader:
            data_batch = batch[0].to(device)
            with torch.no_grad():
                epi_bound = 0.05
                B_domain = model_b(data_batch)
                boundary_index = ((B_domain[:,0] >= -epi_bound) & (B_domain[:,0] <= epi_bound)).nonzero()
                input_boundary = torch.index_select(data_batch, 0, boundary_index[:, 0])
            if len(input_boundary) > 0:
                input_boundary.requires_grad = True
                B_boundary = model_b(input_boundary)
                F_boundary = model_v(input_boundary)[1]
                gradient_boundary = torch.autograd.grad(
                        torch.sum(B_boundary),
                        input_boundary,
                        grad_outputs=None,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=True)[0]
                L_B = torch.sum(gradient_boundary * F_boundary, dim=1)
                epsilon = config["hyperparameters"]["epsilon_b"]
                LBpos = L_B.cpu().detach().numpy() > -epsilon 
                # Adds 10 random elements
                vio = np.nonzero(LBpos)
                if alloted <=0:
                    break
                elif len(vio[0]) > 0:
                    print_warning("Violation of Barrier in Boundary Region")
                    flag_boundary = False
                    allotment = min(alloted, len(vio[0]))
                    for i in np.random.randint(0, len(vio[0]), size=allotment):
                        x_domain = torch.cat((x_domain, input_data[vio[0][i]].unsqueeze(0)),dim=0)     
                    alloted -= allotment
            del data_batch, input_boundary
            torch.cuda.empty_cache()
    flag = flag and flag_boundary
    # If no violations are found
    if flag:
        print_success("Barrier is certified")
    return x_init, x_unsafe, x_domain, flag

