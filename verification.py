from common_header import *

def verify_domain(model_v, model_b, model_f, input_domain, config):
    N = config["counterex"]["no_min"]
    device = next(model_v.parameters()).device
    model_v = model_v.to(device)
    model_b = model_b.to(device)
    model_f = model_f.to(device)
    input_domain = input_domain.float().to(device)
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
    
    #remove points close to equilibrium
    # Define the bounds
    lower_bound = config["counterex"]["lb"]
    upper_bound = config["counterex"]["ub"]

    # Create a boolean mask for points within the bounds
    mask_x = (input_domain_clone[:, 0] < lower_bound) | (input_domain_clone[:, 0] > upper_bound)
    mask_y = (input_domain_clone[:, 1] < lower_bound) | (input_domain_clone[:, 1] > upper_bound)
    mask = mask_x & mask_y

    # Apply the mask to filter out the points
    input_domain_clone = input_domain_clone[mask]    
    V_value = model_v(input_domain_clone)

    f_value = model_f(input_domain_clone)
    #lyapunov lie derivative counterexamples
    lyap_tol = config["counterex"]["lyap_tol"]
    grad_lyap = torch.autograd.grad(
                    torch.sum(V_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_lyap = torch.sum(grad_lyap * f_value, dim=1)
    lyap_mask = lie_lyap > - lyap_tol
    filtered_lie_lyap = lie_lyap[lyap_mask]
    filtered_input_domain = input_domain_clone[lyap_mask]
    
    # get the maximum values of lie_lyap
    if filtered_lie_lyap.numel() > 0:  # Ensure there are valid values
        # Sample indices according to probabilities
        num_elements = filtered_lie_lyap.numel()
        num_samples = min(N, num_elements)
        min_indices = torch.randperm(num_elements)[:num_samples]        # Select the corresponding counterexample points
        lie_lyap_cex = filtered_input_domain[min_indices]
        print("CEs for Lie_lyap:", lie_lyap_cex)
    else:
        lie_lyap_cex = torch.empty((0, filtered_input_domain.shape[1]), device=device)
    
    #lyapunov negative value counterexamples
    pos_tol = config["counterex"]["pos_tol"]
    pos_mask = V_value[:,0] < pos_tol
    filtered_V_Value = V_value[pos_mask].view(-1)
    filtered_input_domain = input_domain_clone[pos_mask]
    
    #get the last minimum values of V_pos

    if filtered_V_Value.numel() > 0:  # Ensure there are valid values
        num_elements = filtered_V_Value.numel()
        num_samples = min(N, num_elements)
        min_indices = torch.randperm(num_elements)[:num_samples]        # Select the corresponding counterexample points
        # Select the corresponding counterexample points
        V_pos_cex = filtered_input_domain[min_indices]
        print("CEs for V_pos:", V_pos_cex)
    else:
        V_pos_cex = torch.empty((0, filtered_input_domain.shape[1]), device=device)

    #barrier lie derivative counterexamples
    bar_tol =  config["counterex"]["bar_tol"]
    lie_tol = config["hyperparameters"]["lie_tol"]
    B_value = model_b(input_domain_clone)
    grad_bar = torch.autograd.grad(
                    torch.sum(B_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_bar = torch.sum(grad_bar * f_value, dim=1)
    bar_mask = (torch.abs(B_value[:,0]) <= lie_tol) & (lie_bar > -bar_tol)
    #bar_mask = (lie_bar > -bar_tol)
    filtered_lie_bar = lie_bar[bar_mask]
    filtered_input_domain = input_domain_clone[bar_mask]

    if filtered_lie_bar.numel() > 0:  # Ensure there are valid values
        num_elements = filtered_lie_bar.numel()
        num_samples = min(N, num_elements)
        min_indices = torch.randperm(num_elements)[:num_samples]        # Select the corresponding counterexample points
        # Select the corresponding counterexample points
        lie_bar_cex = filtered_input_domain[min_indices]
        print("CEs for Lie_bar:", lie_bar_cex)
    else:
        lie_bar_cex = torch.empty((0, filtered_input_domain.shape[1]), device=device)

    tensors = [lie_lyap_cex, V_pos_cex, lie_bar_cex]
    non_empty_tensors = [t for t in tensors if t.numel() > 0]
    if non_empty_tensors:
        concatenated_cex = torch.unique(torch.cat(non_empty_tensors, dim=0), dim = 0)   
    else:
        concatenated_cex = torch.empty((0, input_domain_clone.shape[1]))  # Empty tensor with correct shape
    return concatenated_cex.cpu()

def verify_init(model_b,init_domain, config):
    N = config["counterex"]["no_min"]
    device = next(model_b.parameters()).device
    # barrier initial set counterexamples
    init_domain_clone = init_domain.float().to(device)
    B_value = model_b(init_domain_clone)
    inun_tol = config["counterex"]["inun_tol"]
    mask = B_value[:,0] > -inun_tol
    filtered_B_value = B_value[mask].view(-1)
    filtered_init_domain = init_domain_clone[mask]

    if filtered_B_value.numel() > 0:    
        num_elements = filtered_B_value.numel()
        num_samples = min(N, num_elements)
        min_indices = torch.randperm(num_elements)[:num_samples]  
        # Select the corresponding counterexample points
        cex = filtered_init_domain[min_indices]
    else:
        cex = torch.empty((0, filtered_init_domain.shape[1]), device=device)
    return cex.cpu()

def verify_unsafe(model_b, unsafe_domain, config):
    N = config["counterex"]["no_min"]   
    device = next(model_b.parameters()).device
    # barrier unsafe set counterexamples
    unsafe_domain_clone = unsafe_domain.float().to(device)
    B_value = model_b(unsafe_domain_clone)
    inun_tol = config["counterex"]["inun_tol"]
    mask = B_value[:,0] <= inun_tol
    filtered_B_value = B_value[mask].view(-1)
    filtered_uns_domain = unsafe_domain_clone[mask]
    if filtered_B_value.numel() > 0:    
        num_elements = filtered_B_value.numel()
        num_samples = min(N, num_elements)
        min_indices = torch.randperm(num_elements)[:num_samples]        
        cex = filtered_uns_domain[min_indices]
    else:
        cex = torch.empty((0, filtered_uns_domain.shape[1]), device=device)

    return cex.cpu()

def conformal_prediction(model_v, model_b, model_f,input_domain, init_domain, unsafe_domain, config):
    N=config ["verification"]["N_conf"]
    epsilon = config["verification"]["epsilon"]
    alpha = 0.1*epsilon
    l = math.floor((N+1)*(alpha))
    beta = sp.betainc(N - l + 1, l, 1-epsilon)
    
    device = next(model_v.parameters()).device
    model_v = model_v.to(device)
    model_b = model_b.to(device)
    model_f = model_f.to(device)
    input_domain = input_domain.float().to(device)
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)

    # Define the bounds
    lower_bound = config["counterex"]["lb"]
    upper_bound = config["counterex"]["ub"]

    # Create a boolean mask for points within the bounds
    mask_x = (input_domain_clone[:, 0] < lower_bound) | (input_domain_clone[:, 0] > upper_bound)
    mask_y = (input_domain_clone[:, 1] < lower_bound) | (input_domain_clone[:, 1] > upper_bound)
    mask = mask_x & mask_y

    # Apply the mask to filter out the points
    input_domain_clone = input_domain_clone[mask]    
    V_value = model_v(input_domain_clone)

    f_value = model_f(input_domain_clone)
    #lyapunov lie derivative counterexamples
    grad_lyap = torch.autograd.grad(
                    torch.sum(V_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_lyap = torch.sum(grad_lyap * f_value, dim=1)
    V_pos = -V_value.view(-1)

    lie_tol = config["hyperparameters"]["lie_tol"]
    B_value = model_b(input_domain_clone)
    grad_bar = torch.autograd.grad(
                    torch.sum(B_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_bar = torch.sum(grad_bar * f_value, dim=1)
    bar_mask = (torch.abs(B_value[:,0]) <= lie_tol)
    lie_bar = lie_bar[bar_mask]
    init_domain_clone = init_domain.float().to(device)
    B_init = model_b(init_domain_clone).view(-1)
    
    unsafe_domain_clone = unsafe_domain.float().to(device)
    B_uns = -model_b(unsafe_domain_clone).view(-1)
    
    #Make sure all tensors are the same size (indicator function-append 0s when not in set)
    if lie_bar.numel() < lie_lyap.numel():
        padding = lie_lyap.numel() - lie_bar.numel()
        lie_bar = torch.cat([lie_bar, torch.zeros(padding, device=lie_bar.device)])

    if B_init.numel() < lie_lyap.numel():
        padding = lie_lyap.numel() - B_init.numel()
        B_init = torch.cat([B_init, torch.zeros(padding, device=B_init.device)])

    if B_uns.numel() < lie_lyap.numel():
        padding = lie_lyap.numel() - B_uns.numel()
        B_uns = torch.cat([B_uns, torch.zeros(padding, device=B_uns.device)])
        
    #Sort the data in ascending order
    lie_lyap = torch.sort(lie_lyap, descending=False).values
    V_pos = torch.sort(V_pos, descending=False).values
    lie_bar = torch.sort(lie_bar, descending=False).values
    B_init = torch.sort(B_init, descending=False).values
    B_uns = torch.sort(B_uns, descending=False).values
    #Compute the non conformity prediction scores
    quantile_n = math.ceil((N+1)*(1-epsilon))/N
    score_lie_lyap = torch.quantile(lie_lyap,quantile_n, interpolation='higher')
    score_V_pos = torch.quantile(V_pos,quantile_n, interpolation='higher')
    score_lie_bar = torch.quantile(lie_bar, quantile_n, interpolation='higher')
    score_B_init = torch.quantile(B_init, quantile_n, interpolation='higher')
    score_B_uns = torch.quantile(B_uns, quantile_n, interpolation='higher')
    q = max(score_lie_lyap, score_V_pos, score_lie_bar, score_B_init, score_B_uns)
    return q, beta