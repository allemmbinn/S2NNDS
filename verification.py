from common_header import *
from torch.utils.data import DataLoader, TensorDataset
import scipy.special as sc
def bounds(model):
    x1 = torch.linspace(-1, 1, 50)  # 50 points from -1 to 1
    x2 = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2)  # Create a 2D grid
    # Flatten to pass into the model
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1).requires_grad_()
    Value = model(inputs)
    grad= torch.autograd.grad(
                    torch.sum(Value),
                    inputs,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    return torch.max(torch.linalg.norm(grad,2,dim=1))

def lipschitz_network(weights):
    n= 2
    m = len(weights)
    if m == 0:
        return 1
    if m == 1:
        return torch.linalg.norm(weights[0],n)
    else:
        norm = 0
        i = m-1
        weight = torch.eye(weights[m-1].shape[0])
        while i >= 0:
            weight = torch.matmul(weight, weights[i])
            if i != 0:
                norm += 1/2**(m-i)*torch.linalg.norm(weight,n)*lipschitz_network(weights[0:i])
            if i == 0:
                norm += 1/2**(m-1)*torch.linalg.norm(weight,n)*lipschitz_network(weights[0:i])
            i -= 1
        return norm.item()

def lipschitz_gradient(weights):
    n= 2
    m= len(weights)
    if m == 1:
        return torch.linalg.norm(weights[0]**2,n)
    else:
        norm = 0
        weight_m = weights[m-1]
        norm += torch.linalg.norm(weight_m,n) *lipschitz_network(weights[0:m-1])
        for i in range(m-2, -1, -1):
            weight_m = torch.matmul(weight_m,weights[i])
        norm += torch.linalg.norm(weight_m,n)*lipschitz_network(weights)

        return norm.item()


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

    #get the last minimum values of lie_lyap
    if filtered_lie_lyap.numel() > 0:  # Ensure there are valid values
        _, min_indices = torch.topk(filtered_lie_lyap, k=min(N, filtered_lie_lyap.numel()), largest=True)
        lie_lyap_cex = filtered_input_domain[min_indices]
    else:
        lie_lyap_cex = torch.empty((0, input_domain.shape[1]), device=device)

    
    #lyapunov negative value counterexamples
    pos_tol = config["counterex"]["pos_tol"]
    pos_mask = V_value[:,0] < pos_tol
    filtered_V_Value = V_value[pos_mask].view(-1)
    filtered_input_domain = input_domain_clone[pos_mask]
    
    #get the last minimum values of V_pos
    if filtered_V_Value.numel() > 0:  # Ensure there are valid values
        _, min_indices = torch.topk(filtered_V_Value, k=min(N, filtered_V_Value.numel()), largest=False)
        V_pos_cex = filtered_input_domain[min_indices]
    else:
        V_pos_cex = torch.empty((0, input_domain.shape[1]), device=device)

    #barrier lie derivative counterexamples
    bar_tol =  config["hyperparameters"]["bar_tol"]
    lie_tol = config["hyperparameters"]["lie_tol"]
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
    B_value = model_b(input_domain_clone)
    f_value = model_f(input_domain_clone)
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

    #get the maximum values of lie_bar
    if filtered_lie_bar.numel() > 0:  # Ensure there are valid values
        _, min_indices = torch.topk(filtered_lie_bar, k=min(N, filtered_lie_bar.numel()), largest=True)
        lie_bar_cex = filtered_input_domain[min_indices]
    else:
        lie_bar_cex = torch.empty((0, input_domain.shape[1]), device=device)

    #return counterexamples
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
    if filtered_B_value.numel() > 0:  # Ensure there are valid values
        _, min_indices = torch.topk(filtered_B_value, k=min(N, filtered_B_value.numel()), largest=True)
        cex = filtered_init_domain[min_indices]
    else:
        cex = torch.empty((0, init_domain.shape[1]), device=device)

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
    if filtered_B_value.numel() > 0:  # Ensure there are valid values
        _, min_indices = torch.topk(filtered_B_value, k=min(N, filtered_B_value.numel()), largest=False)
        cex = filtered_uns_domain[min_indices]
    else:
        cex = torch.empty((0, unsafe_domain.shape[1]), device=device)

    return cex.cpu()

def conformal_prediction(model_v, model_b, model_f,input_domain, init_domain, unsafe_domain, config):
    N=config ["verification"]["N_conf"]**2
    epsilon = config["verification"]["epsilon"]
    alpha = 0.01*epsilon
    l = math.floor((N+1)*(alpha))
    beta = sc.betainc(N - l + 1, l, 1-epsilon)
    #Defining the conformal score functions
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
    V_pos = -V_value

    lie_tol = config["hyperparameters"]["lie_tol"]
    B_value = model_b(input_domain_clone)
    bar_mask = (torch.abs(B_value[:,0]) <= lie_tol)
    B_value = B_value[bar_mask]
    grad_bar = torch.autograd.grad(
                    torch.sum(B_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_bar = torch.sum(grad_bar * f_value, dim=1)

    init_domain_clone = init_domain.float().to(device)
    B_init = model_b(init_domain_clone)
    
    unsafe_domain_clone = unsafe_domain.float().to(device)
    B_uns = -model_b(unsafe_domain_clone)

    #Compute the non conformity prediction scores
    quantile_n = math.ceil((N+1)*(1-epsilon))/N
    score_lie_lyap = torch.quantile(lie_lyap,quantile_n, interpolation='lower')
    score_V_pos = torch.quantile(V_pos,quantile_n, interpolation='lower')
    score_lie_bar = torch.quantile(lie_bar, quantile_n, interpolation='lower')
    score_B_init = torch.quantile(B_init, quantile_n, interpolation='lower')
    score_B_uns = torch.quantile(B_uns, quantile_n, interpolation='lower')
    q = max(score_lie_lyap, score_V_pos, score_lie_bar, score_B_init, score_B_uns)
    return beta, q