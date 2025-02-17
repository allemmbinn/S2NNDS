from common_header import *
from torch.utils.data import DataLoader, TensorDataset

def verify_domain(model_v, model_b, model_f, input_domain, config):
    device = next(model_v.parameters()).device
    model_v = model_v.to(device)
    model_b = model_b.to(device)
    model_f = model_f.to(device)
    input_domain = input_domain.float().to(device)
    input_domain_clone = torch.clone(input_domain).requires_grad_().to(device)
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
    mask = lie_lyap > 0
    true_indices_lie_lyap = torch.nonzero(mask, as_tuple=False).squeeze()
    #lyapunov negative value counterexamples
    mask = V_value[:,0] < 0
    true_indices_pos = torch.nonzero(mask, as_tuple=False).squeeze()
    #barrier lie derivative counterexamples
    lie_tol = config["hyperparameters"]["lie_tol"]
    input_domain = input_domain.float().to(device)
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
    mask = (torch.abs(B_value[:,0]) <= lie_tol) & (lie_bar > 0)
    true_indices_lie_bar = torch.nonzero(mask, as_tuple=False).squeeze()
 # Filter out empty tensors and ensure they have at least one dimension
    indices_list = [true_indices_lie_lyap, true_indices_pos, true_indices_lie_bar]
    indices_list = [indices.unsqueeze(0) if indices.dim() == 0 else indices for indices in indices_list if indices.numel() > 0]
    # Get the union of all the indices
    if indices_list:
        true_indices_union = torch.unique(torch.cat(indices_list))
        # Get the input_domain values corresponding to the true indices union
        counterexamples = input_domain[true_indices_union]
    else:
        counterexamples = torch.empty((0, input_domain.shape[1]), device=device)

    return counterexamples

def verify_init(model_b,init_domain):
    device = next(model_b.parameters()).device
    # barrier initial set counterexamples
    init_domain = init_domain.float().to(device)
    B_value = model_b(init_domain)
    mask = B_value[:,0] > 0
    true_indices_init = torch.nonzero(mask, as_tuple=False).squeeze()
    counterexamples = init_domain[true_indices_init]
    return counterexamples

def verify_unsafe(model_b, unsafe_domain):
    device = next(model_b.parameters()).device
    #barrier unsafe set counterexamples
    unsafe_domain = unsafe_domain.float().to(device)
    B_value = model_b(unsafe_domain)
    mask = B_value[:,0] <= 0
    true_indices_unsafe = torch.nonzero(mask, as_tuple=False).squeeze()
    counterexamples = unsafe_domain[true_indices_unsafe]
    return counterexamples


        






