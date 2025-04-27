from common_header import *

def loss_function_dyn(model_f, X_train, y_train, config):
    device = next(model_f.parameters()).device
    X_train_clone = X_train.float().to(device)
    y_train_clone = y_train.float().to(device)
    # MSE Loss
    loss_fn = nn.MSELoss(reduction='mean')
    loss_MSE = loss_fn(model_f(X_train_clone),y_train_clone)
    # Hyperparameter
    DECAY_MSE = config["hyperparameters"]["decay_mse"]
    loss_mse = DECAY_MSE*loss_MSE
    return loss_mse

def loss_function_domain(model_v, model_b, model_f, input_domain, config):
    alpha = config["hyperparameters"]["alpha"] #Parameter for leaky relu
    act= F.elu
    device = next(model_v.parameters()).device
    model_v = model_v.to(device)
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
    grad_bar = torch.autograd.grad(
                    torch.sum(B_value),
                    input_domain_clone,
                    grad_outputs=None,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
    lie_barr = torch.sum(grad_bar * f_value, dim=1)
    #Getting the hyperparameters
    lyap_tol = config["hyperparameters"]["lyap_tol"]
    bar_tol = config["hyperparameters"]["bar_tol"]
    pos_tol = config["hyperparameters"]["lyap_tol"]   
    DECAY_V0 = config["hyperparameters"]["decay_v0"]
    DECAY_VPOS = config["hyperparameters"]["decay_vpos"]
    DECAY_LV = config["hyperparameters"]["decay_lv"]
    DECAY_LB = config["hyperparameters"]["decay_lb"]
    #Invidual weighted losses
    loss_zero = DECAY_V0 * (V_0).pow(2)
    loss_lie_v  = DECAY_LV * (act(lie_lyap + lyap_tol, alpha)).mean()
    loss_lie_b = DECAY_LB * (act(lie_barr + bar_tol, alpha)).mean() 
    loss_vpos = DECAY_VPOS * (act(pos_tol  - V_value, alpha)).mean()
    return loss_lie_v + loss_vpos + loss_zero, loss_lie_b

def loss_function_init(model_b, input_init, config):
    alpha = config["hyperparameters"]["alpha"]
    act= F.elu
    device = next(model_b.parameters()).device
    input_init = input_init.float().to(device)
    # For the init set
    B_init = model_b(input_init)
    # Finding the Lie Derivative for entire Domain
    # Hyperparameters
    tol = config["hyperparameters"]["inun_tol"] 
    DECAY_INIT = config["hyperparameters"]["decay_init"]
    # Total Loss
    loss_init = DECAY_INIT *(act(B_init + tol,alpha)).mean()
    return  loss_init

def loss_function_unsafe(model_b, input_unsafe, config):
    alpha = config["hyperparameters"]["alpha"]
    act= F.elu
    device = next(model_b.parameters()).device
    input_unsafe = input_unsafe.float().to(device)
    # For the unsafe set
    B_unsafe = model_b(input_unsafe)
    #hyperparameters
    tol = config["hyperparameters"]["inun_tol"] 
    DECAY_UNSAFE = config["hyperparameters"]["decay_unsafe"]
    #losses
    loss_unsafe =  DECAY_UNSAFE * (act(- B_unsafe + 0.001 - tol, alpha)).mean()
    return loss_unsafe