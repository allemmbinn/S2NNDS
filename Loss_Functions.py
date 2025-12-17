from common_header import *

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
    # Get Dimensions
    dim_in = input_domain.shape[1]
    #Lyapunov Losses
    # Zero set
    x_0 = torch.zeros([1, dim_in]).to(device)
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
    # Tuning the Lyapunov function to be elliptic
    Circle_Tuning = Tune(input_domain).to(device)    #Lie derivative of barrier 
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
    hyperparameters_config = config["hyperparameters"]
    lyap_tol = hyperparameters_config.get("lyap_tol", 0.01) 
    bar_tol = hyperparameters_config.get("bar_tol", 0.01)
    pos_tol = hyperparameters_config.get("pos_tol", 0.01)
    DECAY_V0 = hyperparameters_config.get("decay_v0", 1.0)
    DECAY_VPOS = hyperparameters_config.get("decay_vpos", 1.0)
    DECAY_LV = hyperparameters_config.get("decay_lv", 1.0)
    DECAY_LB = hyperparameters_config.get("decay_lb", 1.0)
    DECAY_TUNE = hyperparameters_config.get("decay_tune", 0.0)
    #Invidual weighted losses
    loss_zero = DECAY_V0 * (V_0).pow(2)
    loss_lie_v  = DECAY_LV * (act(lie_lyap + lyap_tol, alpha)).mean()
    loss_lie_b = DECAY_LB * (act(lie_barr + bar_tol, alpha)).mean() 
    loss_vpos = DECAY_VPOS * (act(pos_tol  - V_value, alpha)).mean()
    loss_tune = DECAY_TUNE * (Circle_Tuning).pow(2).mean()
    return loss_lie_v + loss_vpos + loss_zero, loss_lie_b + loss_tune

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