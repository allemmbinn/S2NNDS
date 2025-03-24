from common_header import *
from dreal import *
from z3 import *
   
def hyper_tan_dr(x):
    y = x.copy()
    for idx in range(len(y)):
        y[idx] = tanh(y[idx])
    return y

def hyper_tan_der_dr(x):
    y = x.copy()
    for idx in range(len(y)):
        y[idx] = 1 / pow(cosh(y[idx]), 2)
    return y

def hyper_relu_dr(x):
    y = x.copy()
    RELU = lambda x: Max(x, 0)
    simplify = lambda x: x
    for idx in range(len(y)):
        y[idx] = simplify(RELU(y[idx]))
    return y

def hyper_relu_der_dr(x):
    y = x.copy()
    simplify = lambda x: x
    for idx in range(len(y)):
        y[idx] = simplify(if_then_else(x[idx] > 0, 1, 0))
    return y

def hyper_relu_dr_z3(x):
    y = x.copy()
    RELU = lambda x: z3.If(x > 0, x, 0)
    simplify = z3.simplify
    for idx in range(len(y)):
        y[idx] = simplify(RELU(y[idx]))
    return y

def piecewise_tanh_z3(x):
    y=x.copy()
    for idx in range(len(y)):
        y[idx] = simplify(If(y[idx] < -1, -1,
                 If(y[idx] < 0, y[idx],
                    If(y[idx] < 1, y[idx],
                       1))))
    return y

def hyper_relu_der_z3(x):
    y = x.copy()
    simplify = z3.simplify
    for idx in range(len(y)):
        y[idx] = simplify(z3.If(x[idx] > 0, 1, 0))
    return y

def AddCounterexamples(x,CE,N):
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.append(np.random.uniform(lb,ub,N-1),c)
        nearby.append(nearby_)
    for i in range(N):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])
        x = torch.cat((x, torch.tensor([n_pt])), 0)
    return x

def CheckLyapunov(x, V, V_dot, ball_lb, ball_ub, config):
    ball= Expression(0)
    for i in range(len(x)):
        ball += x[i]*x[i]
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)
    condition_pos = logical_imply(ball_in_bound, V >= 0)
    condition_der = logical_imply(ball_in_bound, V_dot <= 0)
    condition = logical_and(condition_pos, condition_der)
    result = CheckSatisfiability(logical_not(condition),config)
    if(result):
        print_warning("SMT Verification for Lyapunov failed... Generating Counterexamples:" )
        print_warning(result)
        return result
    else:
        print_info("SMT Verification for Lyapunov successful")
        return result
