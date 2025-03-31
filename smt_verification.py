from common_header import *
from dreal import *
   
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

def CheckLyapunov(x, f, V, V_dot, ball_lb, ball_ub, config, epsilon):
    ball= Expression(0)
    lie_derivative_of_V = Expression(0)
    for i in range(len(x)):
        ball += x[i]*x[i]
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)
    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)
    condition = logical_imply(ball_in_bound, V >= 0)
    result = CheckSatisfiability(logical_not(condition),config)
    if(result):
        print_warning("Not Satisfied for Positive Definteness. Counterexamples: ")
        print_warning(result)
        return result
    else:
        condition = logical_imply(ball_in_bound, V_dot <= 0)
        result = CheckSatisfiability(logical_not(condition),config)
        if result:
            print_warning("Not Satisfied for Derivative of Lyapunov Function. Counterexamples:")
            print_warning(result)
        return result
