from common_header import *
from cmcrameri import cm

# Load the configuration file
config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file) 

device = config["device"]

def initialDSPlot(model_f, X_train, initial_set_center):
    # Plotting the Training Data after Warm-starting
    fig, ax = plt.subplots()
    X = X_train
    N = 1000
    n = int(X.shape[0]/N)
    for i in range(5):
        ax.plot(X[(i-1)*N+1:i*N,0], X[(i-1)*N+1:i*N,1],"b")
    # Plotting the final trajectory
    n = 3000
    x = torch.zeros((n, 2))
    x[0] = torch.tensor(initial_set_center, dtype=torch.float32)
    x = x.to(device)
    dt = 0.02
    for j in range(1, n):
        Fout = model_f(x[j-1])
        x[j] = x[j-1] + Fout * dt
    x = x.cpu().detach().numpy()
    ax.plot(x[:, 0], x[:, 1],'r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of the Dynamical System')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def lyapunovBarrierPlot(model_v, X_train, mean_point, model_b = None):
    N = 1000
    fig, ax = plt.subplots()
    # Define grid for plotting
    RANGE = config["plotting"]["range"]
    flag_barrier = config["Barrier"]
    flag_contour = config["plotting"]["contour"]
    flag_legend = config["plotting"]["legend"]
    
    len_sample = [128, 128]
    x = np.linspace(RANGE[0][0], RANGE[0][1], len_sample[0])
    y = np.linspace(RANGE[1][0], RANGE[1][1], len_sample[1])
    X, Y = np.meshgrid(x, y)

    # Convert X and Y to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    # Concatenate X and Y to create input data tensor
    input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
    unflatten = torch.nn.Unflatten(0, len_sample)

    # Streamplot
    with torch.no_grad():
        V_out,F_out = model_v(input_data)
        vect_out = unflatten(F_out)
        vect_out = vect_out.cpu().detach().numpy()
        U = vect_out[:,:, 0]
        V = vect_out[:,:,1]
        vout = unflatten(V_out).cpu().detach().numpy()
        if flag_barrier:
            B_out = model_b(input_data)
            bout = unflatten(B_out).cpu().detach().numpy()
    stream = ax.streamplot(X, Y, U, V, density=2, linewidth=1, color='#000000')
    # Create proxy artist for streamplot
    arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='black', marker='>', markeredgewidth=2, markersize=5, label='Dyn. sys.')

    # Contour for Lyapunov Function
    if flag_contour:
        plt.contourf(X, Y, vout[:,:,0], cmap=cm.lajolla)
    # Plot training data and final trajectory
    # Plotting the Training Data
    X_plot = X_train
    n = int(X_plot.shape[0]/N)
    for i in [0,2,3,4]:
        ax.plot(X_plot[(i-1)*N+1:i*N,0], X_plot[(i-1)*N+1:i*N,1],color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
    # Plotting the final trajectory
    n = 10000
    x = torch.zeros((n, 2)).to(device)
    x[0,:] = torch.tensor(mean_point, dtype=torch.float32)
    #x[0] = torch.tensor([1, 0.5], dtype=torch.float32)
    dt = 0.02
    for j in range(1, n):
        Vout, Fout = model_v(x[j-1])
        x[j] = x[j-1] + Fout * dt
    x = x.cpu().detach().numpy()
    ax.plot(x[:, 0], x[:, 1],'#ff00ff', label="Target Trajectory")
    
    if flag_barrier:
        contour_lines = plt.contour(X, Y, bout[:,:,0], levels=[0], colors='red')
        contour_fills = plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)
        #Create proxy artists for contours
        contour_line_legend = mpl.lines.Line2D([0], [0], color='red', label='Barrier (bout=0)')
        contour_fill_legend = mpl.patches.Patch(color='green', alpha=0.5, label='Safe Set')
        unsafe_set_center = config["unsafe"]["centre"]
        unsafe_set_radius = config["unsafe"]["radius"]
        circle2 = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
        ax.add_patch(circle2)
    # Plotting the Initial and Unsafe Set
    initial_set_radius = config["init"]["radius"]
    circle1 = plt.Circle(mean_point, initial_set_radius, facecolor='#00ffff', edgecolor='#008080', linewidth=2, label="Initial Set")
    ax.add_patch(circle1)

    # Equilibrium Point
    plt.plot(0, 0, marker='o', markersize=7.5, color="#000000", label="Equilibrium")

    # Setting labels and grid
    plt.xlabel('x')
    plt.ylabel('y')
    dataset = config["plotting"]["name"]
    plt.title(dataset)
    plt.grid(True)
    plt.axis('equal')

    #Adding all legends
    
    if flag_legend:
        if flag_barrier:
            ax.legend(handles=[arrow_proxy, contour_line_legend, contour_fill_legend, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
                   mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), circle1, circle2,
                   mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                   bbox_to_anchor=(0, 1))
        else:
            ax.legend(handles=[arrow_proxy, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
                        mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), circle1,
                        mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                        bbox_to_anchor=(0, 1))
    plt.show()