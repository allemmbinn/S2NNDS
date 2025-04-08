from common_header import *
from cmcrameri import cm


def initialDSPlot(model_f, X_train, initial_set_center, dt):
    device = next(model_f.parameters()).device
    # Plotting the Training Data after Warm-starting
    fig, ax = plt.subplots()
    X = X_train
    N = 1000
    n = int(X.shape[0]/N)
    for i in range(5):
        ax.plot(X[(i-1)*N+1:i*N,0], X[(i-1)*N+1:i*N,1],"b")
    # Plotting the final trajectory
    n = 3000
    color = ['r','g']
    for i in range(initial_set_center.shape[0]):
        x = torch.zeros((n, 2))
        #x[0] = torch.tensor(initial_set_center[i], dtype=torch.float32)
        x[0] = initial_set_center[i].float()
        x = x.to(device)
        for j in range(1, n):
            Fout = model_f(x[j-1])
            x[j] = x[j-1] + Fout * dt
        x = x.cpu().detach().numpy()
        ax.plot(x[:, 0], x[:, 1],color[i])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of the Dynamical System')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return fig

def lyapunovBarrierPlot(model_v, model_b, model_f, X_train, config):
    device = next(model_v.parameters()).device
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
        V_out = model_v(input_data)
        F_out = model_f(input_data)
        vect_out = unflatten(F_out)
        vect_out = vect_out.cpu().detach().numpy()
        U = vect_out[:,:, 0]
        V = vect_out[:,:,1]
        vout = unflatten(V_out).cpu().detach().numpy()
        if flag_barrier and model_b is not None:
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
    initial_set_center = torch.tensor(config["plotting"]["initial_conditions"])
    X_plot = X_train
    n = int(X_plot.shape[0]/N)
    for i in [0,2,3,4]:
        ax.plot(X_plot[(i-1)*N+1:i*N,0], X_plot[(i-1)*N+1:i*N,1],color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
    # Plotting the final trajectory
    n = 10000
    dt= config["plotting"]["dt"]
    for i in range(initial_set_center.shape[0]):
        x = torch.zeros((n, 2)).to(device)
        x[0,:] = torch.tensor(initial_set_center[i], dtype=torch.float32)
        #x[0] = torch.tensor([1, 0.5], dtype=torch.float32)
        for j in range(1, n):
            Fout = model_f(x[j-1])
            x[j] = x[j-1] + Fout * dt
        x = x.cpu().detach().numpy()
        ax.plot(x[:, 0], x[:, 1],'#ff00ff', label="Target Trajectory")
    
    if flag_barrier:
        plt.contour(X, Y, bout[:,:,0], levels=[0], colors='green')
        plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)
        #Create proxy artists for contours
        contour_line_legend = mpl.lines.Line2D([0], [0], color='red', label='Barrier (bout=0)')
        contour_fill_legend = mpl.patches.Patch(color='green', alpha=0.5, label='Invariant Set')
        #unsafe_set_center = config["unsafe"]["centre"]
        #unsafe_set_radius = config["unsafe"]["radius"]
        #circle2 = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
        #ax.add_patch(circle2)
        
        # Plotting the Initial Set
        init_range = config["plotting"]["init_range"]
        x_min = init_range[0][0]
        x_max = init_range[0][1]
        y_min = init_range[1][0]
        y_max = init_range[1][1]
        initial = patches.Rectangle(
        (x_min, y_min),  # Bottom-left corner (x_min, y_min)
        x_max - x_min,   # Width
        y_max - y_min,   # Height
        linewidth=2,     # Border thickness
        edgecolor='cyan',  # Border color
        facecolor='cyan',   # Transparent fill
        label="Initial Set"
        )

        ax.add_patch(initial)

        if config["unsafe"]["shape"] == 'Rectangle':
            unsafe_rect_range = config["unsafe"]["range"]
            if "unbounded" in config["unsafe"]:
                flag_max_min = config["unsafe"]["max_min"]
                flag_xy = config["unsafe"]["unbounded"]
                if flag_max_min == "min" and flag_xy == "x":
                    unsafe_rect_range[0].append(1.0)
                elif flag_max_min == "max" and flag_xy == "x":
                    unsafe_rect_range[0].insert(0,-1)
                elif flag_max_min == "min" and flag_xy == "y":
                    unsafe_rect_range[1].append(1.0)
                elif flag_max_min == "max" and flag_xy == "y":
                    unsafe_rect_range[1].insert(0,-1)
            x_min = unsafe_rect_range[0][0]
            x_max = unsafe_rect_range[0][1]
            y_min = unsafe_rect_range[1][0]
            y_max = unsafe_rect_range[1][1]
            unsafe = patches.Rectangle(
            (x_min, y_min),  # Bottom-left corner (x_min, y_min)
            x_max - x_min,   # Width
            y_max - y_min,   # Height
            linewidth=2,     # Border thickness
            edgecolor='black',  # Border color
            facecolor='black', # Transparent fill
            alpha = 0.5, 
            label = "Unsafe Set"
            )
            ax.add_patch(unsafe)
        elif config["unsafe"]["shape"] == 'Circle':
            center = config["unsafe"]["center"]
            radius = config["unsafe"]["radius"]
            unsafe = plt.Circle(center, radius, facecolor='black', edgecolor='black', linewidth=2, label="Unsafe Set", alpha = 0.5)
            ax.add_patch(unsafe)
        elif config["unsafe"]["shape"] == 'Custom':
            function = config["unsafe"]["function"]
            x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
            y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
            x,y = np.meshgrid(x, y)
            mask = (eval(function) <= 0)
            plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'black', linewidths=2, label = "Unsafe Set", alpha = 0.5)

            
    
    # initial_set_radius = config["init"]["radius"]
    # circle1 = plt.Circle(initia, initial_set_radius, facecolor='#00ffff', edgecolor='#008080', linewidth=2, label="Initial Set")
    # ax.add_patch(circle1)

    # Equilibrium Point
    plt.plot(0, 0, marker='o', markersize=7.5, color="#000000", label="Equilibrium")


    #Adding all legends
    
    if flag_legend:
        if flag_barrier and model_b is not None:
            ax.legend(handles=[arrow_proxy, contour_line_legend, contour_fill_legend, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
                   mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), initial, unsafe,
                   mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                   bbox_to_anchor=(1.05, 1), fontsize = 8)
        else:
            ax.legend(handles=[arrow_proxy, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
                        mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), initial,
                        mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                        bbox_to_anchor=(1.05, 1), fontsize = 8)

        # Setting labels and grid
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_xlim(RANGE[0][0], RANGE[0][1])
    plt.gca().set_ylim(RANGE[1][0], RANGE[1][1])
    dataset = config["plotting"]["name"]
    plt.title(dataset)
    plt.grid(True)
    plt.axis('auto')
    plt.tight_layout()

    return fig




# def plotObstacle(model_f, model_b, X_train, mean_point, config):
#     mean_point = mean_point.numpy().squeeze()
#     device = next(model_f.parameters()).device
#     N = 1000
#     fig, ax = plt.subplots()
#     # Define grid for plotting
#     RANGE = config["plotting"]["range"]
#     flag_legend = config["plotting"]["legend"]
    
#     len_sample = [128, 128]
#     x = np.linspace(RANGE[0][0], RANGE[0][1], len_sample[0])
#     y = np.linspace(RANGE[1][0], RANGE[1][1], len_sample[1])
#     X, Y = np.meshgrid(x, y)

#     # Convert X and Y to torch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
#     Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

#     # Concatenate X and Y to create input data tensor
#     input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
#     unflatten = torch.nn.Unflatten(0, len_sample)

#     # Streamplot
#     with torch.no_grad():
#         F_out = model_f(input_data)
#         vect_out = unflatten(F_out)
#         vect_out = vect_out.cpu().detach().numpy()
#         U = vect_out[:,:, 0]
#         V = vect_out[:,:,1]
#         if model_b is not None:
#             B_out = model_b(input_data)
#             bout = unflatten(B_out).cpu().detach().numpy()
#     stream = ax.streamplot(X, Y, U, V, density=2, linewidth=1, color='#000000')
#     # Create proxy artist for streamplot
#     arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='black', marker='>', markeredgewidth=2, markersize=5, label='Dyn. sys.')

#     # Plot training data and final trajectory
#     # Plotting the Training Data
#     X_plot = X_train
#     n = int(X_plot.shape[0]/N)
#     for i in [0,2,3,4]:
#         ax.plot(X_plot[(i-1)*N+1:i*N,0], X_plot[(i-1)*N+1:i*N,1],color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
#     # Plotting the final trajectory
#     n = 10000
#     x = torch.zeros((n, 2)).to(device)
#     x[0,:] = torch.tensor(mean_point, dtype=torch.float32)
#     #x[0] = torch.tensor([1, 0.5], dtype=torch.float32)
#     dt = 0.02
#     for j in range(1, n):
#         Fout = model_f(x[j-1])
#         x[j] = x[j-1] + Fout * dt
#     x = x.cpu().detach().numpy()
#     ax.plot(x[:, 0], x[:, 1],'#ff00ff', label="Target Trajectory")
    
#     if model_b is not None:
#         contour_lines = plt.contour(X, Y, bout[:,:,0], levels=[0], colors='red')
#         contour_fills = plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)
#         #Create proxy artists for contours
#         contour_line_legend = mpl.lines.Line2D([0], [0], color='red', label='Barrier (bout=0)')
#         contour_fill_legend = mpl.patches.Patch(color='green', alpha=0.5, label='Safe Set')
#         unsafe_set_shape = config["unsafe"]["shape"]
#         if unsafe_set_shape == "Circle":
#             unsafe_set_center = config["unsafe"]["centre"]
#             unsafe_set_radius = config["unsafe"]["radius"]
#             unsafe_shape = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
#         elif unsafe_set_shape == "Rectangle":

#             unsafe_rect_range = config["unsafe"]["range"]
#             if "unbounded" in config["unsafe"]:
#                 flag_max_min = config["unsafe"]["max_min"]
#                 flag_xy = config["unsafe"]["unbounded"]
#                 if flag_max_min == "min" and flag_xy == "x":
#                     unsafe_rect_range[0].append(1.0)
#                 elif flag_max_min == "max" and flag_xy == "x":
#                     unsafe_rect_range[0].insert(0,-1)
#                 elif flag_max_min == "min" and flag_xy == "y":
#                     unsafe_rect_range[1].append(1.0)
#                 elif flag_max_min == "max" and flag_xy == "y":
#                     unsafe_rect_range[1].insert(0,-1) 
#             unsafe_shape = plt.Polygon([[unsafe_rect_range[0][0], unsafe_rect_range[1][0]], [unsafe_rect_range[0][1],unsafe_rect_range[1][0]], [unsafe_rect_range[0][1], unsafe_rect_range[1][1]], [unsafe_rect_range[0][0], unsafe_rect_range[1][1]]] , facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
#         ax.add_patch(unsafe_shape)
#     # Plotting the Initial and Unsafe Set
#     initial_set_radius = config["init"]["radius"]
#     # TODO: Add a provision for circle and rectangle
#     init_shape = plt.Rectangle((mean_point[0], mean_point[1]), initial_set_radius, initial_set_radius, facecolor='#00ffff', edgecolor='#008080', linewidth=2, label="Initial Set")
#     ax.add_patch(init_shape)
#     # circle1 = plt.Circle(mean_point, initial_set_radius, facecolor='#00ffff', edgecolor='#008080', linewidth=2, label="Initial Set")
#     # ax.add_patch(circle1)

#     # Equilibrium Point
#     plt.plot(0, 0, marker='o', markersize=7.5, color="#000000", label="Equilibrium")

#     # Setting labels and grid
#     plt.xlabel('x')
#     plt.ylabel('y')
#     dataset = config["plotting"]["name"]
#     plt.title(dataset)
#     plt.grid(True)
#     plt.axis('equal')

#     #Adding all legends
    
#     if flag_legend:
#         if model_b is not None:
#             ax.legend(handles=[arrow_proxy, contour_line_legend, contour_fill_legend, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
#                    mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), init_shape, unsafe_shape,
#                    mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
#                    bbox_to_anchor=(0, 1))
#         else:
#             ax.legend(handles=[arrow_proxy, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
#                         mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'), init_shape,
#                         mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
#                         bbox_to_anchor=(0, 1))
#     plt.show()


def plotLyapunov(model_v):
    x1 = torch.linspace(-1, 1, 50)  # 50 points from -1 to 1
    x2 = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2)  # Create a 2D grid
    # Flatten to pass into the model
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    V_value = model_v(inputs).detach().numpy()
    V_value = V_value.reshape(50,50)
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, V_value, levels=50, cmap="inferno")
    plt.colorbar(label="Lyapunov ")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Lyapunov Heatmap")
    plt.show()

def plotBarrier(model_b):
    x1 = torch.linspace(-1, 1, 50)  # 50 points from -1 to 1
    x2 = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2)  # Create a 2D grid
    # Flatten to pass into the model
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    B_value = model_b(inputs).detach().numpy()
    B_value = B_value.reshape(50,50)
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, B_value, levels=50, cmap="inferno")
    plt.colorbar(label="Barrier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Barrier Heatmap")
    plt.show()
