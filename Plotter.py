from common_header import *
from cmcrameri import cm

# Plotting of the Dynamics without the Barrier
def initialDSPlot(model_f, demos, initial_set_center, dim_in, config):
    if dim_in == 2:
        device = next(model_f.parameters()).device
        # Create a figure and 2D axes
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(demos.shape[0]):
            ax.plot(demos[i].pos[0,:], demos[i].pos[1,:], 'blue', label='Training Data')
        # Plotting the final trajectory
        dt = 0.01
        n = 3000
        color = ['r','g']
        # Plotting for the Trajectories
        for i in range(initial_set_center.shape[0]):
            x = torch.zeros((n, 2))
            x[0] = initial_set_center[i].float()
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * dt
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1],color[i], label=f'Final Trajectory {i+1}')
        # # Plotting the initial set
        # initial_set_radius = config["init"]["radius"]
        # x_min = initial_set_center[0][0] - initial_set_radius
        # x_max = initial_set_center[0][0] + initial_set_radius
        # y_min = initial_set_center[0][1] - initial_set_radius
        # y_max = initial_set_center[0][1] + initial_set_radius
        # initial_set = patches.Rectangle(
        #     (x_min, y_min),  # Bottom-left corner (x_min, y_min)
        #     x_max - x_min,   # Width
        #     y_max - y_min,   # Height
        #     linewidth=2,     # Border thickness
        #     edgecolor='black',  # Border color
        #     facecolor='black', # Transparent fill
        #     alpha = 0.5, 
        #     label = "Initial Set"
        # )
        # ax.add_patch(initial_set)
        # # Plotting the Unsafe Set
        # if config["unsafe"]["shape"] == 'Rectangle':
        #     unsafe_rect_range = config["unsafe"]["range"]
        #     if "unbounded" in config["unsafe"]:
        #         flag_max_min = config["unsafe"]["max_min"]
        #         flag_xy = config["unsafe"]["unbounded"]
        #         if flag_max_min == "min" and flag_xy == "x":
        #             unsafe_rect_range[0].append(1.0)
        #         elif flag_max_min == "max" and flag_xy == "x":
        #             unsafe_rect_range[0].insert(0,-1)
        #         elif flag_max_min == "min" and flag_xy == "y":
        #             unsafe_rect_range[1].append(1.0)
        #         elif flag_max_min == "max" and flag_xy == "y":
        #             unsafe_rect_range[1].insert(0,-1)
        #     x_min = unsafe_rect_range[0][0]
        #     x_max = unsafe_rect_range[0][1]
        #     y_min = unsafe_rect_range[1][0]
        #     y_max = unsafe_rect_range[1][1]
        #     unsafe = patches.Rectangle(
        #     (x_min, y_min),  # Bottom-left corner (x_min, y_min)
        #     x_max - x_min,   # Width
        #     y_max - y_min,   # Height
        #     linewidth=2,     # Border thickness
        #     edgecolor='black',  # Border color
        #     facecolor='black', # Transparent fill
        #     alpha = 0.5, 
        #     label = "Unsafe Set"
        #     )
        #     ax.add_patch(unsafe)
        # elif config["unsafe"]["shape"] == 'Circle':
        #     center = config["unsafe"]["center"]
        #     radius = config["unsafe"]["radius"]
        #     unsafe = plt.Circle(center, radius, facecolor='black', edgecolor='black', linewidth=2, label="Unsafe Set", alpha = 0.5)
        #     ax.add_patch(unsafe)
        # elif config["unsafe"]["shape"] == 'Custom':
        #     RANGE = config["plotting"].get("range", [[-1, 1], [-1, 1]])
        #     function = config["unsafe"]["function"]
        #     function = function.replace("torch.max", "np.maximum")
        #     function = function.replace("torch.", "np.")
        #     x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
        #     y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
        #     x,y = np.meshgrid(x, y)
        #     mask = (eval(function) <= 0)
        #     plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'black', linewidths=2, label = "Unsafe Set", alpha = 0.5)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.title('Trajectories of the Dynamical System')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
    elif dim_in == 3:
        device = next(model_f.parameters()).device
        # Create a figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        for i in range(demos.shape[0]):
            ax.plot3D(demos[i].pos[0,:], demos[i].pos[1,:], demos[i].pos[2,:], 'blue')
        # Plotting the final trajectory
        n = 3000
        dt = 0.01
        color = ['r','g']
        for i in range(initial_set_center.shape[0]):
            x = torch.zeros((n, 3))
            x[0, :] = initial_set_center
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * dt
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1], x[:, 2],'red')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('Trajectories of the Dynamical System')
        plt.grid(True)
        plt.show()

# Plotting the Lyapunov Function        
def plotLyapunov(model_v, dim_in=2):
    x1 = torch.linspace(-1, 1, 50)  # 50 points from -1 to 1
    x2 = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2)  # Create a 2D grid
    # Flatten to pass into the model
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1).to(model_v.parameters().device)
    V_value = model_v(inputs).detach().numpy()
    V_value = V_value.reshape(50,50)
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, V_value, levels=50, cmap="inferno")
    plt.colorbar(label="Lyapunov ")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Lyapunov Heatmap")
    plt.show()
    
# Plotting the Barrier Function
def plotBarrier(model_b, dim_in=2):
    x1 = torch.linspace(-1, 1, 50)  # 50 points from -1 to 1
    x2 = torch.linspace(-1, 1, 50)
    X1, X2 = torch.meshgrid(x1, x2)  # Create a 2D grid
    # Flatten to pass into the model
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1).to(model_b.device)
    B_value = model_b(inputs).detach().numpy()
    B_value = B_value.reshape(50,50)
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, B_value, levels=50, cmap="inferno")
    plt.colorbar(label="Barrier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Barrier Heatmap")
    plt.show()

def lyapunovBarrierPlot(model_v, model_b, model_f, demos, config):
    device = next(model_v.parameters()).device
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
    stream = ax.streamplot(X, Y, U, V, density=2, linewidth=1, color='#bbbbbb')
    # Create proxy artist for streamplot
    arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='#bbbbbb', marker='>', markeredgewidth=2, markersize=5, label='Vector Field')

    # Contour for Lyapunov Function
    if flag_contour:
        plt.contourf(X, Y, vout[:,:,0], cmap=cm.lajolla)
    # Plot training data and final trajectory
    # Plotting the Training Data
    initial_set_center = torch.tensor(config["plotting"]["initial_conditions"])
    # X_plot = X_train
    # n = int(X_plot.shape[0]/N)
    # for i in [0,2,3,4]:
    #     ax.plot(X_plot[(i-1)*N+1:i*N,0], X_plot[(i-1)*N+1:i*N,1],color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
    for i in range(4):
        ax.plot(demos[i].pos[0,:], demos[i].pos[1,:], color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
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
        ax.plot(x[:, 0], x[:, 1],'#ff00ff', label="Learned Trajectory")

    # # Plotting the robot trajectory
    # ax.plot(x_data, y_data, "#49332b", label="Robot Trajectory")   
    
    if flag_barrier:
        plt.contour(X, Y, bout[:,:,0], levels=[0], colors='#cdebc5')
        plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='#cdebc5')
        #Create proxy artists for contours
        #contour_line_legend = mpl.lines.Line2D([0], [0], color='red')
        contour_fill_legend = mpl.patches.Patch(color='#cdebc5', label=' $ \{x \in X \mid \mathrm{B}(x) \leq 0\}$')
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
            edgecolor='red',  # Border color
            facecolor='red', # Transparent fill
            alpha = 0.5, 
            label = "Unsafe Set"
            )
            ax.add_patch(unsafe)
        elif config["unsafe"]["shape"] == 'Circle':
            # center = config["unsafe"]["center"]
            # radius = config["unsafe"]["radius"]
            unsafe_set_center = config["unsafe"]["center"]
            unsafe_set_radius = config["unsafe"]["radius"]
            if isinstance(unsafe_set_center[0], (int, float)):
                unsafe_shape = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='r', edgecolor='r', linewidth=2, label="Unsafe Set")
                ax.add_patch(unsafe_shape)
            else:
                for ind, center in enumerate(unsafe_set_center):
                    unsafe_shape = plt.Circle(center, unsafe_set_radius, facecolor='r', edgecolor='r', linewidth=2, label=f"Unsafe Set {ind+1}")
                    ax.add_patch(unsafe_shape)
            # unsafe = plt.Circle(center, radius, facecolor='r', edgecolor='r', linewidth=2, label="Unsafe Set", alpha = 0.5)
            # ax.add_patch(unsafe)
        elif config["unsafe"]["shape"] == 'Custom':
            function = config["unsafe"]["function"]
            function = function.replace("torch.max", "np.maximum")
            function = function.replace("torch.", "np.")
            x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
            y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
            x,y = np.meshgrid(x, y)
            mask = (eval(function) <= 0)
            plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'r', linewidths=2, label = "Unsafe Set", alpha = 0.5)

            
    
    # initial_set_radius = config["init"]["radius"]
    # circle1 = plt.Circle(initia, initial_set_radius, facecolor='#00ffff', edgecolor='#008080', linewidth=2, label="Initial Set")
    # ax.add_patch(circle1)

    # Equilibrium Point
    plt.plot(0, 0, marker='o', markersize=7.5, color="#000000", label="Equilibrium")


    #Adding all legends
    
    if flag_legend:
        if flag_barrier and model_b is not None:
            ax.legend(handles=[arrow_proxy, contour_fill_legend, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Demonstrated Trajectories'),
                   mpl.lines.Line2D([0], [0], color='#ff00ff', label='Learned Trajectories'), mpl.lines.Line2D([0], [0], color='#49332b', label='Robot Trajectory'), initial, unsafe,
                   mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                   bbox_to_anchor=(1.05, 1), fontsize = 8)
        else:
            ax.legend(handles=[arrow_proxy, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Demonstrated Trajectories'),
                        mpl.lines.Line2D([0], [0], color='#ff00ff', label='Learned Trajectories'), initial,
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
    plt.margins(x=0,y=0)
    plt.tight_layout()

    return fig
    
def plotObstacle(model_f, model_b, X_train, initial_set_center, config):
    initial_set_center = initial_set_center.numpy().squeeze()
    device = next(model_f.parameters()).device
    if X_train.shape[1] == 3:
        x = np.linspace(-1.2, 1.2, 50)
        y = np.linspace(-1.2, 1.2, 50)
        z = np.linspace(-1.2, 1.2, 50)
        X, Y, Z = np.meshgrid(x, y, z)
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
        # Concatenate X, Y, Z to create input data tensor
        input_data = torch.stack((X_tensor, Y_tensor, Z_tensor), dim=-1).reshape(-1, 3).to(device)
        unflatten = torch.nn.Unflatten(0, (50, 50, 50))
    elif X_train.shape[1] == 2:
        x = np.linspace(-1.2, 1.2, 50)
        y = np.linspace(-1.2, 1.2, 50)
        X, Y = np.meshgrid(x, y)
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        # Concatenate X, Y to create input data tensor
        input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
        unflatten = torch.nn.Unflatten(0, (50, 50))
    with torch.no_grad():
        F_out = model_f(input_data)
        B_out = model_b(input_data)
        vect_out = unflatten(F_out).cpu().detach().numpy()
        bout = unflatten(B_out).cpu().detach().numpy()
        U = vect_out[:,:, 0]
        V = vect_out[:,:,1]
        if X_train.shape[1] == 3:
            W = vect_out[:,:,2]
            # Create a figure and 3D axes
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            # Plot the vector field
            ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
            # Plot the initial set
            ax.scatter(initial_set_center[0], initial_set_center[1], initial_set_center[2], color='green', s=100, label='Initial Set')
            # Plot the training data
            ax.plot(X_train[:, 0], X_train[:, 1], X_train[:, 2], color='black', label='Training Data')
            # Plot the final trajectory
            n = 3000
            x = torch.zeros((n, 3))
            x[0, :] = torch.tensor(initial_set_center)
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * 0.05
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1], x[:, 2], color='green', label='Final Trajectory')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('Trajectories of the Dynamical System')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif X_train.shape[1] == 2:
            # Create a figure and 2D axes
            fig, ax = plt.subplots(figsize=(10, 8))
            # Plot the vector field
            strm = ax.streamplot(X, Y, U, V, color='blue', linewidth=1, density=2)
            # Plot the initial set
            circle = plt.Circle((initial_set_center[0], initial_set_center[1]), 0.05, color='green', label='Initial Set')
            ax.add_patch(circle)
            # Plot the training data
            ax.plot(X_train[:, 0], X_train[:, 1], color='black', label='Training Data')
            # Plot the final trajectory
            n = 3000
            x = torch.zeros((n, 2))
            x[0, :] = torch.tensor(initial_set_center)
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * 0.05
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1], color='cyan', label='Final Trajectory', markersize=5)
            plt.contour(X, Y, bout[:,:,0], levels=[0], colors='red')
            plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)
            #Create proxy artists for contours
            mpl.lines.Line2D([0], [0], color='red', label='Barrier (bout=0)')
            mpl.patches.Patch(color='green', alpha=0.5, label='Safe Set')
            unsafe_set_shape = config["unsafe"]["shape"]
            if unsafe_set_shape == "Circle":
                unsafe_set_center = config["unsafe"]["center"]
                unsafe_set_radius = config["unsafe"]["radius"]
                if isinstance(unsafe_set_center[0], (int, float)):
                    unsafe_shape = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
                    ax.add_patch(unsafe_shape)
                else:
                    for center in unsafe_set_center:
                        unsafe_shape = plt.Circle(center, unsafe_set_radius, facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
                        ax.add_patch(unsafe_shape)
                        
            elif unsafe_set_shape == "Rectangle":

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
                unsafe_shape = plt.Polygon([[unsafe_rect_range[0][0], unsafe_rect_range[1][0]], [unsafe_rect_range[0][1],unsafe_rect_range[1][0]], [unsafe_rect_range[0][1], unsafe_rect_range[1][1]], [unsafe_rect_range[0][0], unsafe_rect_range[1][1]]] , facecolor='#505050', edgecolor='#303030', linewidth=2, label="Unsafe Set")
                ax.add_patch(unsafe_shape)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            plt.title('Trajectories of the Dynamical System')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()


def finalDSPlot(model_f, model_b, initial_set_center, dim_in, config):
    if dim_in == 2:
        device = next(model_f.parameters()).device
        # Create a figure and 2D axes
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plotting the final trajectory
        dt = 0.01
        n = 3000
        color = ['r','cyan']
        # Plotting for the Trajectories
        for i in range(initial_set_center.shape[0]):
            x = torch.zeros((n, 2))
            x[0] = initial_set_center[i].float()
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * dt
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1],color[i])
        # Plotting the initial set
        initial_set_radius = config["init"]["radius"]
        x_min = initial_set_center[0][0] - initial_set_radius
        x_max = initial_set_center[0][0] + initial_set_radius
        y_min = initial_set_center[0][1] - initial_set_radius
        y_max = initial_set_center[0][1] + initial_set_radius
        initial_set = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='black', facecolor='black', alpha = 0.5, label = "Initial Set")
        ax.add_patch(initial_set)
        # Plotting the Unsafe Set
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
            RANGE = config["plotting"].get("range", [[-1, 1], [-1, 1]])
            function = config["unsafe"]["function"]
            function = function.replace("torch.max", "np.maximum")
            function = function.replace("torch.", "np.")
            x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
            y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
            x,y = np.meshgrid(x, y)
            mask = (eval(function) <= 0)
            plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'black', linewidths=2, label = "Unsafe Set", alpha = 0.5)

        # Plotting the Streamlines and Contours
        x = np.linspace(-1.2, 1.2, 50)
        y = np.linspace(-1.2, 1.2, 50)
        X, Y = np.meshgrid(x, y)
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
        # Concatenate X, Y to create input data tensor
        input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
        unflatten = torch.nn.Unflatten(0, (50, 50))
        with torch.no_grad():
            Fout = model_f(input_data)
            Bout = model_b(input_data)
            vector_field = unflatten(Fout).cpu().detach().numpy()
            bout = unflatten(Bout).cpu().detach().numpy()
            strm = ax.streamplot(X, Y, vector_field[:,:, 0], vector_field[:,:, 1], color='k', linewidth=1, density=2)
            arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='black', marker='>', markeredgewidth=2, markersize=5, label='Dyn. sys.')
            plt.contour(X, Y, bout[:,:,0], levels=[0], colors='green')
            plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)
            #Create proxy artists for contours
            contour_line_legend = mpl.lines.Line2D([0], [0], color='red', label='Barrier (bout=0)')
            contour_fill_legend = mpl.patches.Patch(color='green', alpha=0.5, label='Invariant Set')
        ax.legend(handles=[arrow_proxy, contour_line_legend, contour_fill_legend, mpl.lines.Line2D([0], [0], color='#1F75FE', label='Actual Trajectory'),
                   mpl.lines.Line2D([0], [0], color='#ff00ff', label='Target Trajectory'),
                   mpl.lines.Line2D([0], [0], marker='o', color='black', label='Equilibrium')], loc='upper left', edgecolor='black', facecolor='white', framealpha = 1,
                   bbox_to_anchor=(1.05, 1), fontsize = 8)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        plt.title('Trajectories of the Dynamical System')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()