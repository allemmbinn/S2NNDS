from common_header import *
from cmcrameri import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Plotting of the Dynamics without the Barrier
def initialDSPlot(model_f, demos, initial_set_center, dim_in, config, model_b=None):
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
        # Plotting the Unsafe Set
        unsafe_config = config["unsafe"]
        unsafe_shape = unsafe_config["shape"]
        if unsafe_shape == 'Rectangle':
            unsafe_rect_range = unsafe_config.get("range", [[-1, 1], [-1, 1]])
            if "unbounded" in unsafe_config:
                flag_max_min = unsafe_config.get("max_min")
                flag_xy = unsafe_config.get("unbounded")
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
            (x_min, y_min),  
            x_max - x_min,   
            y_max - y_min,   
            linewidth=2,     
            edgecolor='black',  
            facecolor='black',
            alpha = 0.5, 
            label = "Unsafe Set"
            )
            ax.add_patch(unsafe)
        elif unsafe_shape == 'Circle':
            center = unsafe_config.get("center")
            radius = unsafe_config.get("radius", 0.01)
            if isinstance(center[0], (int, float)):
                unsafe = plt.Circle(center, radius, facecolor='black', edgecolor='black', linewidth=2, label="Unsafe Set", alpha = 0.5)
                ax.add_patch(unsafe)
            else:
                for c in center:
                    unsafe = plt.Circle(c, radius, facecolor='black', edgecolor='black', linewidth=2, label="Unsafe Set", alpha = 0.5)
                    ax.add_patch(unsafe)
        elif unsafe_shape == 'Custom':
            RANGE = config["plotting"].get("range", [[-1, 1], [-1, 1]])
            function = config["unsafe"]["function"]
            function = function.replace("torch.max", "np.maximum")
            function = function.replace("torch.", "np.")
            x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
            y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
            x,y = np.meshgrid(x, y)
            mask = (eval(function) <= 0)
            plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'black', linewidths=2, label = "Unsafe Set", alpha = 0.5)
        # Plotting the Safe Region
        if model_b is not None:
            len_sample = [128, 128]
            RANGE = config["plotting"].get("range", [[-1, 1], [-1, 1]])
            x = np.linspace(RANGE[0][0], RANGE[0][1], len_sample[0])
            y = np.linspace(RANGE[1][0], RANGE[1][1], len_sample[1])
            X, Y = np.meshgrid(x, y)
            # Convert X and Y to torch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
            input_data = torch.stack((X_tensor, Y_tensor), dim=-1).reshape(-1, 2).to(device)
            unflatten = torch.nn.Unflatten(0, len_sample)
            # Streamplot
            with torch.no_grad():
                F_out = model_f(input_data)
                vect_out = unflatten(F_out)
                vect_out = vect_out.cpu().detach().numpy()
                U = vect_out[:,:, 0]
                V = vect_out[:,:,1]
                B_out = model_b(input_data)
                bout = unflatten(B_out).cpu().detach().numpy()
            stream = ax.streamplot(X, Y, U, V, density=2, linewidth=1, color='#a5a1a1')
            # Create proxy artist for streamplot
            arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='#a5a1a1', marker='>', markeredgewidth=2, markersize=5, label='Vector Field')
            plt.contour(X, Y, bout[:,:,0], levels=[0], colors='#cdebc5')
            plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='#cdebc5')
            contour_fill_legend = mpl.patches.Patch(color='#cdebc5', label=' $ \{x \in X \mid \mathrm{B}(x) \leq 0\}$')        
            # Plotting the Invariant Set
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
        initial_set_center = initial_set_center.reshape(-1, 3)
        for i in range(initial_set_center.shape[0]):
            x = torch.zeros((n, 3))
            x[0, :] = initial_set_center[i, :]
            x = x.to(device)
            for j in range(1, n):
                Fout = model_f(x[j-1])
                x[j] = x[j-1] + Fout * dt
            x = x.cpu().detach().numpy()
            ax.plot(x[:, 0], x[:, 1], x[:, 2],'red')
        # Plot the Obstacles
        if config["unsafe"]["shape"] == "Circle":
            theta = np.linspace(0, 2 * np.pi, 100)
            phi = np.linspace(0, np.pi, 50)
            theta, phi = np.meshgrid(theta, phi)
            unsafe_set_center = config["unsafe"]["center"]
            unsafe_set_radius = config["unsafe"]["radius"]
            if isinstance(unsafe_set_center[0], (int, float)):
                x = unsafe_set_center[0] + unsafe_set_radius * np.sin(phi) * np.cos(theta)
                y = unsafe_set_center[1] + unsafe_set_radius * np.sin(phi) * np.sin(theta)
                z = unsafe_set_center[2] + unsafe_set_radius * np.cos(phi)
                ax.plot_surface(x, y, z, facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.05), linewidth=2, label="Unsafe Set", alpha = 0.1)
            else:
                for center in unsafe_set_center:
                    x = center[0] + unsafe_set_radius * np.sin(phi) * np.cos(theta)
                    y = center[1] + unsafe_set_radius * np.sin(phi) * np.sin(theta)
                    z = center[2] + unsafe_set_radius * np.cos(phi)
                    ax.plot_surface(x, y, z, facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.05), linewidth=2, label="Unsafe Set", alpha = 0.1)
        elif config["unsafe"]["shape"] == "Rectangle":
            RANGE = np.array(config["unsafe"]["range"])
            RANGE = RANGE.reshape(-1, 3, 2)
            for i in range(RANGE.shape[0]):
                x_min, x_max = RANGE[i][0]
                y_min, y_max = RANGE[i][1]
                z_min, z_max = RANGE[i][2]

                # Define the 8 corners of the cuboid
                corners = np.array([
                    [x_min, y_min, z_min],
                    [x_max, y_min, z_min],
                    [x_max, y_max, z_min],
                    [x_min, y_max, z_min],
                    [x_min, y_min, z_max],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max]
                ])

                # Define the 6 faces using the corners
                faces = [
                    [corners[0], corners[1], corners[2], corners[3]],  # bottom
                    [corners[4], corners[5], corners[6], corners[7]],  # top
                    [corners[0], corners[1], corners[5], corners[4]],  # front
                    [corners[2], corners[3], corners[7], corners[6]],  # back
                    [corners[1], corners[2], corners[6], corners[5]],  # right
                    [corners[3], corners[0], corners[4], corners[7]],  # left
                ]
                
                ax.add_collection3d(Poly3DCollection(faces, facecolors='red', linewidths=1, edgecolors='red', alpha=0.2))
        # For the Init Cube
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
    model_v = model_v.cpu()
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1).to('cpu')
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
    model_b = model_b.cpu()
    inputs = torch.stack([X1.flatten(), X2.flatten()], dim=1).to('cpu')
    B_value = model_b(inputs).detach().numpy()
    B_value = B_value.reshape(50,50)
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, B_value, levels=50, cmap="inferno")
    plt.colorbar(label="Barrier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Barrier Heatmap")
    plt.show()

# Plotting 2D Dynamics
def lyapunovBarrierPlot(model_v, model_b, model_f, demos, config, x_data=None, y_data=None):
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
    stream = ax.streamplot(X, Y, U, V, density=2, linewidth=1, color='#a5a1a1')
    # Create proxy artist for streamplot
    arrow_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color='#a5a1a1', marker='>', markeredgewidth=2, markersize=5, label='Vector Field')
    # Contour for Lyapunov Function
    if flag_contour:
        plt.contourf(X, Y, vout[:,:,0], cmap=cm.lajolla)
    # Plotting the Training Data
    initial_set_center = torch.tensor(config["plotting"]["initial_conditions"])
    for i in range(len(demos)):
        ax.plot(demos[i].pos[0,:], demos[i].pos[1,:], color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
    # Plotting the final trajectory
    n = 10000
    dt= config["plotting"]["dt"]
    for i in range(initial_set_center.shape[0]):
        x = torch.zeros((n, 2)).to(device)
        x[0,:] = initial_set_center[i].clone().detach()
        for j in range(1, n):
            Fout = model_f(x[j-1])
            x[j] = x[j-1] + Fout * dt
        x = x.cpu().detach().numpy()
        ax.plot(x[:, 0], x[:, 1],'#ff00ff', label="Learned Trajectory")

    # Plotting the robot trajectory
    if x_data is not None and y_data is not None:
        ax.plot(x_data, y_data, "#49332b", label="Robot Trajectory")   
    
    if flag_barrier:
        plt.contour(X, Y, bout[:,:,0], levels=[0], colors='#cdebc5')
        plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='#cdebc5')
        contour_fill_legend = mpl.patches.Patch(color='#cdebc5', label=' $ \{x \in X \mid \mathrm{B}(x) \leq 0\}$')        
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
            unsafe_set_center = config["unsafe"]["center"]
            unsafe_set_radius = config["unsafe"]["radius"]
            if isinstance(unsafe_set_center[0], (int, float)):
                unsafe_shape = plt.Circle(unsafe_set_center, unsafe_set_radius, facecolor='r', edgecolor='r', linewidth=2, alpha = 0.5, label="Unsafe Set")
                ax.add_patch(unsafe_shape)
            else:
                for ind, center in enumerate(unsafe_set_center):
                    unsafe_shape = plt.Circle(center, unsafe_set_radius, facecolor='r', edgecolor='r', linewidth=2, alpha = 0.5, label=f"Unsafe Set {ind+1}")
                    ax.add_patch(unsafe_shape)
        elif config["unsafe"]["shape"] == 'Custom':
            function = config["unsafe"]["function"]
            function = function.replace("torch.max", "np.maximum")
            function = function.replace("torch.", "np.")
            x = np.linspace(RANGE[0][0], RANGE[0][1], 500)
            y = np.linspace(RANGE[1][0], RANGE[1][1], 500)
            x,y = np.meshgrid(x, y)
            mask = (eval(function) <= 0)
            plt.contourf(x, y, mask.astype(int), levels = [0.5, 1], colors = 'r', linewidths=2, label = "Unsafe Set", alpha = 0.5)
            # plt.contour(X, Y, bout[:,:,0], levels=[0], colors='green')
            # plt.contourf(X, Y, bout[:,:,0], levels=[-np.inf, 0], colors='green', alpha=0.5)

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
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_xlim(RANGE[0][0], RANGE[0][1])
    plt.gca().set_ylim(RANGE[1][0], RANGE[1][1])
    plotting_config = config["plotting"]
    plot_name = plotting_config.get("name", "Final Plot with Obstacles")
    plt.title(plot_name)
    plt.grid(True)
    # plt.axis('auto')
    plt.axis('scaled') 
    plt.margins(x=0,y=0)
    plt.tight_layout()
    return fig

# Plotting 3D Dynamics
def final3DDSPlot(model_f, demos, initial_set_center, config, data_1=None, model_b=None):
    # device = next(model_f.parameters()).device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    for i in range(demos.shape[0]):
        ax.plot3D(demos[i].pos[0,:], demos[i].pos[1,:], demos[i].pos[2,:], color = "#1F75FE", label="Actual Trajectory" if i == 1 else "")
    # Plotting the final trajectory
    n = 3000
    dt = 0.01
    for i in range(initial_set_center.shape[0]):
        x = torch.zeros((n, 3)).to(device)
        x[0,:] = initial_set_center[i].clone().detach()
        for j in range(1, n):
            Fout = model_f(x[j-1])
            x[j] = x[j-1] + Fout * dt
        x = x.cpu().detach().numpy()
        ax.plot(x[:, 0], x[:, 1], x[:, 2], '#ff00ff', label="Learned Trajectory", linewidth=3)
    N = 7
    try:
        RANGE = config["plotting"]["range"]
    except KeyError:
        RANGE = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    x = np.linspace(RANGE[0][0],RANGE[0][1], N)
    y = np.linspace(RANGE[1][0],RANGE[1][1], N)
    z = np.linspace(RANGE[2][0],RANGE[2][1], N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
    # Concatenate X, Y, Z to create input data tensor
    input_data = torch.stack((X_tensor, Y_tensor, Z_tensor), dim=-1).reshape(-1, 3).to(device)
    unflatten = torch.nn.Unflatten(0, (N, N, N))
    with torch.no_grad():
        F_out = model_f(input_data).squeeze()
        vect_out = unflatten(F_out).cpu().detach().numpy()
        U = vect_out[:,:,:,0]
        V = vect_out[:,:,:,1]
        W = vect_out[:,:,:,2]
        # ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True, color='#a5a1a1')        
    # For the sphere
    if config["unsafe"]["shape"] == "Circle":
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        unsafe_set_center = config["unsafe"]["center"]
        unsafe_set_radius = config["unsafe"]["radius"]
        if isinstance(unsafe_set_center[0], (int, float)):
            x = unsafe_set_center[0] + unsafe_set_radius * np.sin(phi) * np.cos(theta)
            y = unsafe_set_center[1] + unsafe_set_radius * np.sin(phi) * np.sin(theta)
            z = unsafe_set_center[2] + unsafe_set_radius * np.cos(phi)
            ax.plot_surface(x, y, z, facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.05), linewidth=2, label="Unsafe Set", alpha = 0.1)
        else:
            for center in unsafe_set_center:
                x = center[0] + unsafe_set_radius * np.sin(phi) * np.cos(theta)
                y = center[1] + unsafe_set_radius * np.sin(phi) * np.sin(theta)
                z = center[2] + unsafe_set_radius * np.cos(phi)
                ax.plot_surface(x, y, z, facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.05), linewidth=2, label="Unsafe Set", alpha = 0.5)
    elif config["unsafe"]["shape"] == "Rectangle":
        RANGE = np.array(config["unsafe"]["range"])
        RANGE = RANGE.reshape(-1, 3, 2)
        for i in range(RANGE.shape[0]):
            x_min, x_max = RANGE[i][0]
            y_min, y_max = RANGE[i][1]
            z_min, z_max = RANGE[i][2]

            # Define the 8 corners of the cuboid
            corners = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])

            # Define the 6 faces using the corners
            faces = [
                [corners[0], corners[1], corners[2], corners[3]],  # bottom
                [corners[4], corners[5], corners[6], corners[7]],  # top
                [corners[0], corners[1], corners[5], corners[4]],  # front
                [corners[2], corners[3], corners[7], corners[6]],  # back
                [corners[1], corners[2], corners[6], corners[5]],  # right
                [corners[3], corners[0], corners[4], corners[7]],  # left
            ]
            
            ax.add_collection3d(Poly3DCollection(faces, facecolors='red', linewidths=1, edgecolors='red', alpha=0.2))

    # For the Init Cube
    x_min, x_max = config["plotting"]["init_range"][0]
    y_min, y_max = config["plotting"]["init_range"][1]
    z_min, z_max = config["plotting"]["init_range"][2]

    # Define the 8 corners of the cuboid
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    # Define the 6 faces using the corners
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[3], corners[0], corners[4], corners[7]],  # left
    ]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='cyan', alpha=0.2))

    # Plotting the Robot Trajectories
    if data_1 is not None:
        ax.plot(data_1["x"].to_numpy(), data_1["y"].to_numpy(), data_1["z"].to_numpy(), color="#49332b", label="Robot Trajectory", linewidth=3) 
    ax.scatter(0, 0, 0, color='black', s=150, label='Equilibrium (0,0,0)')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(config["plotting"]["name"])
    plt.grid(True)
    plt.tight_layout()
    return fig
