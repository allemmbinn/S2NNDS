from common_header import *

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
        # Plotting the initial set
        initial_set_radius = config["init"]["radius"]
        x_min = initial_set_center[0][0] - initial_set_radius
        x_max = initial_set_center[0][0] + initial_set_radius
        y_min = initial_set_center[0][1] - initial_set_radius
        y_max = initial_set_center[0][1] + initial_set_radius
        initial_set = patches.Rectangle(
            (x_min, y_min),  # Bottom-left corner (x_min, y_min)
            x_max - x_min,   # Width
            y_max - y_min,   # Height
            linewidth=2,     # Border thickness
            edgecolor='black',  # Border color
            facecolor='black', # Transparent fill
            alpha = 0.5, 
            label = "Initial Set"
        )
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