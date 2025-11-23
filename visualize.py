
import pyvista as pv
import numpy as np
import h5py

def visualize_case(load_case="compression"):
    print(f"Loading results for {load_case}...")
    
    results_file = f"results_{load_case}.h5"
    try:
        f = h5py.File(results_file, "r")
    except Exception as e:
        print(f"Error opening {results_file}: {e}")
        return None
    
    # Load geometry, topology, displacement
    try:
        geom = f["Mesh"]["mesh"]["geometry"][:]
        topo = f["Mesh"]["mesh"]["topology"][:]
        disp = f["Function"]["f"]["0"][:]
    except KeyError as e:
        print(f"Error reading basic datasets: {e}")
        f.close()
        return None

    # Load stress
    try:
        stress = f["Function"]["von_Mises"]["0"][:]
    except KeyError:
        print("Von Mises stress not found.")
        stress = None
    
    f.close()
    
    points = geom
    cells = topo
    
    # Create PyVista mesh
    num_cells = len(cells)
    cells_pv = np.hstack((np.full((num_cells, 1), 4), cells)).flatten()
    cell_type = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)
    
    grid = pv.UnstructuredGrid(cells_pv, cell_type, points)
    
    # Add displacement
    grid.point_data["displacement"] = disp
    
    # Add stress if available
    if stress is not None:
        if len(stress) == num_cells:
             grid.cell_data["von_Mises"] = stress.flatten()
             print(f"Stress min: {np.min(stress):.2e} Pa, max: {np.max(stress):.2e} Pa")
        else:
             print(f"Stress size mismatch: {len(stress)} vs {num_cells} cells")

    # Warp
    max_disp = np.max(np.linalg.norm(disp, axis=1))
    print(f"Max displacement: {max_disp:.2e} m")
    
    if max_disp == 0:
        scale_factor = 1.0
    else:
        bounds = grid.bounds
        size = bounds[1] - bounds[0]
        target_disp = size * 0.1
        scale_factor = target_disp / max_disp
        
    print(f"Using scale factor: {scale_factor:.2f}")
    
    warped = grid.warp_by_vector("displacement", factor=scale_factor)
    
    return warped, load_case

def visualize_all():
    load_cases = ["compression", "flexion", "extension", "lateral_left", "lateral_right"]
    
    results = []
    for case in load_cases:
        result = visualize_case(case)
        if result:
            results.append(result)
    
    if not results:
        print("No results to visualize.")
        return
    
    # Create multi-window plot
    n = len(results)
    plotter = pv.Plotter(shape=(2, 3))
    
    for i, (warped, case) in enumerate(results):
        row = i // 3
        col = i % 3
        plotter.subplot(row, col)
        plotter.add_text(f"{case.replace('_', ' ').title()}", font_size=10)
        
        # Get stress statistics for better color scaling
        if "von_Mises" in warped.array_names:
            stress_data = warped["von_Mises"]
            # Use 90th percentile as max for better visualization
            stress_max = np.percentile(stress_data, 90)
            stress_min = 0
            print(f"{case}: Using colormap range [0, {stress_max:.2e}] Pa (90th percentile)")
            
            plotter.add_mesh(warped, scalars="von_Mises", cmap="jet", 
                           show_edges=False, clim=[stress_min, stress_max])
        else:
            plotter.add_mesh(warped, scalars="von_Mises", cmap="jet", show_edges=False)
        
        plotter.view_isometric()
    
    print("Opening interactive plot window...")
    plotter.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Visualize single case
        load_case = sys.argv[1]
        result = visualize_case(load_case)
        if result:
            plotter = pv.Plotter(shape=(1, 2))
            warped, case = result
            
            # Displacement
            plotter.subplot(0, 0)
            plotter.add_text("Displacement", font_size=12)
            plotter.add_mesh(warped, scalars="displacement", cmap="jet", show_edges=False)
            plotter.view_isometric()
            
            # Stress with custom colormap range
            plotter.subplot(0, 1)
            plotter.add_text("Von Mises Stress", font_size=12)
            
            if "von_Mises" in warped.array_names:
                stress_data = warped["von_Mises"]
                stress_max = np.percentile(stress_data, 90)
                print(f"Using stress colormap range: [0, {stress_max:.2e}] Pa (90th percentile)")
                plotter.add_mesh(warped, scalars="von_Mises", cmap="jet", 
                               show_edges=False, clim=[0, stress_max])
            else:
                plotter.add_mesh(warped, scalars="von_Mises", cmap="jet", show_edges=False)
            
            plotter.view_isometric()
            
            plotter.show()
    else:
        # Visualize all cases
        visualize_all()
