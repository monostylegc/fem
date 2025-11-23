
import pyvista as pv
import numpy as np
import h5py

def visualize():
    print("Loading results using h5py...")
    try:
        f = h5py.File("results.h5", "r")
    except Exception as e:
        print(f"Error opening H5 file: {e}")
        return

    print("Keys in H5:", list(f.keys()))
    
    # Structure usually: /Mesh/mesh/geometry, /Mesh/mesh/topology
    # /Function/f/0
    
    # Load geometry, topology, displacement
    try:
        geom = f["Mesh"]["mesh"]["geometry"][:]
        topo = f["Mesh"]["mesh"]["topology"][:]
        disp = f["Function"]["f"]["0"][:]
    except KeyError as e:
        print(f"Error reading basic datasets: {e}")
        return

    # Load stress
    try:
        stress = f["Function"]["von_Mises"]["0"][:]
        print(f"Stress shape: {stress.shape}")
    except KeyError:
        print("Von Mises stress not found in results.")
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
        # Check shape
        if len(stress) == num_cells:
             grid.cell_data["von_Mises"] = stress.flatten()
        else:
             print(f"Stress size mismatch: {len(stress)} vs {num_cells} cells")

    # Warp
    max_disp = np.max(np.linalg.norm(disp, axis=1))
    print(f"Max displacement: {max_disp}")
    
    if max_disp == 0:
        scale_factor = 1.0
    else:
        bounds = grid.bounds
        size = bounds[1] - bounds[0]
        target_disp = size * 0.1
        scale_factor = target_disp / max_disp
        
    print(f"Using scale factor: {scale_factor}")
    
    warped = grid.warp_by_vector("displacement", factor=scale_factor)
    
    # Plot
    plotter = pv.Plotter(shape=(1, 2))
    
    # Subplot 1: Displacement
    plotter.subplot(0, 0)
    plotter.add_text("Displacement", font_size=12)
    plotter.add_mesh(warped, scalars="displacement", cmap="jet", show_edges=False)
    plotter.view_isometric()
    
    # Subplot 2: Von Mises Stress
    if stress is not None:
        plotter.subplot(0, 1)
        plotter.add_text("Von Mises Stress", font_size=12)
        plotter.add_mesh(warped, scalars="von_Mises", cmap="jet", show_edges=False)
        plotter.view_isometric()
    
    print("Opening interactive plot window...")
    plotter.show()

if __name__ == "__main__":
    visualize()
