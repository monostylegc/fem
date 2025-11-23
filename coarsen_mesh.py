
import pyvista as pv
import numpy as np
import meshio
from scipy.spatial import cKDTree

def coarsen_mesh(input_file, output_file, target_voxels=50000):
    print(f"Reading {input_file} using meshio...")
    try:
        msh = meshio.read(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    points = msh.points
    print(f"Original points: {len(points)}")
    
    # Extract labels
    labels = None
    for key, data in msh.cell_data.items():
        if key == 'labels' or key == 'scalars':
            labels = data[0]
            break
            
    if labels is None and len(msh.cell_data) > 0:
        labels = list(msh.cell_data.values())[0][0]
        
    if labels is None:
        print("No labels found. Cannot preserve material properties.")
        return

    # To map labels to points (since we are doing point-based proximity), 
    # we ideally need point labels. But we have cell labels.
    # We can assign the cell label to its centroid, and build the tree on centroids.
    
    print("Computing cell centroids...")
    tetra_cells = msh.cells_dict['tetra']
    # Calculate centroids
    # points[tetra_cells] shape (N, 4, 3)
    # This might be memory intensive. 
    # Optimization: Calculate mean manually or in chunks if needed.
    # 3.3M cells * 4 points * 3 floats * 8 bytes ~ 300MB. It fits in memory.
    cell_points = points[tetra_cells]
    centroids = np.mean(cell_points, axis=1)
    
    print("Building KDTree on centroids...")
    tree = cKDTree(centroids)
    
    # Define Grid
    bounds = [
        np.min(points[:, 0]), np.max(points[:, 0]),
        np.min(points[:, 1]), np.max(points[:, 1]),
        np.min(points[:, 2]), np.max(points[:, 2])
    ]
    
    x_len = bounds[1] - bounds[0]
    y_len = bounds[3] - bounds[2]
    z_len = bounds[5] - bounds[4]
    
    vol_bb = x_len * y_len * z_len
    # s^3 = vol / target
    s = (vol_bb / target_voxels) ** (1/3)
    
    dims = (np.array([x_len, y_len, z_len]) / s).astype(int)
    print(f"Grid dims: {dims}")
    
    grid = pv.ImageData()
    grid.dimensions = dims + 1
    grid.origin = (bounds[0], bounds[2], bounds[4])
    grid.spacing = (s, s, s)
    
    # Grid points (cell centers of the grid)
    # We want to label the *cells* of the grid (voxels)
    # ImageData cells are defined by the points.
    # Let's use cell_centers() of the grid.
    
    print("Generating grid cell centers...")
    # This creates a PolyData of centers
    grid_centers = grid.cell_centers().points
    
    print(f"Querying {len(grid_centers)} voxels...")
    dists, indices = tree.query(grid_centers)
    
    # Threshold based on distance
    # If a voxel is too far from any real cell, it's empty space.
    # Threshold: diagonal of the voxel? or just s?
    # s is the side length. diagonal is s * sqrt(3).
    threshold_dist = s * 1.0 # Slightly generous
    
    valid_mask = dists < threshold_dist
    
    print(f"Valid voxels: {np.sum(valid_mask)}")
    
    # Assign labels
    # 0 for invalid
    grid_labels = np.zeros(grid.n_cells, dtype=labels.dtype)
    grid_labels[valid_mask] = labels[indices[valid_mask]]
    
    grid.cell_data["labels"] = grid_labels
    
    # Threshold to remove 0
    print("Thresholding...")
    # threshold() works on scalars.
    thresh = grid.threshold(0.5, scalars="labels")
    
    print("Triangulating...")
    tetra = thresh.triangulate()
    
    print(f"Final mesh: {tetra.n_cells} cells")
    
    print(f"Saving to {output_file}...")
    tetra.save(output_file)
    
    print("Visualizing coarse mesh...")
    p = pv.Plotter()
    p.add_mesh(tetra, scalars="labels", cmap="coolwarm", show_edges=True)
    p.add_text(f"Coarse Mesh: {tetra.n_cells} cells", position="upper_left")
    p.show()

if __name__ == "__main__":
    # Adjust target_voxels to control coarseness
    coarsen_mesh("Model.vtk", "Model_coarse.vtu", target_voxels=1000000)
