import pyvista as pv
import argparse
import numpy as np
import os

def visualize_mpm(vtk_file, off_screen=False):
    """
    Visualize MPM results (VTK point cloud) using PyVista
    """
    if not os.path.exists(vtk_file):
        print(f"Error: File {vtk_file} not found.")
        return

    print(f"Loading {vtk_file}...")
    mesh = pv.read(vtk_file)
    
    n_points = mesh.n_points
    print(f"Loaded {n_points} particles")
    
    # Check available arrays
    print("Available arrays:", mesh.point_data.keys())
    
    # Create plotter
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    
    # Scalar to plot
    scalar_name = "von_mises_stress"
    if scalar_name not in mesh.point_data:
        print(f"Warning: '{scalar_name}' not found. Using solid color.")
        scalar_name = None
    else:
        # Print stress stats
        stress = mesh.point_data[scalar_name]
        print(f"Stress range: {np.min(stress)/1e6:.2f} to {np.max(stress)/1e6:.2f} MPa")
    
    # Add mesh (particles)
    # render_points_as_spheres makes them look like actual particles
    plotter.add_mesh(mesh, 
                     scalars=scalar_name, 
                     cmap="jet", 
                     point_size=6, 
                     render_points_as_spheres=True,
                     scalar_bar_args={'title': "Von Mises Stress (Pa)"},
                     show_scalar_bar=True)
    
    # Add axes and grid
    plotter.add_axes()
    # plotter.show_grid()
    
    # Set camera view
    plotter.view_isometric()
    plotter.camera.zoom(1.2)
    
    # Add text info
    plotter.add_text(f"MPM Simulation: {os.path.basename(vtk_file)}", position='upper_left', font_size=10, color='black')
    plotter.add_text(f"Particles: {n_points}", position='upper_right', font_size=10, color='black')

    if off_screen:
        # Save screenshot in off-screen mode
        output_img = vtk_file.replace(".vtk", ".png")
        print(f"Saving screenshot to {output_img}...")
        plotter.screenshot(output_img)
    else:
        # Show interactive window
        print("Showing interactive window...")
        print("Controls: Left-click drag to rotate, Mouse wheel to zoom, 'q' to quit")
        plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MPM VTK results")
    parser.add_argument("file", nargs="?", default="results_mpm_compression.vtk", help="Path to VTK file")
    parser.add_argument("--off-screen", action="store_true", help="Render off-screen and save screenshot only")
    
    args = parser.parse_args()
    visualize_mpm(args.file, args.off_screen)
