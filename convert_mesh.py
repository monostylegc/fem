
import meshio
import numpy as np

def convert_mesh(input_file, output_mesh, output_tags):
    print(f"Reading {input_file}...")
    try:
        msh = meshio.read(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("File read successfully.")
    print(f"Cells keys: {msh.cells_dict.keys()}")
    print(f"Point data keys: {msh.point_data.keys()}")
    print(f"Cell data keys: {msh.cell_data.keys()}")

    # Extract tetra cells
    if 'tetra' not in msh.cells_dict:
        print("Error: No tetra cells found in the mesh.")
        return

    tetra_cells = msh.cells_dict['tetra']
    points = msh.points

    # Extract cell data (labels)
    # The user said labels are 1, 2, 3. We need to find which array holds them.
    # Usually it's in cell_data. We'll look for a likely candidate or just take the first one if scalar.
    cell_data_name = None
    for key, data in msh.cell_data.items():
        # meshio stores cell_data as a list of arrays, one for each cell block
        # We need the one corresponding to 'tetra'
        # But msh.cell_data[key] is a list. We need to match it with msh.cells
        # Simplified approach: create a new mesh with only tetra and the corresponding data
        pass
    
    # Let's assume the first cell data array is the label if we can't identify it by name
    # Or we can inspect the keys.
    # Common names: "MaterialID", "Label", "scalars"
    
    # We will construct a new meshio object to write
    # We need to ensure we get the correct data for the tetra cells
    
    # Find the index of the tetra block in msh.cells
    tetra_index = -1
    for i, cell_block in enumerate(msh.cells):
        if cell_block.type == 'tetra':
            tetra_index = i
            break
            
    if tetra_index == -1:
        print("Could not find tetra block in cells list.")
        return

    # Get the data for this block
    # We'll look for the first cell_data that has the same length as the tetra block
    num_cells = len(tetra_cells)
    labels = None
    
    print("Looking for cell data...")
    for key, data_list in msh.cell_data.items():
        data = data_list[tetra_index]
        print(f"Key: {key}, Shape: {data.shape}")
        if len(data) == num_cells:
            print(f"Found matching data: {key}")
            labels = data
            break
    
    if labels is None:
        print("Warning: No matching cell data found for labels. Creating dummy labels.")
        labels = np.zeros(num_cells, dtype=np.int32)
    
    # Create mesh for geometry
    mesh_out = meshio.Mesh(
        points=points,
        cells=[("tetra", tetra_cells)]
    )
    
    # Write geometry
    print(f"Writing mesh to {output_mesh}...")
    meshio.write(output_mesh, mesh_out)
    
    # Create mesh for tags (cell data)
    # FEniCSx likes the tags in a separate file or the same file but we need to be careful with format
    # We will write a separate XDMF for the mesh tags
    
    # For FEniCSx read, it's often easiest to have one XDMF with the grid and attributes
    # But let's stick to the plan: write mesh and tags.
    # Actually, writing them together is often easier.
    
    mesh_out.cell_data = {"labels": [labels]}
    meshio.write(output_mesh, mesh_out) # Overwrite with data
    
    print("Conversion complete.")

if __name__ == "__main__":
    convert_mesh("Model_coarse.vtu", "mesh.xdmf", "mt.xdmf")
