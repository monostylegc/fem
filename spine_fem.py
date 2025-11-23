
import numpy as np
import ufl
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

def run_fem(load_case="compression"):
    """
    Run FEM analysis with different load cases.
    
    Args:
        load_case: "compression" (downward), "flexion" (forward bend), 
                   "extension" (backward bend), "lateral_left", "lateral_right"
    """
    import meshio
    print(f"Loading mesh for load case: {load_case}...")
    msh = meshio.read("mesh.xdmf")
    
    # Extract geometry and topology
    points = msh.points
    cells = msh.cells_dict["tetra"]
    print(f"Points shape: {points.shape}, type: {points.dtype}")
    print(f"Cells shape: {cells.shape}, type: {cells.dtype}")
    
    # Create dolfinx mesh
    import dolfinx.cpp.mesh as cppmesh
    from dolfinx.mesh import create_mesh, CellType
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    # Create mesh
    # dolfinx.mesh.create_mesh(comm, cells, x, domain)
    # We need to define the element type
    import basix.ufl
    element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
    domain_ufl = ufl.Mesh(element)
    
    # Convert to appropriate types
    points = np.ascontiguousarray(points.astype(np.float64))
    cells = np.ascontiguousarray(cells.astype(np.int64))
    
    # IMPORTANT: Scale coordinates from millimeters to meters
    # The original Model.vtk is in mm, but FEniCSx expects SI units (meters)
    points = points * 0.001  # mm -> m
    
    print(f"Points (converted): {points.shape}, {points.dtype}")
    print(f"Cells (converted): {cells.shape}, {cells.dtype}")
    print(f"Coordinate range after scaling: Z from {np.min(points[:,2]):.4f} to {np.max(points[:,2]):.4f} m")
    
    domain = create_mesh(comm, cells, domain_ufl, points)
    
    # Create meshtags
    # msh.cell_data["labels"][0] contains the tags
    labels = msh.cell_data["labels"][0].astype(np.int32)
    
    # We need to create MeshTags
    # dolfinx.mesh.meshtags(mesh, dim, indices, values)
    # Here we have values for ALL cells.
    # So indices are 0 to N-1
    num_cells = len(cells)
    cell_indices = np.arange(num_cells, dtype=np.int32)
    
    ct = mesh.meshtags(domain, domain.topology.dim, cell_indices, labels)


    print("Defining function spaces...")
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    
    # Material properties
    # Label 1: L4 (Bone) -> E = 12 GPa
    # Label 2: L5 (Bone) -> E = 12 GPa
    # Label 3: Disc -> E = 4.2 MPa (0.0042 GPa) - simplified
    # Using SI units (Pa)
    E_bone = 12e9
    E_disc = 4.2e6
    nu_bone = 0.3
    nu_disc = 0.45

    # Define material parameters as functions
    Q = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(Q)
    nu = fem.Function(Q)
    
    # Assign values based on cell tags
    # We need to map the meshtags to the function
    # ct.values contains the tags, ct.indices contains the cell indices
    
    print("Assigning material properties...")
    # Default to bone
    E.x.array[:] = E_bone
    nu.x.array[:] = nu_bone
    
    # Get indices for disc (Label 3)
    disc_cells = ct.indices[ct.values == 3]
    E.x.array[disc_cells] = E_disc
    nu.x.array[disc_cells] = nu_disc
    
    # Lame parameters
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    print("Defining boundary conditions...")
    # Find bottom of L5 (Label 2)
    # We'll find points with minimal Z in the L5 region
    # This is a bit heuristic without explicit surface tags
    
    # Get all cells for L5
    l5_cells = ct.indices[ct.values == 2]
    
    # We need to find the boundary facets of L5 that are at the bottom
    # Strategy: Find the bounding box of L5, then take the bottom 5% or so
    # Or just find the global min Z and fix everything close to it?
    # Let's try global min Z first, assuming the spine is upright and L5 is at the bottom.
    
    geometry = domain.geometry.x
    min_z = np.min(geometry[:, 2])
    max_z = np.max(geometry[:, 2])
    print(f"Mesh Z range: {min_z} to {max_z}")
    
    # Fix bottom (z near min_z)
    def bottom_boundary(x):
        return np.isclose(x[2], min_z, atol=(max_z - min_z) * 0.05)

    fdim = domain.topology.dim - 1
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
    
    # Check if we found any facets
    if len(bottom_facets) == 0:
        print("Warning: No bottom facets found with current tolerance. Increasing tolerance.")
        # Fallback or just proceed (will likely fail to solve if empty)
        
    u_bc = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, bottom_facets), V)
    
    # Apply load to top of L4 (Label 1)
    # We'll apply a traction to the top surface
    ds = ufl.Measure("ds", domain=domain)
    
    # We need to mark the top boundary for the integral
    # Create a new meshtag for boundaries
    # 1: Top, 2: Bottom (already handled by Dirichlet), 0: Other
    
    facet_indices = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], max_z, atol=(max_z - min_z) * 0.05))
    facet_tags = np.zeros_like(facet_indices, dtype=np.int32)
    facet_tags[:] = 1 # Tag 1 for top
    
    # Create the meshtags object
    # We need all facets, but we only care about the top one for the Neumann BC
    # For 'ds', we can use subdomain_data
    
    # Note: mesh.meshtags requires sorted indices
    sorted_order = np.argsort(facet_indices)
    mt_boundary = mesh.meshtags(domain, fdim, facet_indices[sorted_order], facet_tags[sorted_order])
    
    # Calculate area of top surface for force -> pressure conversion
    # We need to integrate 1 over the top surface ds(1)
    ds_temp = ufl.Measure("ds", domain=domain, subdomain_data=mt_boundary)
    area_form = fem.form(1.0 * ds_temp(1))
    top_area = domain.comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
    print(f"Top surface area: {top_area:.6f} m^2")
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt_boundary)
    
    # Define force in Newtons (instead of pressure)
    # Typical spine loads: 500-2000 N for daily activities
    applied_force = 3000.0  # N
    
    # Convert force to pressure
    P = applied_force / top_area  # Pa
    print(f"Applied force: {applied_force} N")
    print(f"Resulting pressure: {P/1e6:.4f} MPa") 
    
    # Define direction based on load case
    if load_case == "compression":
        # Downward (-Z)
        T = fem.Constant(domain, default_scalar_type((0, 0, -P)))
        direction = "downward (compression)"
    elif load_case == "flexion":
        # Forward bending (+Y direction)
        T = fem.Constant(domain, default_scalar_type((0, P, 0)))
        direction = "forward (flexion)"
    elif load_case == "extension":
        # Backward bending (-Y direction)
        T = fem.Constant(domain, default_scalar_type((0, -P, 0)))
        direction = "backward (extension)"
    elif load_case == "lateral_left":
        # Left bending (+X direction)
        T = fem.Constant(domain, default_scalar_type((P, 0, 0)))
        direction = "left (lateral)"
    elif load_case == "lateral_right":
        # Right bending (-X direction)
        T = fem.Constant(domain, default_scalar_type((-P, 0, 0)))
        direction = "right (lateral)"
    else:
        print(f"Unknown load case: {load_case}, using compression")
        T = fem.Constant(domain, default_scalar_type((0, 0, -P)))
        direction = "downward (compression)"
    
    print(f"Load direction: {direction}")
    
    print("Defining variational problem...")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(T, v) * ds(1) # Integrate over top surface (tag 1)

    print("Solving...")
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-6}, petsc_options_prefix="fem_")
    uh = problem.solve()

    print("Calculating Von Mises stress...")
    s = sigma(uh) - 1./3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    
    # We need to project or interpolate this into a function space
    # DG0 (cell-wise constant) is often good for stress
    W = fem.functionspace(domain, ("DG", 0))
    stress_expr = fem.Expression(von_Mises, W.element.interpolation_points)
    stress_h = fem.Function(W)
    stress_h.name = "von_Mises"
    stress_h.interpolate(stress_expr)
    
    # Print stress statistics
    stress_values = stress_h.x.array
    print(f"Von Mises Stress Statistics:")
    print(f"  Min: {np.min(stress_values):.2e} Pa ({np.min(stress_values)/1e6:.2f} MPa)")
    print(f"  Max: {np.max(stress_values):.2e} Pa ({np.max(stress_values)/1e6:.2f} MPa)")
    print(f"  Mean: {np.mean(stress_values):.2e} Pa ({np.mean(stress_values)/1e6:.2f} MPa)")

    print("Saving results...")
    output_file = f"results_{load_case}.xdmf"
    with io.XDMFFile(domain.comm, output_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
        xdmf.write_function(stress_h)
        
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    # Check if load case is specified
    if len(sys.argv) > 1:
        load_case = sys.argv[1]
        run_fem(load_case)
    else:
        # Run all load cases
        load_cases = ["compression", "flexion", "extension", "lateral_left", "lateral_right"]
        for case in load_cases:
            print(f"\n{'='*60}")
            print(f"Running load case: {case}")
            print(f"{'='*60}\n")
            run_fem(case)
