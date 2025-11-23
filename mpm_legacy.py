"""
Material Point Method (MPM) Simulation for Spine Analysis using Taichi

Converts the FEniCSx-based FEM analysis to a particle-based MPM approach.
Supports multiple load cases: compression, flexion, extension, lateral bending.
"""

import numpy as np
import taichi as ti
import meshio
import argparse
from pathlib import Path

# Initialize Taichi
ti.init(arch=ti.gpu)  # Use GPU if available, falls back to CPU

# MPM simulation parameters
@ti.data_oriented
class MPMSimulator:
    def __init__(self, mesh_file="mesh.xdmf", grid_res=64):
        """
        Initialize MPM simulator
        
        Args:
            mesh_file: Path to mesh file (XDMF format)
            grid_res: Grid resolution (will be adjusted based on mesh size)
        """
        self.mesh_file = mesh_file
        self.base_grid_res = grid_res
        
        # Load mesh and extract data
        print(f"Loading mesh from {mesh_file}...")
        self.load_mesh()
        
        # Setup grid based on mesh bounds
        self.setup_grid()
        
        # Material properties
        self.E_bone = 12e9  # Pa
        self.E_disc = 4.2e6  # Pa
        self.nu_bone = 0.3
        self.nu_disc = 0.45
        self.rho_bone = 1850  # kg/m^3 (cortical bone)
        self.rho_disc = 1040  # kg/m^3 (intervertebral disc)
        
        # Simulation parameters
        self.dt = 1e-7  # Extremely small time step for stability
        self.gravity = ti.Vector([0, 0, 0.0])
        self.damping = 0.95  # Strong damping
        self.mass_scaling = 10.0  # Moderate mass scaling for stability
        
        # Applied force (N)
        self.applied_force = 3000.0
        
        # Convergence monitoring
        self.kinetic_energy = ti.field(dtype=ti.f32, shape=())
        self.force_nodes_count = ti.field(dtype=ti.i32, shape=())
        
        # Allocate particle data
        self.setup_particles()
        
        # Setup visualization (needs to be after particles are initialized)
        self.setup_visualization()
        
    def load_mesh(self):
        """Load mesh from XDMF file"""
        msh = meshio.read(self.mesh_file)
        
        # Extract geometry and topology
        self.mesh_points = msh.points * 0.001  # mm -> m
        self.mesh_cells = msh.cells_dict["tetra"]
        self.mesh_labels = msh.cell_data["labels"][0]
        
        print(f"Mesh loaded: {len(self.mesh_points)} vertices, {len(self.mesh_cells)} cells")
        
        # Compute bounding box
        self.bbox_min = np.min(self.mesh_points, axis=0)
        self.bbox_max = np.max(self.mesh_points, axis=0)
        self.bbox_size = self.bbox_max - self.bbox_min
        
        print(f"Bounding box: [{self.bbox_min[0]:.4f}, {self.bbox_min[1]:.4f}, {self.bbox_min[2]:.4f}] to "
              f"[{self.bbox_max[0]:.4f}, {self.bbox_max[1]:.4f}, {self.bbox_max[2]:.4f}]")
        
    def setup_grid(self):
        """Setup background Eulerian grid"""
        # Adjust grid resolution based on bounding box
        max_dim = np.max(self.bbox_size)
        dx_value = max_dim / self.base_grid_res  # Grid spacing
        
        # Grid dimensions (add padding)
        padding = 2
        grid_size_np = np.ceil(self.bbox_size / dx_value).astype(int) + 2 * padding
        grid_offset_np = self.bbox_min - padding * dx_value
        
        print(f"Grid setup: resolution={grid_size_np}, dx={dx_value:.6f} m")
        
        # Store as Taichi fields/constants for kernel access
        self.dx = ti.field(dtype=ti.f32, shape=())
        self.dx[None] = dx_value
        self.inv_dx = ti.field(dtype=ti.f32, shape=())
        self.inv_dx[None] = 1.0 / dx_value
        
        # Grid offset as Taichi vector field
        self.grid_offset = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.grid_offset[None] = ti.Vector(grid_offset_np.astype(np.float32))
        
        # Grid dimensions for boundary checks
        self.grid_size_x = grid_size_np[0]
        self.grid_size_y = grid_size_np[1]
        self.grid_size_z = grid_size_np[2]
        
        # Grid fields (mass, momentum, velocity)
        self.grid_m = ti.field(dtype=ti.f32, shape=tuple(grid_size_np))
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=tuple(grid_size_np))
        
    def setup_particles(self):
        """Convert mesh tetrahedra to material points"""
        # Sample particles from each tetrahedron
        # Use centroid + additional samples for better representation
        particles_per_tet = 4  # 1 at centroid + 3 additional
        
        self.n_particles = len(self.mesh_cells) * particles_per_tet
        
        print(f"Initializing {self.n_particles} particles...")
        
        # Particle fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)  # Position
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)  # Velocity
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)  # Affine momentum
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)  # Deformation gradient
        self.V = ti.field(dtype=ti.f32, shape=self.n_particles)  # Volume
        self.mass = ti.field(dtype=ti.f32, shape=self.n_particles)  # Mass
        
        # Material properties per particle
        self.E = ti.field(dtype=ti.f32, shape=self.n_particles)  # Young's modulus
        self.nu = ti.field(dtype=ti.f32, shape=self.n_particles)  # Poisson's ratio
        self.rho = ti.field(dtype=ti.f32, shape=self.n_particles)  # Density
        
        # Material type (1=L4, 2=L5, 3=Disc)
        self.mat_label = ti.field(dtype=ti.i32, shape=self.n_particles)
        
        # Stress field
        self.stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)
        self.von_mises = ti.field(dtype=ti.f32, shape=self.n_particles)
        
        # Initialize particles from mesh
        self.initialize_particles()
        
    def initialize_particles(self):
        """Initialize particle positions and properties from mesh"""
        particle_positions = []
        particle_volumes = []
        particle_labels = []
        
        for tet_idx, tet in enumerate(self.mesh_cells):
            # Get tetrahedron vertices
            v0, v1, v2, v3 = self.mesh_points[tet]
            
            # Compute tetrahedron volume
            mat = np.column_stack([v1-v0, v2-v0, v3-v0])
            tet_volume = abs(np.linalg.det(mat)) / 6.0
            
            # Sample particles
            # 1. Centroid
            centroid = (v0 + v1 + v2 + v3) / 4.0
            particle_positions.append(centroid)
            particle_volumes.append(tet_volume / 4)
            particle_labels.append(self.mesh_labels[tet_idx])
            
            # 2-4. Additional samples (using barycentric coordinates)
            for _ in range(3):
                # Random barycentric coordinates
                r = np.random.random(4)
                r = r / r.sum()
                pos = r[0]*v0 + r[1]*v1 + r[2]*v2 + r[3]*v3
                particle_positions.append(pos)
                particle_volumes.append(tet_volume / 4)
                particle_labels.append(self.mesh_labels[tet_idx])
        
        # Convert to numpy arrays
        particle_positions = np.array(particle_positions, dtype=np.float32)
        particle_volumes = np.array(particle_volumes, dtype=np.float32)
        particle_labels = np.array(particle_labels, dtype=np.int32)
        
        # Copy to Taichi fields
        self.x.from_numpy(particle_positions)
        self.V.from_numpy(particle_volumes)
        self.mat_label.from_numpy(particle_labels)
        
        # Initialize material properties
        E_array = np.where(particle_labels == 3, self.E_disc, self.E_bone)
        nu_array = np.where(particle_labels == 3, self.nu_disc, self.nu_bone)
        rho_array = np.where(particle_labels == 3, self.rho_disc, self.rho_bone)
        
        self.E.from_numpy(E_array.astype(np.float32))
        self.nu.from_numpy(nu_array.astype(np.float32))
        self.rho.from_numpy(rho_array.astype(np.float32))
        
        # Initialize mass with scaling
        mass_array = rho_array * particle_volumes * self.mass_scaling
        self.mass.from_numpy(mass_array.astype(np.float32))
        
        # Initialize deformation gradient to identity
        F_init = np.tile(np.eye(3, dtype=np.float32), (self.n_particles, 1, 1))
        self.F.from_numpy(F_init)
        
        # Initialize velocity to zero
        self.v.from_numpy(np.zeros((self.n_particles, 3), dtype=np.float32))
        self.C.from_numpy(np.zeros((self.n_particles, 3, 3), dtype=np.float32))
        
        print(f"Particles initialized:")
        print(f"  Bone particles: {np.sum(particle_labels != 3)}")
        print(f"  Disc particles: {np.sum(particle_labels == 3)}")
        
    @ti.kernel
    def clear_grid(self):
        """Clear grid quantities"""
        for i, j, k in self.grid_m:
            self.grid_m[i, j, k] = 0.0
            self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
    
    @ti.kernel
    def particle_to_grid(self):
        """Transfer particle mass and momentum to grid (P2G)"""
        for p in range(self.n_particles):
            # Particle position in grid coordinates
            base = ti.cast((self.x[p] - self.grid_offset[None]) * self.inv_dx[None] - 0.5, ti.i32)
            fx = (self.x[p] - self.grid_offset[None]) * self.inv_dx[None] - ti.cast(base, ti.f32)
            
            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            
            # Linear elastic constitutive model (SAME AS FEM!)
            # Small strain assumption: epsilon = 0.5 * (F + F^T) - I
            F = self.F[p]
            
            # Lame parameters (same formulation as FEM)
            E = self.E[p]
            nu = self.nu[p]
            mu = E / (2.0 * (1.0 + nu))
            lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            
            # Small strain tensor
            epsilon = 0.5 * (F + F.transpose()) - ti.Matrix.identity(ti.f32, 3)
            trace_eps = epsilon.trace()
            
            # Cauchy stress (linear elastic) - EXACTLY as in FEM
            # sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
            stress = lambda_ * trace_eps * ti.Matrix.identity(ti.f32, 3) + 2.0 * mu * epsilon
            
            self.stress[p] = stress
            
            # Affine momentum from APIC (stress contribution)
            affine = stress * self.V[p] * 4.0 * self.inv_dx[None] * self.inv_dx[None]
            
            # Scatter to grid
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(offset, ti.f32) - fx) * self.dx[None]
                        weight = w[i].x * w[j].y * w[k].z
                        
                        grid_idx = base + offset
                        # Check bounds
                        if (0 <= grid_idx[0] < self.grid_size_x and
                            0 <= grid_idx[1] < self.grid_size_y and
                            0 <= grid_idx[2] < self.grid_size_z):
                            
                            self.grid_m[grid_idx] += weight * self.mass[p]
                            self.grid_v[grid_idx] += weight * (self.mass[p] * self.v[p] + affine @ dpos)
    
    @ti.kernel
    def grid_update(self, force_dir: ti.types.vector(3, ti.f32), current_time: ti.f32):
        """Update grid velocities with forces and boundary conditions"""
        # Force ramping: linearly increase force over 0.1s
        ramp = ti.min(current_time / 0.1, 1.0)
        current_force = self.applied_force * ramp
        
        self.force_nodes_count[None] = 0
        
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 1e-5:  # Threshold
                # Normalize momentum to velocity
                self.grid_v[i, j, k] /= self.grid_m[i, j, k]
                
                # Clamp velocity to prevent explosion/NaN
                v_norm = self.grid_v[i, j, k].norm()
                if v_norm > 100.0:
                    self.grid_v[i, j, k] *= 100.0 / v_norm
                if ti.math.isnan(v_norm):
                    self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                
                # Convert grid position to world coordinates
                grid_pos = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)]) * self.dx[None] + self.grid_offset[None]
                
                # Apply gravity
                self.grid_v[i, j, k] += self.dt * self.gravity
                
                # Apply external force to top surface
                # Identify top region - use a more generous threshold
                # Top is around -0.7159, grid max z is slightly higher
                z_max = self.bbox_max[2]
                z_threshold = z_max - 0.05  # Top 5cm (generous to ensure capture)
                
                if grid_pos[2] > z_threshold:
                    # Apply force only if sufficient mass exists to avoid instability
                    if self.grid_m[i, j, k] > 1e-5:
                        # Force per unit mass (acceleration)
                        # Distribute total force roughly over the top nodes
                        # This is an approximation
                        accel = force_dir * (current_force / self.grid_m[i, j, k])
                        
                        # Clamp acceleration to prevent explosion
                        accel_norm = accel.norm()
                        if accel_norm > 5000.0:
                            accel *= 5000.0 / accel_norm
                            
                        self.grid_v[i, j, k] += self.dt * accel
                        self.force_nodes_count[None] += 1
                
                # Boundary conditions
                # Bottom surface: fix velocity (Dirichlet BC)
                z_bottom = self.grid_offset[None][2] + 2 * self.dx[None]
                if grid_pos[2] < z_bottom:
                    self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                
                # Side boundaries: free slip (prevent leaking)
                if i < 2 or i >= self.grid_size_x - 2:
                    self.grid_v[i, j, k][0] = 0.0
                    self.grid_v[i, j, k][1] *= 0.9  # Friction
                    self.grid_v[i, j, k][2] *= 0.9
                if j < 2 or j >= self.grid_size_y - 2:
                    self.grid_v[i, j, k][1] = 0.0
                    self.grid_v[i, j, k][0] *= 0.9
                    self.grid_v[i, j, k][2] *= 0.9
                if k < 2 or k >= self.grid_size_z - 2:
                    self.grid_v[i, j, k][2] = 0.0
                    self.grid_v[i, j, k][0] *= 0.9
                    self.grid_v[i, j, k][1] *= 0.9
    
    @ti.kernel
    def grid_to_particle(self):
        """Transfer grid velocities back to particles (G2P)"""
        for p in range(self.n_particles):
            # Particle position in grid coordinates
            base = ti.cast((self.x[p] - self.grid_offset[None]) * self.inv_dx[None] - 0.5, ti.i32)
            fx = (self.x[p] - self.grid_offset[None]) * self.inv_dx[None] - ti.cast(base, ti.f32)
            
            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            
            # Initialize new velocity and affine matrix
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            
            # Gather from grid
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(offset, ti.f32) - fx) * self.dx[None]
                        weight = w[i].x * w[j].y * w[k].z
                        
                        grid_idx = base + offset
                        if (0 <= grid_idx[0] < self.grid_size_x and
                            0 <= grid_idx[1] < self.grid_size_y and
                            0 <= grid_idx[2] < self.grid_size_z):
                            
                            grid_v = self.grid_v[grid_idx]
                            new_v += weight * grid_v
                            new_C += 4.0 * self.inv_dx[None] * weight * grid_v.outer_product(dpos)
            
            # Update particle velocity with damping
            self.v[p] = new_v * self.damping
            
            # Clamp velocity to reasonable values (prevent explosion)
            v_norm = self.v[p].norm()
            if v_norm > 10.0:
                self.v[p] *= 10.0 / v_norm
            
            self.C[p] = new_C
            
            # Update particle position
            self.x[p] += self.dt * self.v[p]
            
            # Update deformation gradient (small strain)
            # For small strain, F stays close to identity
            # F_new = (I + dt * grad_v) * F
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * new_C) @ self.F[p]
            
            # Keep F close to identity for small strain assumption
            # If |F - I| becomes too large, reset towards identity
            F_dev = self.F[p] - ti.Matrix.identity(ti.f32, 3)
            F_dev_norm = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    F_dev_norm += F_dev[i, j] * F_dev[i, j]
            
            # If strain is too large, apply stronger damping
            if F_dev_norm > 0.01:  # Strain > 10%
                self.F[p] = ti.Matrix.identity(ti.f32, 3) + F_dev * 0.1
    
    @ti.kernel
    def compute_von_mises(self):
        """Compute von Mises stress for each particle"""
        for p in range(self.n_particles):
            s = self.stress[p]
            
            # Deviatoric stress
            s_mean = (s[0, 0] + s[1, 1] + s[2, 2]) / 3.0
            s_dev = s - s_mean * ti.Matrix.identity(ti.f32, 3)
            
            # Von Mises stress = sqrt(3/2 * s_dev : s_dev)
            vm = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    vm += s_dev[i, j] * s_dev[i, j]
            
            self.von_mises[p] = ti.sqrt(1.5 * vm)
    
    def setup_visualization(self):
        """Setup Taichi GGUI for 3D visualization"""
        # Create window
        self.window = ti.ui.Window("MPM Spine Simulation", (1280, 960), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        
        # Camera setup - position to view spine from angle
        # Spine bounding box center
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        bbox_size = self.bbox_max - self.bbox_min
        
        # Position camera to see the whole spine
        camera_distance = np.max(bbox_size) * 2.0
        self.camera.position(self.bbox_center[0] + camera_distance * 0.7, 
                            self.bbox_center[1] - camera_distance * 0.7,
                            self.bbox_center[2] + camera_distance * 0.5)
        self.camera.lookat(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
        self.camera.up(0, 0, 1)  # Z is up
        
        # Particle colors based on stress (will update each frame)
        self.particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        
        print("\nVisualization Controls:")
        print("  Right mouse drag: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  ESC: Exit visualization")
        print("  SPACE: Pause/Resume simulation")
        print(f"\nCamera initial position: looking at spine center {self.bbox_center}")
    
    @ti.kernel
    def update_particle_colors(self, max_stress: ti.f32):
        """Update particle colors based on von Mises stress"""
        for p in range(self.n_particles):
            # Normalize stress to [0, 1]
            stress_norm = ti.min(self.von_mises[p] / max_stress, 1.0)
            
            # Jet colormap: blue -> cyan -> green -> yellow -> red
            if stress_norm < 0.25:
                # Blue to cyan
                t = stress_norm / 0.25
                self.particle_colors[p] = ti.Vector([0.0, t, 1.0])
            elif stress_norm < 0.5:
                # Cyan to green
                t = (stress_norm - 0.25) / 0.25
                self.particle_colors[p] = ti.Vector([0.0, 1.0, 1.0 - t])
            elif stress_norm < 0.75:
                # Green to yellow
                t = (stress_norm - 0.5) / 0.25
                self.particle_colors[p] = ti.Vector([t, 1.0, 0.0])
            else:
                # Yellow to red
                t = (stress_norm - 0.75) / 0.25
                self.particle_colors[p] = ti.Vector([1.0, 1.0 - t, 0.0])
    
    def step(self, force_direction, current_time):
        """Perform one MPM time step"""
        self.clear_grid()
        self.particle_to_grid()
        self.grid_update(force_direction, current_time)
        self.grid_to_particle()
        self.compute_von_mises()  # Compute stress at each step for visualization
    
    def render(self):
        """Render current simulation state"""
        # Update colors based on current stress
        von_mises_np = self.von_mises.to_numpy()
        max_stress = np.percentile(von_mises_np, 95)  # Use 95th percentile for better color range
        if max_stress > 1e-6:  # Avoid division by zero
            self.update_particle_colors(max_stress)
        
        # Update camera with current particle center
        x_np = self.x.to_numpy()
        current_center = np.mean(x_np, axis=0)
        
        # Update camera to look at current particle center
        self.camera.lookat(current_center[0], current_center[1], current_center[2])
        self.camera.track_user_inputs(self.window, movement_speed=0.003, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        
        # Set lighting
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.0, 0.5, 1.0), color=(1, 1, 1))
        
        # Render particles with larger size for better visibility
        self.scene.particles(self.x, radius=0.002, per_vertex_color=self.particle_colors)
        
        # Draw to canvas
        self.canvas.scene(self.scene)
        self.window.show()
    
    def run_simulation(self, load_case="compression", n_steps=1000, visualize=True):
        """Run MPM simulation for specified load case"""
        print(f"\nRunning MPM simulation: {load_case}")
        print(f"Steps: {n_steps}, dt: {self.dt}")
        print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
        
        # Define force direction based on load case
        force_directions = {
            "compression": ti.Vector([0.0, 0.0, -1.0]),
            "flexion": ti.Vector([0.0, 1.0, 0.0]),
            "extension": ti.Vector([0.0, -1.0, 0.0]),
            "lateral_left": ti.Vector([1.0, 0.0, 0.0]),
            "lateral_right": ti.Vector([-1.0, 0.0, 0.0])
        }
        
        force_dir = force_directions.get(load_case, ti.Vector([0.0, 0.0, -1.0]))
        print(f"Force direction: {force_dir}")
        
        # Time stepping
        step = 0
        paused = False
        current_time = 0.0
        
        while step < n_steps:
            if visualize:
                # Handle window events
                for e in self.window.get_events(ti.ui.PRESS):
                    if e.key == ti.ui.ESCAPE:
                        print("\nVisualization interrupted by user")
                        return
                    elif e.key == ti.ui.SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                
                if not paused:
                    self.step(force_dir, current_time)
                    step += 1
                    current_time += self.dt
                    
                    # Progress update
                    if step % 200 == 0:
                        print(f"  Step {step}/{n_steps} (t={current_time:.6f}s) - Force nodes: {self.force_nodes_count[None]}")
                
                # Render every frame
                self.render()
                
                # Check if window is closed
                if not self.window.running:
                    break
            else:
                # No visualization - run faster
                self.step(force_dir, current_time)
                step += 1
                current_time += self.dt
                
                if step % 200 == 0:
                    print(f"  Step {step}/{n_steps} (t={current_time:.6f}s) - Force nodes: {self.force_nodes_count[None]}")
        
        # Compute final von Mises stress (if not already computed)
        if not visualize:
            self.compute_von_mises()
        
        # Print statistics
        self.print_statistics()
        
    def print_statistics(self):
        """Print stress and displacement statistics"""
        von_mises_np = self.von_mises.to_numpy()
        x_np = self.x.to_numpy()
        
        print(f"\nVon Mises Stress Statistics:")
        print(f"  Min: {np.min(von_mises_np):.2e} Pa ({np.min(von_mises_np)/1e6:.2f} MPa)")
        print(f"  Max: {np.max(von_mises_np):.2e} Pa ({np.max(von_mises_np)/1e6:.2f} MPa)")
        print(f"  Mean: {np.mean(von_mises_np):.2e} Pa ({np.mean(von_mises_np)/1e6:.2f} MPa)")
        
        print(f"\nDisplacement Statistics:")
        print(f"  Max Z displacement: {np.max(x_np[:, 2]) - self.bbox_max[2]:.6f} m")
        print(f"  Min Z displacement: {np.min(x_np[:, 2]) - self.bbox_min[2]:.6f} m")
    
    def save_results(self, output_file="results_mpm.vtk"):
        """Save particle data to VTK file"""
        print(f"\nSaving results to {output_file}...")
        
        # Get particle data as numpy arrays
        positions = self.x.to_numpy()
        velocities = self.v.to_numpy()
        von_mises = self.von_mises.to_numpy()
        mat_labels = self.mat_label.to_numpy()
        E_values = self.E.to_numpy()
        
        # Create point data dictionary
        point_data = {
            "velocity": velocities,
            "von_mises_stress": von_mises,
            "material_label": mat_labels,
            "youngs_modulus": E_values
        }
        
        # Create meshio points object (particles as points)
        cells = []  # No cells, just points
        
        # Write to VTK
        mesh = meshio.Mesh(
            points=positions,
            cells=cells,
            point_data=point_data
        )
        
        mesh.write(output_file)
        print(f"Results saved successfully!")


def main():
    """Main function to run MPM simulation"""
    parser = argparse.ArgumentParser(description="MPM Spine Simulation")
    parser.add_argument("load_case", nargs="?", default="compression",
                        choices=["compression", "flexion", "extension", 
                                "lateral_left", "lateral_right", "all"],
                        help="Load case to simulate")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps")
    parser.add_argument("--grid-res", type=int, default=48,
                        help="Grid resolution")
    parser.add_argument("--mesh", type=str, default="mesh.xdmf",
                        help="Input mesh file")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable GUI visualization")
    
    args = parser.parse_args()
    
    # Create simulator
    sim = MPMSimulator(mesh_file=args.mesh, grid_res=args.grid_res)
    
    # Determine if visualization should be enabled
    visualize = not args.no_gui
    
    # Run simulation(s)
    if args.load_case == "all":
        # For "all" mode, disable visualization by default (too many cases)
        visualize = False
        print("\nRunning all load cases (visualization disabled for batch mode)\n")
        
        load_cases = ["compression", "flexion", "extension", "lateral_left", "lateral_right"]
        for case in load_cases:
            print(f"\n{'='*60}")
            print(f"Running load case: {case}")
            print(f"{'='*60}")
            
            # Re-initialize particles for each case
            sim.initialize_particles()
            
            # Run simulation
            sim.run_simulation(load_case=case, n_steps=args.steps, visualize=visualize)
            
            # Save results
            output_file = f"results_mpm_{case}.vtk"
            sim.save_results(output_file)
    else:
        # Run single load case
        sim.run_simulation(load_case=args.load_case, n_steps=args.steps, visualize=visualize)
        
        # Save results
        output_file = f"results_mpm_{args.load_case}.vtk"
        sim.save_results(output_file)
    
    print("\n" + "="*60)
    print("MPM simulation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
