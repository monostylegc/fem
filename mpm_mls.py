import taichi as ti
import numpy as np
import meshio
import argparse
import time

# Initialize Taichi
ti.init(arch=ti.gpu)

@ti.data_oriented
class MLS_MPM_Simulator:
    def __init__(self, mesh_file="mesh.xdmf", grid_res=64):
        self.mesh_file = mesh_file
        self.grid_res = grid_res
        
        # Load mesh to determine bounds
        print(f"Loading mesh from {mesh_file}...")
        self.mesh = meshio.read(mesh_file)
        self.points = self.mesh.points * 0.001  # Convert mm to m
        self.cells = self.mesh.cells[0].data
        
        # Calculate bounding box
        self.bbox_min = np.min(self.points, axis=0)
        self.bbox_max = np.max(self.points, axis=0)
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        self.bbox_size = self.bbox_max - self.bbox_min
        
        print(f"Bounding box: {self.bbox_min} to {self.bbox_max}")
        
        # Grid setup
        # Add padding
        padding = 0.1 * np.max(self.bbox_size)
        self.grid_origin = ti.Vector(self.bbox_min - padding)
        self.grid_width = np.max(self.bbox_size) + 2 * padding
        
        self.dx = self.grid_width / grid_res
        self.inv_dx = 1.0 / self.dx
        
        self.grid_size = ti.Vector([grid_res, grid_res, grid_res])
        print(f"Grid setup: resolution={grid_res}, dx={self.dx:.6f} m")
        
        # Simulation parameters
        self.dt = 1e-5  # Larger dt possible with MLS-MPM
        self.gravity = ti.Vector([0, 0, 0.0])  # Start with 0 gravity
        self.E_bone = 10e9  # 10 GPa
        self.nu_bone = 0.3
        self.E_disc = 5e6   # 5 MPa
        self.nu_disc = 0.45
        
        # Particle setup
        self.n_particles = len(self.cells) * 4  # 4 particles per tet
        print(f"Initializing {self.n_particles} particles...")
        
        # Fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)
        
        # Material properties
        self.E = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.nu = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.mu = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.la = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.material_type = ti.field(dtype=ti.i32, shape=self.n_particles) # 0: bone, 1: disc
        
        # Grid fields
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(grid_res, grid_res, grid_res))
        self.grid_m = ti.field(dtype=ti.f32, shape=(grid_res, grid_res, grid_res))
        
        # Visualization
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.von_mises = ti.field(dtype=ti.f32, shape=self.n_particles)
        
        # Initialize particles
        self.initialize_particles()
        
        # Applied force
        self.applied_force = 3000.0
        
        # Visualization setup
        self.window = None
        
    def initialize_particles(self):
        # Generate particles from mesh (4 per tet) using Numpy
        print("Generating particles on CPU...")
        
        # Material assignment based on Z height
        z_min, z_max = self.bbox_min[2], self.bbox_max[2]
        disc_z_min = z_min + 0.4 * (z_max - z_min)
        disc_z_max = z_min + 0.6 * (z_max - z_min)
        
        # Barycentric coordinates
        s = (5.0 - np.sqrt(5.0)) / 20.0
        t = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
        bary_coords = np.array([
            [s, s, s, t],
            [s, s, t, s],
            [s, t, s, s],
            [t, s, s, s]
        ], dtype=np.float32)
        
        # Prepare Numpy arrays
        num_cells = len(self.cells)
        total_particles = num_cells * 4
        
        x_np = np.zeros((total_particles, 3), dtype=np.float32)
        material_type_np = np.zeros(total_particles, dtype=np.int32)
        E_np = np.zeros(total_particles, dtype=np.float32)
        nu_np = np.zeros(total_particles, dtype=np.float32)
        mu_np = np.zeros(total_particles, dtype=np.float32)
        la_np = np.zeros(total_particles, dtype=np.float32)
        
        # Vectorized initialization
        # cell_points: (num_cells, 4, 3)
        cell_points = self.points[self.cells]
        
        # Centers: (num_cells, 3)
        centers = np.mean(cell_points, axis=1)
        
        # Determine materials
        is_disc = (centers[:, 2] > disc_z_min) & (centers[:, 2] < disc_z_max)
        is_disc_expanded = np.repeat(is_disc, 4)
        
        # Assign properties
        E_np[:] = self.E_bone
        nu_np[:] = self.nu_bone
        material_type_np[:] = 0
        
        E_np[is_disc_expanded] = self.E_disc
        nu_np[is_disc_expanded] = self.nu_disc
        material_type_np[is_disc_expanded] = 1
        
        # Compute Lame parameters
        mu_np = E_np / (2 * (1 + nu_np))
        la_np = E_np * nu_np / ((1 + nu_np) * (1 - 2 * nu_np))
        
        # Compute positions
        # We need to repeat cell_points 4 times for each barycentric coord
        # But easier to just loop 4 times
        for k in range(4):
            # b: (4,)
            b = bary_coords[k]
            # pos = b0*v0 + b1*v1 + b2*v2 + b3*v3
            # cell_points: (N, 4, 3)
            # pos: (N, 3)
            pos = b[0]*cell_points[:, 0, :] + \
                  b[1]*cell_points[:, 1, :] + \
                  b[2]*cell_points[:, 2, :] + \
                  b[3]*cell_points[:, 3, :]
            
            # Fill arrays
            # Indices: k, k+4, k+8... no, it's cell_idx*4 + k
            indices = np.arange(num_cells) * 4 + k
            x_np[indices] = pos
            
        # Transfer to Taichi fields
        print("Transferring data to GPU...")
        self.x.from_numpy(x_np)
        self.material_type.from_numpy(material_type_np)
        self.E.from_numpy(E_np)
        self.nu.from_numpy(nu_np)
        self.mu.from_numpy(mu_np)
        self.la.from_numpy(la_np)
        
        # Initialize F to identity
        # We can use a kernel for this or just fill numpy
        F_np = np.zeros((total_particles, 3, 3), dtype=np.float32)
        F_np[:, 0, 0] = 1.0
        F_np[:, 1, 1] = 1.0
        F_np[:, 2, 2] = 1.0
        self.F.from_numpy(F_np)
        
        self.J.fill(1.0)
        print("Initialization complete.")
    
    @ti.kernel
    def substep(self, force_dir: ti.types.vector(3, ti.f32), current_time: ti.f32):
        # Clear grid
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = [0, 0, 0]
            self.grid_m[I] = 0
            
        # P2G
        for p in self.x:
            base = (self.x[p] - self.grid_origin) * self.inv_dx
            base_i = int(base)
            fx = base - base_i
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            # Fixed Corotated Model
            F = self.F[p]
            U, sig, V = ti.svd(F)
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                J *= new_sig
            
            # Stress
            mu = self.mu[p]
            la = self.la[p]
            
            stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose() + \
                     ti.Matrix.identity(ti.f32, 3) * la * J * (J - 1)
            
            stress = (-self.dt * 4 * self.inv_dx * self.inv_dx) * stress * 1e-4 # Volume scaling approx
            affine = stress + 1.0 * self.C[p] # mass = 1.0 approx
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                
                idx = base_i + offset
                if 0 <= idx[0] < self.grid_res and 0 <= idx[1] < self.grid_res and 0 <= idx[2] < self.grid_res:
                    self.grid_v[idx] += weight * (self.v[p] + affine @ dpos)
                    self.grid_m[idx] += weight
        
        # Grid Update
        # Force ramping
        ramp = ti.min(current_time / 0.1, 1.0)
        current_force = self.applied_force * ramp
        
        # Top surface threshold
        z_top = self.bbox_max[2] - 0.02 # Top 2cm
        
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                # Momentum to velocity
                self.grid_v[I] /= self.grid_m[I]
                
                # Gravity
                self.grid_v[I] += self.dt * self.gravity
                
                # External Force
                # Convert grid index to world pos
                world_pos = self.grid_origin + I.cast(float) * self.dx
                
                if world_pos[2] > z_top:
                     # Apply force (acceleration)
                     # Simple constant acceleration for stability
                     accel = force_dir * 10.0 * ramp # 10 m/s^2
                     self.grid_v[I] += self.dt * accel
                
                # Boundary Conditions
                # Box confinement
                if I[0] < 2 and self.grid_v[I][0] < 0: self.grid_v[I][0] = 0
                if I[0] > self.grid_res - 3 and self.grid_v[I][0] > 0: self.grid_v[I][0] = 0
                if I[1] < 2 and self.grid_v[I][1] < 0: self.grid_v[I][1] = 0
                if I[1] > self.grid_res - 3 and self.grid_v[I][1] > 0: self.grid_v[I][1] = 0
                
                # Bottom fixed
                if world_pos[2] < self.bbox_min[2] + 0.02:
                    self.grid_v[I] = [0, 0, 0]
                    
        # G2P
        for p in self.x:
            base = (self.x[p] - self.grid_origin) * self.inv_dx
            base_i = int(base)
            fx = base - base_i
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                
                idx = base_i + offset
                if 0 <= idx[0] < self.grid_res and 0 <= idx[1] < self.grid_res and 0 <= idx[2] < self.grid_res:
                    g_v = self.grid_v[idx]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            
            self.v[p] = new_v
            self.x[p] += self.dt * self.v[p]
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * new_C) @ self.F[p]
            
            # Update von Mises for visualization
            F = self.F[p]
            U, sig, V = ti.svd(F)
            J = 1.0
            for d in ti.static(range(3)):
                J *= sig[d, d]
            stress = 2 * self.mu[p] * (F - U @ V.transpose()) @ F.transpose() + \
                     ti.Matrix.identity(ti.f32, 3) * self.la[p] * J * (J - 1)
            
            # Von Mises approx
            dev_stress = stress - stress.trace() / 3 * ti.Matrix.identity(ti.f32, 3)
            von_mises = ti.sqrt(1.5 * (dev_stress * dev_stress).sum())
            self.von_mises[p] = von_mises

    def setup_visualization(self):
        self.window = ti.ui.Window("MLS-MPM Spine", (1024, 768), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        
        camera_dist = np.max(self.bbox_size) * 2.0
        self.camera.position(self.bbox_center[0], self.bbox_center[1] - camera_dist, self.bbox_center[2])
        self.camera.lookat(self.bbox_center[0], self.bbox_center[1], self.bbox_center[2])
        self.camera.up(0, 0, 1)

    @ti.kernel
    def update_colors(self):
        max_stress = 1e6 # 1 MPa visualization scale
        for p in self.colors:
            s = self.von_mises[p] / max_stress
            s = ti.min(s, 1.0)
            self.colors[p] = ti.Vector([s, 0.5, 1.0 - s])

    def save_results(self, filename):
        print(f"Saving results to {filename}...")
        x_np = self.x.to_numpy()
        v_np = self.v.to_numpy()
        von_mises_np = self.von_mises.to_numpy()
        material_type_np = self.material_type.to_numpy()
        
        # Create PyVista PolyData
        import pyvista as pv
        cloud = pv.PolyData(x_np)
        cloud.point_data["velocity"] = v_np
        cloud.point_data["von_mises_stress"] = von_mises_np
        cloud.point_data["material_type"] = material_type_np
        
        cloud.save(filename)
        print("Results saved successfully!")

    def run(self, load_case="compression", steps=1000, visualize=True):
        if visualize:
            self.setup_visualization()
            
        force_dir = ti.Vector([0.0, 0.0, -1.0])
        if load_case == "flexion": force_dir = ti.Vector([0.0, 1.0, 0.0])
        elif load_case == "extension": force_dir = ti.Vector([0.0, -1.0, 0.0])
        elif load_case == "lateral_left": force_dir = ti.Vector([1.0, 0.0, 0.0])
        elif load_case == "lateral_right": force_dir = ti.Vector([-1.0, 0.0, 0.0])
        
        current_time = 0.0
        for s in range(steps):
            self.substep(force_dir, current_time)
            current_time += self.dt
            
            if s % 50 == 0:
                print(f"Step {s}/{steps}")
            
            if visualize:
                self.update_colors()
                self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
                self.scene.set_camera(self.camera)
                self.scene.ambient_light((0.5, 0.5, 0.5))
                self.scene.point_light(pos=(0, 0, 2), color=(1, 1, 1))
                self.scene.particles(self.x, radius=0.002, per_vertex_color=self.colors)
                self.canvas.scene(self.scene)
                self.window.show()
                if not self.window.running:
                    break
        
        # Save results
        self.save_results(f"results_mpm_{load_case}.vtk")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_case", default="compression")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    
    sim = MLS_MPM_Simulator()
    sim.run(args.load_case, args.steps, not args.no_gui)
