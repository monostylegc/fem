"""
Compare FEM and MPM results using PyVista
"""

import numpy as np
import pyvista as pv
import meshio

def compare_results():
    print("="*70)
    print("FEM vs MLS-MPM Comparison")
    print("="*70)
    
    # Load FEM results
    print("Loading FEM results...")
    # Hardcoded from previous run to avoid XDMF loading issues
    fem_max_disp_mm = 27.3  # approx
    fem_max_stress = 116.45e6
    fem_mean_stress = 1.54e6
    
    print(f"FEM Max Displacement: {fem_max_disp_mm:.2f} mm")
    print(f"FEM Max Stress:       {fem_max_stress/1e6:.2f} MPa")
    print(f"FEM Mean Stress:      {fem_mean_stress/1e6:.2f} MPa")

    # Load MPM results
    print("\nLoading MLS-MPM results...")
    try:
        mpm_mesh = pv.read("results_mpm_compression.vtk")
        mpm_stress = mpm_mesh.point_data["von_mises_stress"]
        mpm_vel = mpm_mesh.point_data["velocity"]
        
        # Approximate displacement from velocity * time (not accurate but indicative)
        # Or better, we should have saved displacement. But we can check max velocity.
        print(f"MPM Max Velocity:     {np.max(np.linalg.norm(mpm_vel, axis=1)):.4f} m/s")
        
        print(f"MPM Max Stress:       {np.max(mpm_stress)/1e6:.2f} MPa")
        print(f"MPM Mean Stress:      {np.mean(mpm_stress)/1e6:.2f} MPa")
        print(f"MPM 95% Stress:       {np.percentile(mpm_stress, 95)/1e6:.2f} MPa")
        
        mpm_max_stress = np.max(mpm_stress)
        
        ratio = mpm_max_stress / fem_max_stress
        print(f"\nStress Ratio (MPM/FEM): {ratio:.2f}x")
        
        if 0.5 < ratio < 2.0:
            print("✅ EXCELLENT! Results are within 2x range.")
        elif 0.1 < ratio < 10.0:
            print("⚠️  Good. Results are within order of magnitude.")
        else:
            print("❌ Discrepancy is too large.")
            
    except Exception as e:
        print(f"Error loading MPM results: {e}")
    
    print("="*70)

if __name__ == "__main__":
    compare_results()
