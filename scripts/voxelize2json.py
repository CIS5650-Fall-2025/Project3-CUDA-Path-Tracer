import argparse, json, numpy as np
import trimesh
import matplotlib.pyplot as plt

def visualize_voxels_idx(idx):
    if idx.size == 0:
        print("[!] No voxels to preview."); return
    mins = idx.min(axis=0)
    comp = idx - mins
    shape = comp.max(axis=0) + 1
    grid = np.zeros(shape, dtype=bool)
    grid[comp[:,0], comp[:,1], comp[:,2]] = True
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(grid, edgecolor='k')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout(); plt.show()

def fit_mesh_to_bbox(mesh: trimesh.Trimesh, bb_min, bb_max, pad=0.0):
    m_min, m_max = mesh.bounds
    m_size = m_max - m_min
    bb_min = np.asarray(bb_min, float)
    bb_max = np.asarray(bb_max, float)
    extent = bb_max - bb_min

    m_size = np.maximum(m_size, 1e-9)
    extent_eff = extent * (1.0 - pad*2.0)
    scale = float(np.min(extent_eff / m_size))

    m_center = 0.5 * (m_min + m_max)
    bb_center = 0.5 * (bb_min + bb_max)

    T = np.eye(4)
    T[:3,:3] *= scale
    T[:3, 3] = bb_center - scale * m_center
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh, scale, T

def main():
    ap = argparse.ArgumentParser(description="Voxelize OBJ with trimesh and append cubes to scene JSON.")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--bbox-min", nargs=3, type=float, required=True, metavar=('X','Y','Z'))
    ap.add_argument("--bbox-max", nargs=3, type=float, required=True, metavar=('X','Y','Z'))
    ap.add_argument("--grid",     nargs=3, type=int,   required=True, metavar=('NX','NY','NZ'))
    ap.add_argument("--in-json",  required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--material", default="diffuse_white")
    ap.add_argument("--solid", action="store_true", help="Fill interior voxels (watertight mesh recommended).")
    ap.add_argument("--preview", action="store_true", help="Show a quick matplotlib 3D preview.")
    ap.add_argument("--fit", choices=["none", "center", "bbox"], default="bbox")
    ap.add_argument("--pad", type=float, default=0.02)
    args = ap.parse_args()

    bb_min = np.array(args.bbox_min, dtype=float)
    bb_max = np.array(args.bbox_max, dtype=float)
    extent = bb_max - bb_min
    Nx, Ny, Nz = args.grid
    cell = np.array([extent[0]/Nx, extent[1]/Ny, extent[2]/Nz], dtype=float)

    print(f"[*] Loading mesh: {args.obj}")
    mesh = trimesh.load(args.obj, force='mesh')
    if mesh.is_empty:
        print("[!] Loaded mesh is empty. Abort."); return
    
    # angle = np.deg2rad(90)
    # R = trimesh.transformations.rotation_matrix(angle, [1,0,0], [0,0,0])
    # mesh.apply_transform(R)

    if args.fit == "bbox":
        mesh, scale, T = fit_mesh_to_bbox(mesh, bb_min, bb_max, pad=args.pad)
        print(f"[*] Fit mesh into bbox with uniform scale {scale:.6f} and placed at bbox center (pad={args.pad}).")
    elif args.fit == "center":
        m_center = mesh.bounding_box.centroid
        bb_center = 0.5 * (bb_min + bb_max)
        T = np.eye(4); T[:3,3] = bb_center - m_center
        mesh = mesh.copy(); mesh.apply_transform(T)
        print("[*] Centered mesh to bbox center (no scaling).")
    else:
        print("[*] Using original mesh coordinates (no fit).")

    pitch = float(np.min(cell))
    print(f"[*] Voxelizing with pitch={pitch:.6f} (cubic voxels)")
    vg = mesh.voxelized(pitch=pitch)
    if vg is None or vg.points.size == 0:
        print("[!] No voxels produced. Try a larger pitch (coarser grid) or check bbox/fit."); return
    if args.solid:
        print("[*] Filling interior…"); vg = vg.fill()

    centers = vg.points  # (N,3)

    mask = np.all((centers >= bb_min) & (centers <= bb_max), axis=1)
    centers = centers[mask]
    if centers.shape[0] == 0:
        print("[!] All voxels are outside the bbox — check bbox or pitch/grid."); return

    idxf = (centers - bb_min) / cell
    idx = np.floor(idxf + 0.5).astype(int)
    in_bounds = (idx[:,0]>=0)&(idx[:,0]<Nx)&(idx[:,1]>=0)&(idx[:,1]<Ny)&(idx[:,2]>=0)&(idx[:,2]<Nz)
    idx = idx[in_bounds]
    if idx.size == 0:
        print("[!] No in-bounds voxels after snapping."); return
    idx = np.unique(idx, axis=0)

    if args.preview:
        visualize_voxels_idx(idx)

    with open(args.in_json, "r", encoding="utf-8") as f:
        scene = json.load(f)
    if "Objects" not in scene or not isinstance(scene["Objects"], list):
        scene["Objects"] = []

    half_out = cell
    count = 0
    for i,j,k in idx:
        c = bb_min + (np.array([i,j,k], float) + 0.5) * cell
        obj = {
            "TYPE": "cube",
            "MATERIAL": args.material,
            "TRANS": [round(float(c[0]),6), round(float(c[1]),6), round(float(c[2]),6)],
            "ROTAT": [0.0, 0.0, 0.0],
            "SCALE": [round(float(half_out[0]),6), round(float(half_out[1]),6), round(float(half_out[2]),6)]
        }
        scene["Objects"].append(obj)
        count += 1

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=4)

    print(f"[✓] Appended {count} cubes to {args.out_json}")

if __name__ == "__main__":
    main()