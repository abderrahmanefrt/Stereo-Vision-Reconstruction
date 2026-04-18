"""
ÉTAPE 5 : Visualisation 3D du nuage de points
==============================================
Deux méthodes :
  A) Matplotlib (intégré, simple)
  B) Open3D (interactif, professionnel — pip install open3d)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
pts3d = np.load('points_3d.npy')
print(f"[INFO] {len(pts3d)} points 3D chargés")

X = pts3d[:, 0]
Y = pts3d[:, 1]
Z = pts3d[:, 2]

# ─────────────────────────────────────────────
# A) MATPLOTLIB — Vue nuage de points coloré par profondeur
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
ax  = fig.add_subplot(111, projection='3d')

# Colorer par profondeur Z (plus sombre = plus loin)
sc = ax.scatter(X, Z, -Y,
                c=Z,
                cmap='viridis',
                s=2,
                alpha=0.6)

plt.colorbar(sc, ax=ax, label='Profondeur Z (mm)', shrink=0.5)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Z — profondeur (mm)')
ax.set_zlabel('Y (mm)')
ax.set_title('Nuage de points 3D reconstruit\n(coloré par profondeur)', fontsize=13)

# Meilleure vue initiale
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.savefig('pointcloud_3d.png', dpi=150)
plt.show()
print("✅ Sauvegardé : pointcloud_3d.png")

# ─────────────────────────────────────────────
# B) OPEN3D — Visualisation interactive
# ─────────────────────────────────────────────
try:
    import open3d as o3d

    # Création du nuage de points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)

    # Colorier par profondeur Z (rouge=proche, bleu=loin)
    z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
    colors  = plt.cm.viridis(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimation des normales (améliore le rendu)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30)
    )

    # Visualisation interactive (souris pour tourner)
    print("\n[Open3D] Fenêtre interactive ouverte")
    print("  Clic gauche + glisser : rotation")
    print("  Clic molette + glisser : pan")
    print("  Molette : zoom")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Nuage de points 3D — Stéréovision",
        width=1024, height=768,
        point_show_normal=False
    )

    # Sauvegarde au format PLY (ouvert dans MeshLab, CloudCompare, Blender...)
    o3d.io.write_point_cloud("pointcloud.ply", pcd)
    print("💾 Sauvegardé : pointcloud.ply (ouvrir avec MeshLab ou CloudCompare)")

except ImportError:
    print("\n[INFO] Open3D non installé.")
    print("  Pour l'installer : pip install open3d")
    print("  Alternative : ouvrir points_3d.txt dans MeshLab ou CloudCompare")

# ─────────────────────────────────────────────
# EXPORT POUR MESHLAB / CLOUDCOMPARE
# ─────────────────────────────────────────────
def export_ply_manual(pts, filename='pointcloud.ply'):
    """Exporte un fichier PLY sans dépendance Open3D"""
    n = len(pts)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    print(f"💾 Exporté : {filename}")

export_ply_manual(pts3d, 'pointcloud_manual.ply')
