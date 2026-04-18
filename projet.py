"""


USAGE :
  python main_stereo.py --mode demo        # Démo avec données synthétiques
  python main_stereo.py --mode real        # Avec vos vraies images
  python main_stereo.py --baseline 120     # Spécifier la baseline en mm
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse, os, sys

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='Pipeline Stéréovision')
parser.add_argument('--mode',     default='demo', choices=['demo', 'real'])
parser.add_argument('--baseline', type=float, default=100.0,
                    help='Distance entre les deux positions (mm)')
parser.add_argument('--left',     default='image_left_undist.jpg')
parser.add_argument('--right',    default='image_right_undist.jpg')
args = parser.parse_args()

BASELINE = args.baseline
print(f"\n{'='*60}")
print(f"  PIPELINE STÉRÉOVISION — baseline = {BASELINE} mm")
print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 0 : MODE DÉMO — Génération de données synthétiques
# ═══════════════════════════════════════════════════════════════
def generate_synthetic_scene():
    """
    Crée une scène synthétique avec 3 boîtes et génère les deux images.
    Utile pour tester le pipeline sans matériel physique.
    """
    print("[DEMO] Génération de la scène synthétique avec 3 boîtes...")

    # Paramètres caméra synthétique
    W, H = 640, 480
    K = np.array([[600,   0, 320],
                  [  0, 600, 240],
                  [  0,   0,   1]], dtype=float)

    # Points 3D simulant les coins de 3 boîtes (en mm)
    # Boîte 1 : 80x80x80 mm, position (−150, 0, 800)
    # Boîte 2 : 60x60x100 mm, position (0, 0, 900)
    # Boîte 3 : 100x50x60 mm, position (160, 0, 850)
    pts_3d = []

    def add_box(cx, cy, cz, w, h, d, n=20):
        """Génère des points aléatoires sur les faces visibles d'une boîte"""
        for _ in range(n):
            face = np.random.choice(['top', 'front', 'right'])
            if face == 'top':
                x = cx + np.random.uniform(-w/2, w/2)
                y = cy - h
                z = cz + np.random.uniform(-d/2, d/2)
            elif face == 'front':
                x = cx + np.random.uniform(-w/2, w/2)
                y = cy + np.random.uniform(-h, 0)
                z = cz - d/2
            else:
                x = cx + w/2
                y = cy + np.random.uniform(-h, 0)
                z = cz + np.random.uniform(-d/2, d/2)
            pts_3d.append([x, y, z])

    add_box(-150,  50, 800,  80, 80,  80, n=80)
    add_box(   0,  50, 900,  60, 100, 60, n=80)
    add_box( 160,  50, 850, 100, 50,  60, n=80)

    pts_3d = np.array(pts_3d, dtype=float)
    print(f"  {len(pts_3d)} points 3D générés pour 3 boîtes")

    def project(pts, K, t_cam=np.zeros(3)):
        """Projette des points 3D en 2D"""
        pts_shifted = pts.copy()
        pts_shifted[:, 0] -= t_cam[0]  # translation caméra
        uvw = (K @ pts_shifted.T).T
        uv  = uvw[:, :2] / uvw[:, 2:3]
        return uv

    pts2d_left  = project(pts_3d, K, t_cam=[0, 0, 0])
    pts2d_right = project(pts_3d, K, t_cam=[BASELINE, 0, 0])

    # Ajout de bruit (~0.5px)
    noise = 0.5
    pts2d_left  += np.random.randn(*pts2d_left.shape)  * noise
    pts2d_right += np.random.randn(*pts2d_right.shape) * noise

    # Filtrer les points hors image
    def in_image(uv, W, H):
        return (uv[:, 0] > 0) & (uv[:, 0] < W) & \
               (uv[:, 1] > 0) & (uv[:, 1] < H)

    mask = in_image(pts2d_left, W, H) & in_image(pts2d_right, W, H)
    return K, pts2d_left[mask], pts2d_right[mask], pts_3d[mask]


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 1 : CALIBRATION (ou chargement)
# ═══════════════════════════════════════════════════════════════
def get_camera_matrix(mode):
    if mode == 'demo':
        K = np.array([[600,   0, 320],
                      [  0, 600, 240],
                      [  0,   0,   1]], dtype=float)
        print(f"[Étape 1] Matrice K synthétique utilisée (mode démo)")
        return K
    elif os.path.exists('camera_K.npy'):
        K = np.load('camera_K.npy')
        print(f"[Étape 1] ✅ Matrice K chargée depuis camera_K.npy")
        return K
    else:
        print("[ERREUR] Lancez d'abord step1_calibration.py !")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 2+3 : SIFT et MATCHING (ou données synthétiques)
# ═══════════════════════════════════════════════════════════════
def get_correspondences(mode, K):
    if mode == 'demo':
        _, pts_l, pts_r, pts3d_gt = generate_synthetic_scene()
        print(f"[Étapes 2-3] ✅ {len(pts_l)} correspondances synthétiques")
        return pts_l, pts_r, pts3d_gt

    # Mode réel
    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)
    if img_l is None or img_r is None:
        print(f"[ERREUR] Images non trouvées. Lancez step2_acquisition.py d'abord.")
        sys.exit(1)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY), None)

    flann   = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts_l = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in good])

    _, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, 3.0, 0.99)
    m = mask.ravel() == 1
    print(f"[Étapes 2-3] ✅ {m.sum()} correspondances SIFT après RANSAC")
    return pts_l[m], pts_r[m], None


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 4 : RECONSTRUCTION 3D
# ═══════════════════════════════════════════════════════════════
def reconstruct_3d(pts_l, pts_r, K, baseline):
    """Triangulation DLT via OpenCV"""
    t  = np.array([[-baseline], [0.0], [0.0]])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), t])

    pts4d = cv2.triangulatePoints(P1, P2,
                                   pts_l.T.astype(np.float64),
                                   pts_r.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3]).T
    valid = pts3d[:, 2] > 0
    pts3d = pts3d[valid]

    # Filtre outliers
    med = np.median(pts3d[:, 2])
    std = np.std(pts3d[:, 2])
    pts3d = pts3d[np.abs(pts3d[:, 2] - med) < 3 * std]

    print(f"[Étape 4] ✅ {len(pts3d)} points 3D reconstruits")
    return pts3d


# ═══════════════════════════════════════════════════════════════
# ÉTAPE 5 : VISUALISATION 3D
# ═══════════════════════════════════════════════════════════════
def visualize_3d(pts3d, pts3d_gt=None):
    fig = plt.figure(figsize=(15, 6))

    # Vue de face (X-Z)
    ax1 = fig.add_subplot(131)
    ax1.scatter(pts3d[:, 0], pts3d[:, 2], s=1, c='steelblue', alpha=0.5)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Z profondeur (mm)')
    ax1.set_title('Vue de dessus (X-Z)')
    ax1.invert_yaxis()

    # Vue latérale (Y-Z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(pts3d[:, 2], pts3d[:, 1], s=1, c='tomato', alpha=0.5)
    ax2.set_xlabel('Z profondeur (mm)'); ax2.set_ylabel('Y (mm)')
    ax2.set_title('Vue latérale (Z-Y)')
    ax2.invert_yaxis()

    # Vue 3D
    ax3 = fig.add_subplot(133, projection='3d')
    sc = ax3.scatter(pts3d[:, 0], pts3d[:, 2], -pts3d[:, 1],
                     c=pts3d[:, 2], cmap='viridis', s=2, alpha=0.7)
    if pts3d_gt is not None:
        ax3.scatter(pts3d_gt[:, 0], pts3d_gt[:, 2], -pts3d_gt[:, 1],
                    c='red', s=10, marker='x', label='Vérité terrain', alpha=0.3)
        ax3.legend()
    plt.colorbar(sc, ax=ax3, label='Z (mm)', shrink=0.5)
    ax3.set_xlabel('X'); ax3.set_ylabel('Z (profondeur)'); ax3.set_zlabel('Y')
    ax3.set_title('Nuage de points 3D')
    ax3.view_init(elev=25, azim=-50)

    plt.suptitle(f'Reconstruction 3D — {len(pts3d)} points | Baseline={BASELINE}mm',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('result_3d_reconstruction.png', dpi=150, bbox_inches='tight')
    print("[Étape 5] ✅ Visualisation sauvegardée : result_3d_reconstruction.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# EXÉCUTION DU PIPELINE
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    K              = get_camera_matrix(args.mode)
    pts_l, pts_r, pts3d_gt = get_correspondences(args.mode, K)
    pts3d          = reconstruct_3d(pts_l, pts_r, K, BASELINE)

    np.save('points_3d.npy', pts3d)
    np.savetxt('points_3d.txt', pts3d, fmt='%.3f', header='X(mm) Y(mm) Z(mm)')

    print(f"\n{'='*50}")
    print(f"  RÉSULTAT FINAL : {len(pts3d)} points 3D reconstruits")
    print(f"  Profondeur min : {pts3d[:,2].min():.0f} mm")
    print(f"  Profondeur max : {pts3d[:,2].max():.0f} mm")
    print(f"{'='*50}\n")

    visualize_3d(pts3d, pts3d_gt)