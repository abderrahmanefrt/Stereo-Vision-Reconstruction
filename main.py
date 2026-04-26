"""
╔══════════════════════════════════════════════════════════════════╗
║   PIPELINE STÉRÉOVISION — VERSION FINALE CORRIGÉE               ║
║   Images : image_left_undist4.jpg / image_right_undist4.jpg      ║
║   Baseline : 90 mm                                               ║
║   CORRECTION : K portrait 3024×4032 → paysage 2880×2160         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
LEFT_IMG  = 'image_left_undist4.jpg'
RIGHT_IMG = 'image_right_undist4.jpg'
BASELINE  = 90.0   # mm

# Résolution des photos de calibration (téléphone en PORTRAIT)
CALIB_W = 3024   # largeur  portrait
CALIB_H = 4032   # hauteur  portrait

# Résolution de tes images stéréo (téléphone en PAYSAGE)
# NOTE : en paysage, largeur > hauteur
IMG_W = 2880     # largeur  paysage  (la GRANDE dimension)
IMG_H = 2160     # hauteur  paysage  (la PETITE dimension)


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 2 : CHARGEMENT DES IMAGES  (en premier pour détecter taille)
# ══════════════════════════════════════════════════════════════════
def load_images():
    for path in [LEFT_IMG, RIGHT_IMG]:
        if not os.path.exists(path):
            print(f"❌ Fichier introuvable : {path}")
            sys.exit(1)

    img_l = cv2.imread(LEFT_IMG)
    img_r = cv2.imread(RIGHT_IMG)

    if img_l is None or img_r is None:
        print("❌ Impossible de lire les images")
        sys.exit(1)

    # shape retourne (hauteur, largeur, canaux)
    h_real, w_real = img_l.shape[:2]

    # largeur = grande dimension, hauteur = petite dimension (paysage)
    real_W = max(w_real, h_real)
    real_H = min(w_real, h_real)

    print(f"\n✅ [Étape 2] Images chargées")
    print(f"   shape brut opencv : {w_real}×{h_real} px  (largeur×hauteur)")
    print(f"   Interprété comme  : {real_W}×{real_H} px  (W×H paysage)")

    global IMG_W, IMG_H
    IMG_W, IMG_H = real_W, real_H

    diff = np.mean(np.abs(img_l.astype(float) - img_r.astype(float)))
    print(f"   Différence gauche/droite : {diff:.2f}")
    if diff < 1.0:
        print("   ⚠️  Les deux images semblent identiques !")

    return img_l, img_r


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 1 : CHARGEMENT ET CORRECTION DE K
#  Portrait → Paysage : les axes X/Y sont ÉCHANGÉS
# ══════════════════════════════════════════════════════════════════
def get_camera_matrix():
    if not os.path.exists('camera_K.npy'):
        print("❌ camera_K.npy introuvable")
        sys.exit(1)

    K_orig = np.load('camera_K.npy')
    dist   = np.load('camera_dist.npy') if os.path.exists('camera_dist.npy') \
             else np.zeros(5)

    print(f"\n✅ [Étape 1] K chargé depuis camera_K.npy")
    print(f"   K original (PORTRAIT {CALIB_W}×{CALIB_H}) :")
    print(f"   fx={K_orig[0,0]:.1f}  fy={K_orig[1,1]:.1f}  "
          f"cx={K_orig[0,2]:.1f}  cy={K_orig[1,2]:.1f}")

    fx_orig = K_orig[0, 0]   # suit axe X portrait = CALIB_W (3024)
    fy_orig = K_orig[1, 1]   # suit axe Y portrait = CALIB_H (4032)
    cx_orig = K_orig[0, 2]   # centre X portrait
    cy_orig = K_orig[1, 2]   # centre Y portrait

    # ── Rotation portrait → paysage ───────────────────────────
    # En paysage : axe X image = ancien axe Y portrait
    #              axe Y image = ancien axe X portrait
    #
    # fx_new = fy_orig × (IMG_W / CALIB_H)
    #        = 3100.6  × (2880  / 4032  ) = 2214.7
    #
    # fy_new = fx_orig × (IMG_H / CALIB_W)
    #        = 3107.8  × (2160  / 3024  ) = 2219.9
    #
    # cx_new = cy_orig × (IMG_W / CALIB_H)
    #        = 2010.4  × (2880  / 4032  ) = 1436.0
    #
    # cy_new = cx_orig × (IMG_H / CALIB_W)
    #        = 1526.3  × (2160  / 3024  ) = 1089.6

    fx_new = fy_orig * (IMG_W / CALIB_H)
    fy_new = fx_orig * (IMG_H / CALIB_W)
    cx_new = cy_orig * (IMG_W / CALIB_H)
    cy_new = cx_orig * (IMG_H / CALIB_W)

    K = np.array([[fx_new,  0,      cx_new],
                  [0,       fy_new, cy_new],
                  [0,       0,      1     ]], dtype=np.float64)

    print(f"\n   K corrigé (PAYSAGE {IMG_W}×{IMG_H}) :")
    print(f"   fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # Vérifications
    fx_ok = abs(K[0,0] - K[1,1]) < 200
    cx_ok = abs(K[0,2] - IMG_W/2) < IMG_W * 0.15
    cy_ok = abs(K[1,2] - IMG_H/2) < IMG_H * 0.15
    print(f"\n   Vérifications :")
    print(f"   fx ≈ fy  ? {K[0,0]:.0f} ≈ {K[1,1]:.0f}  "
          f"{'✅' if fx_ok else '⚠️ trop différents'}")
    print(f"   cx ≈ W/2 ? {K[0,2]:.0f} ≈ {IMG_W//2}  "
          f"{'✅' if cx_ok else '⚠️'}")
    print(f"   cy ≈ H/2 ? {K[1,2]:.0f} ≈ {IMG_H//2}  "
          f"{'✅' if cy_ok else '⚠️'}")

    if not fx_ok:
        print("\n   ⚠️  fx ≠ fy — vérifie CALIB_W/CALIB_H dans la config")

    return K, dist


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 3 : SIFT + MATCHING
# ══════════════════════════════════════════════════════════════════
def detect_and_match(img_l, img_r):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    sift      = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)
    print(f"\n✅ [Étape 3] SIFT : {len(kp1)} kp gauche | {len(kp2)} kp droite")

    if len(kp1) < 10 or len(kp2) < 10:
        print("❌ Trop peu de keypoints")
        sys.exit(1)

    flann   = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 100})
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"   Après ratio test Lowe : {len(good)} matches")

    pts_l = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(
        pts_l, pts_r, cv2.FM_RANSAC,
        ransacReprojThreshold=2.0, confidence=0.99
    )
    m     = mask.ravel() == 1
    pts_l = pts_l[m]
    pts_r = pts_r[m]
    print(f"   Après RANSAC : {len(pts_l)} correspondances valides")

    disp = pts_l[:, 0] - pts_r[:, 0]
    print(f"\n   [Diagnostic disparité]")
    print(f"   Min={disp.min():.1f}  Max={disp.max():.1f}  Moy={disp.mean():.1f} px")
    print(f"   d>0 : {(disp>0).sum()}  |  d<0 : {(disp<0).sum()}")

    good_filtered = [good[i] for i in range(len(good)) if m[i]]
    img_matches   = cv2.drawMatches(
        img_l, kp1, img_r, kp2,
        good_filtered[:60], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite('sift_matches.png', img_matches)
    print("   💾 sift_matches.png sauvegardé")

    return pts_l, pts_r


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 4 : RECONSTRUCTION 3D
# ══════════════════════════════════════════════════════════════════
def reconstruct_3d(pts_l, pts_r, K):
    print(f"\n[Étape 4] Reconstruction 3D — baseline={BASELINE} mm")

    best_pts  = None
    best_n    = -1
    best_sign = None

    for sign in [+1, -1]:
        t  = np.array([[sign * BASELINE], [0.0], [0.0]])
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([np.eye(3), t])

        pts4d = cv2.triangulatePoints(
            P1, P2,
            pts_l.T.astype(np.float64),
            pts_r.T.astype(np.float64)
        )

        w_coord = pts4d[3]
        valid_w = np.abs(w_coord) > 1e-6
        pts3d   = np.full((pts4d.shape[1], 3), np.nan)
        pts3d[valid_w] = (pts4d[:3, valid_w] / w_coord[valid_w]).T

        mask = (pts3d[:, 2] > 0) & np.isfinite(pts3d).all(axis=1)
        n_ok = mask.sum()
        print(f"   sign={sign:+d} → {n_ok} points avec Z > 0")

        if n_ok > best_n:
            best_n    = n_ok
            best_pts  = pts3d[mask]
            best_sign = sign

    print(f"   ✅ Signe choisi : {best_sign:+d}")

    if best_n == 0:
        print("❌ Aucun point 3D valide")
        sys.exit(1)

    z   = best_pts[:, 2]
    med = np.median(z)
    std = np.std(z)
    if std > 0:
        best_pts = best_pts[np.abs(z - med) < 3 * std]

    print(f"   Après filtre outliers : {len(best_pts)} points 3D")
    print(f"\n   Statistiques :")
    print(f"   X : [{best_pts[:,0].min():.0f} ; {best_pts[:,0].max():.0f}] mm")
    print(f"   Y : [{best_pts[:,1].min():.0f} ; {best_pts[:,1].max():.0f}] mm")
    print(f"   Z : [{best_pts[:,2].min():.0f} ; {best_pts[:,2].max():.0f}] mm")
    print(f"   Profondeur médiane : {np.median(best_pts[:,2]):.0f} mm "
          f"({np.median(best_pts[:,2])/10:.1f} cm)")

    return best_pts


# ══════════════════════════════════════════════════════════════════
#  ÉTAPE 5 : VISUALISATION
# ══════════════════════════════════════════════════════════════════
def visualize_3d(pts3d):
    X, Y, Z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f'Reconstruction 3D — {len(pts3d)} points | Baseline={BASELINE}mm',
        fontsize=13, fontweight='bold'
    )

    ax1 = fig.add_subplot(131)
    sc1 = ax1.scatter(X, Z, c=Z, cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Z — profondeur (mm)')
    ax1.set_title('Vue de dessus (X-Z)')
    ax1.invert_yaxis()
    plt.colorbar(sc1, ax=ax1, shrink=0.7)

    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(Z, Y, c=Z, cmap='plasma', s=2, alpha=0.6)
    ax2.set_xlabel('Z — profondeur (mm)'); ax2.set_ylabel('Y (mm)')
    ax2.set_title('Vue latérale (Z-Y)')
    ax2.invert_yaxis()
    plt.colorbar(sc2, ax=ax2, shrink=0.7)

    ax3 = fig.add_subplot(133, projection='3d')
    sc3 = ax3.scatter(X, Z, -Y, c=Z, cmap='viridis', s=2, alpha=0.7)
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Z (mm)'); ax3.set_zlabel('Y (mm)')
    ax3.set_title('Vue 3D')
    ax3.view_init(elev=20, azim=-55)
    plt.colorbar(sc3, ax=ax3, label='Z (mm)', shrink=0.5)

    plt.tight_layout()
    plt.savefig('resultat_3d.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ [Étape 5] resultat_3d.png sauvegardé")
    plt.show()

    with open('nuage_points.ply', 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts3d:
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    print("✅ nuage_points.ply sauvegardé (ouvrir avec MeshLab)")


# ══════════════════════════════════════════════════════════════════
#  EXÉCUTION
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print(f"\n{'═'*60}")
    print(f"  STÉRÉOVISION — baseline={BASELINE}mm")
    print(f"  Gauche  : {LEFT_IMG}")
    print(f"  Droite  : {RIGHT_IMG}")
    print(f"  Calibration PORTRAIT {CALIB_W}×{CALIB_H} → PAYSAGE {IMG_W}×{IMG_H}")
    print(f"{'═'*60}")

    # ⚠️ load_images() EN PREMIER pour détecter la vraie taille
    img_l, img_r = load_images()
    K, dist      = get_camera_matrix()   # utilise IMG_W/IMG_H mis à jour
    pts_l, pts_r = detect_and_match(img_l, img_r)
    pts3d        = reconstruct_3d(pts_l, pts_r, K)

    np.save('points_3d.npy', pts3d)
    np.savetxt('points_3d.txt', pts3d, fmt='%.3f', header='X(mm) Y(mm) Z(mm)')
    print(f"\n💾 points_3d.npy / points_3d.txt sauvegardés")

    print(f"\n{'═'*50}")
    print(f"  RÉSULTAT : {len(pts3d)} points 3D reconstruits")
    print(f"  Profondeur min : {pts3d[:,2].min():.0f} mm")
    print(f"  Profondeur max : {pts3d[:,2].max():.0f} mm")
    print(f"{'═'*50}\n")

    visualize_3d(pts3d)