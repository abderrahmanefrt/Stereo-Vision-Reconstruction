

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys


LEFT_IMG  = 'image_left_undist4.jpg'
RIGHT_IMG = 'image_right_undist4.jpg'
BASELINE  = 150.0  # mm

CALIB_W = 3024
CALIB_H = 4032

IMG_W = 2880
IMG_H = 2160


#ÉTAPE 2:CHARGEMENT DES IMAGES
def load_images():
    for path in [LEFT_IMG, RIGHT_IMG]:
        if not os.path.exists(path):
            print(f"Fichier introuvable : {path}")
            sys.exit(1)

    img_l = cv2.imread(LEFT_IMG)
    img_r = cv2.imread(RIGHT_IMG)

    if img_l is None or img_r is None:
        print("Impossible de lire les images")
        sys.exit(1)

    h_real, w_real = img_l.shape[:2]
    real_W = max(w_real, h_real)
    real_H = min(w_real, h_real)

    print(f"\n [Étape 2] Images chargées")
    print(f"   shape opencv  : {w_real}×{h_real} px")
    print(f"   Paysage       : {real_W}×{real_H} px")

    global IMG_W, IMG_H
    IMG_W, IMG_H = real_W, real_H

    diff = np.mean(np.abs(img_l.astype(float) - img_r.astype(float)))
    if diff < 1.0:
        print("     Les deux images semblent identiques !")

    return img_l, img_r


# ÉTAPE 1 : CHARGEMENT DE K
def get_camera_matrix():
    if not os.path.exists('camera_K.npy'):
        print("camera_K.npy introuvable")
        sys.exit(1)

    K_orig = np.load('camera_K.npy')
    dist   = np.load('camera_dist.npy') if os.path.exists('camera_dist.npy') \
             else np.zeros(5)

    print(f"\n [Étape 1] K chargé (PORTRAIT {CALIB_W}×{CALIB_H})")
    

    fx_orig = K_orig[0, 0]
    fy_orig = K_orig[1, 1]
    cx_orig = K_orig[0, 2]
    cy_orig = K_orig[1, 2]

    # Rotation portrait → paysage
    fx_new = fy_orig * (IMG_W / CALIB_H)
    fy_new = fx_orig * (IMG_H / CALIB_W)
    cx_new = cy_orig * (IMG_W / CALIB_H)
    cy_new = cx_orig * (IMG_H / CALIB_W)

    K = np.array([[fx_new, 0,      cx_new],
                  [0,      fy_new, cy_new],
                  [0,      0,      1     ]], dtype=np.float64)

    print(f"\n   K corrigé (PAYSAGE {IMG_W}×{IMG_H})")
    

   
    

    return K, dist


#ÉTAPE 3 : SIFT + MATCHING 
def detect_and_match(img_l, img_r, K):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    
    sift      = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)
    print(f"\n [Étape 3] SIFT : {len(kp1)} kp gauche | {len(kp2)} kp droite")

    if len(kp1) < 10 or len(kp2) < 10:
        print("Trop peu de keypoints")
        sys.exit(1)

    flann   = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 100})
    matches = flann.knnMatch(des1, des2, k=2)

    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"   Après ratio test Lowe (0.75) : {len(good)} matches")

    pts_l = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask_e = cv2.findEssentialMat(
        pts_l, pts_r, K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    m_e   = mask_e.ravel() == 1
    pts_l = pts_l[m_e]
    pts_r = pts_r[m_e]
    print(f"   Après RANSAC (EssentialMat)  : {len(pts_l)} correspondances")

    _, R, t, mask_rp = cv2.recoverPose(E, pts_l, pts_r, K)
    m_rp  = mask_rp.ravel() > 0
    pts_l = pts_l[m_rp]
    pts_r = pts_r[m_rp]
    print(f"   Après recoverPose            : {len(pts_l)} points valides")
    print(f"   ||t|| = {np.linalg.norm(t):.4f}  →  "
          f"facteur d'échelle = {BASELINE / np.linalg.norm(t):.2f}×")

    h_img, w_img = img_l.shape[:2]
    K_inv = np.linalg.inv(K)
    F     = K_inv.T @ E @ K_inv

    ret_rect, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_l.reshape(-1, 1, 2),
        pts_r.reshape(-1, 1, 2),
        F, imgSize=(w_img, h_img)
    )

    dy_before = np.std(pts_l[:, 1] - pts_r[:, 1])

    if ret_rect:
        pts_l_rect = cv2.perspectiveTransform(pts_l.reshape(-1,1,2), H1).reshape(-1,2)
        pts_r_rect = cv2.perspectiveTransform(pts_r.reshape(-1,1,2), H2).reshape(-1,2)
        dy_after   = np.std(pts_l_rect[:, 1] - pts_r_rect[:, 1])

        print(f"\n   [Rectification ]")
        print(f"   σ(dy) avant : {dy_before:.2f} px  →  après : {dy_after:.2f} px")

        if dy_after < dy_before:
            pts_l_use = pts_l_rect
            pts_r_use = pts_r_rect
            K_rect_l  = H1 @ K   
            K_rect_r  = H2 @ K
        else:
            pts_l_use = pts_l
            pts_r_use = pts_r
            K_rect_l  = K
            K_rect_r  = K
    else:
        print(f"    stereoRectifyUncalibrated échoué — points originaux")
        pts_l_use = pts_l
        pts_r_use = pts_r
        K_rect_l  = K
        K_rect_r  = K

    # Diagnostic disparité
    disp = pts_l_use[:, 0] - pts_r_use[:, 0]
    print(f"\n   [Diagnostic disparité]")
    print(f"   Min={disp.min():.1f}  Max={disp.max():.1f}  Moy={disp.mean():.1f} px")
    print(f"   d>0 : {(disp>0).sum()}  |  d<0 : {(disp<0).sum()}")
    # focale (fx)
    f = K_rect_l[0, 0]

# baseline
    B = BASELINE         # mm

    
    #formule cours
    Z_simple = (f * B) / disp
    print("Z :", np.median(Z_simple))

    # Visualisation matches
    good_e = [good[i] for i in range(len(good)) if m_e[i]]
    good_rp = [good_e[i] for i in range(len(good_e)) if m_rp[i]]
    img_matches = cv2.drawMatches(
        img_l, kp1, img_r, kp2, good_rp[:60], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite('sift_matches.png', img_matches)
    print("   sift_matches.png sauvegardé")

    return pts_l_use, pts_r_use, R, t, K_rect_l, K_rect_r


#ÉTAPE 4 : RECONSTRUCTION 3D
def reconstruct_3d(pts_l, pts_r, K_rect_l, K_rect_r, R, t):
    print(f"\n[Étape 4] Reconstruction 3D — baseline={BASELINE} mm")

    P1 = K_rect_l @ np.hstack([np.eye(3),   np.zeros((3, 1))])
    P2 = K_rect_r @ np.hstack([R,            t])

    pts4d = cv2.triangulatePoints(
        P1, P2,
        pts_l.T.astype(np.float64),
        pts_r.T.astype(np.float64)
    )

    w_coord = pts4d[3]
    valid   = np.abs(w_coord) > 0
    pts3d   = np.full((pts4d.shape[1], 3), np.nan)
    pts3d[valid] = (pts4d[:3, valid] / w_coord[valid]).T

  
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        scale  = BASELINE / t_norm
        pts3d *= scale
        print(f"   Facteur d'échelle appliqué : ×{scale:.2f}")

    # Garder Z > 0 et valeurs finies
    mask  = (pts3d[:, 2] > 0) & np.isfinite(pts3d).all(axis=1)
    pts3d = pts3d[mask]
    print(f"   Points avec Z > 0 : {len(pts3d)}")

    if len(pts3d) == 0:
        print("Aucun point 3D valide")
        sys.exit(1)

    # Filtre outliers 3σ sur Z
    z   = pts3d[:, 2]
    med = np.median(z)
    std = np.std(z)
    if std > 0:
        pts3d = pts3d[np.abs(z - med) < 3 * std]

    print(f"   Après filtre outliers : {len(pts3d)} points 3D")
    
    print(f"   Profondeur médiane : {np.median(pts3d[:,2]):.0f} mm "
          f"({np.median(pts3d[:,2])/10:.1f} cm)")

    return pts3d


#ÉTAPE 5 : VISUALISATION
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
    print(f"\n  [Étape 5] resultat_3d.png sauvegardé")
    plt.show()

    with open('nuage_points.ply', 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z_val in pts3d:
            f.write(f"{x:.3f} {y:.3f} {z_val:.3f}\n")
    print("  nuage_points.ply sauvegardé (ouvrir avec MeshLab)")


#EXÉCUTION
if __name__ == '__main__':

    print(f"\n{'═'*60}")
    print(f"  STÉRÉOVISION — baseline={BASELINE}mm")
    print(f"  Gauche  : {LEFT_IMG}")
    print(f"  Droite  : {RIGHT_IMG}")
    print(f"{'═'*60}")

    img_l, img_r                = load_images()
    K, dist                     = get_camera_matrix()
    pts_l, pts_r, R, t, Kl, Kr = detect_and_match(img_l, img_r, K)
    pts3d                       = reconstruct_3d(pts_l, pts_r, Kl, Kr, R, t)

    np.save('points_3d.npy', pts3d)
    np.savetxt('points_3d.txt', pts3d, fmt='%.3f', header='X(mm) Y(mm) Z(mm)')
    print(f"\n  points_3d.npy / points_3d.txt sauvegardés")

    print(f"\n{'═'*50}")
    print(f"  RÉSULTAT : {len(pts3d)} points 3D reconstruits")
    print(f"  Profondeur min : {pts3d[:,2].min():.0f} mm")
    print(f"  Profondeur max : {pts3d[:,2].max():.0f} mm")
    print(f"{'═'*50}\n")

    visualize_3d(pts3d)