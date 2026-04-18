

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os

# ── CONFIGURATION ──────────────────────────────────────────
IMAGE_1          = "image_gauche.jpg"
IMAGE_2          = "image_droite.jpg"
BASELINE_MM      = 100.0               # distance entre les deux prises (mm)
DOSSIER_CALIB    = "calibration_images/"
TAILLE_ECHIQUIER = (9, 6)              # coins intérieurs de l'échiquier
TAILLE_CASE_MM   = 25                  # taille d'une case en mm
# ───────────────────────────────────────────────────────────


# ── ÉTAPE 1 : CALIBRATION ──────────────────────────────────
def calibrer_camera():
    print("\n--- ÉTAPE 1 : Calibration ---")

    criteres = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, lignes = TAILLE_ECHIQUIER

    # Points 3D réels dans le plan de l'échiquier (Z = 0)
    pts_obj = np.zeros((cols * lignes, 3), np.float32)
    pts_obj[:, :2] = np.mgrid[0:cols, 0:lignes].T.reshape(-1, 2) * TAILLE_CASE_MM

    liste_3d, liste_2d = [], []
    images = glob.glob(DOSSIER_CALIB + "*.jpg") + glob.glob(DOSSIER_CALIB + "*.png")

    # Matrice de démonstration si pas d'images de calibration
    if not images:
        print("Aucune image de calibration — matrice par défaut utilisée.")
        K    = np.array([[1500, 0, 640], [0, 1500, 360], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1))
        return K, dist

    taille_img = None
    for f in images:
        gris = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        taille_img = gris.shape[::-1]
        ok, coins = cv2.findChessboardCorners(gris, (cols, lignes), None)
        if ok:
            liste_3d.append(pts_obj)
            liste_2d.append(cv2.cornerSubPix(gris, coins, (11, 11), (-1, -1), criteres))
            print(f"  ✓ {os.path.basename(f)}")

    if len(liste_3d) < 5:
        print("Pas assez d'images valides — matrice par défaut utilisée.")
        K    = np.array([[1500, 0, 640], [0, 1500, 360], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((5, 1))
        return K, dist

    err, K, dist, _, _ = cv2.calibrateCamera(liste_3d, liste_2d, taille_img, None, None)
    print(f"  Erreur reprojection : {err:.4f} px | fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    return K, dist


# ── ÉTAPE 2 : CHARGEMENT DES IMAGES ────────────────────────
def charger_images(K, dist):
    print("\n--- ÉTAPE 2 : Chargement des images ---")

    img1 = cv2.imread(IMAGE_1)
    img2 = cv2.imread(IMAGE_2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Images stéréo introuvables. Vérifier les chemins.")

    h, w = img1.shape[:2]
    K_opt, _ = cv2.getOptimalNewCameraMatrix(dist, dist, (w, h), 1, (w, h))

    img1 = cv2.undistort(img1, K, dist, None, K_opt)
    img2 = cv2.undistort(img2, K, dist, None, K_opt)

    print(f"  Images chargées et distorsion corrigée ({w}x{h} px)")
    gris1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gris1, gris2


# ── ÉTAPE 3 : SIFT + MATCHING ──────────────────────────────
def sift_et_matching(gris1, gris2, img1, img2):
    print("\n--- ÉTAPE 3 : SIFT et mise en correspondance ---")

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gris1, None)
    kp2, desc2 = sift.detectAndCompute(gris2, None)
    print(f"  Keypoints : {len(kp1)} (img1) | {len(kp2)} (img2)")

    # Matching FLANN + test ratio de Lowe
    flann   = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(desc1, desc2, k=2)
    bons    = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"  Bons matches : {len(bons)} / {len(matches)}")

    if len(bons) < 8:
        raise ValueError("Pas assez de bons matches !")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in bons])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in bons])

    # Filtrage RANSAC via la matrice fondamentale
    _, masque = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    masque = masque.ravel().astype(bool)
    pts1, pts2 = pts1[masque], pts2[masque]
    print(f"  Après RANSAC : {len(pts1)} inliers")

    # Affichage
    bons_filtres = [bons[i] for i, ok in enumerate(masque) if ok]
    img_m = cv2.drawMatches(img1, kp1, img2, kp2, bons_filtres[:50], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16, 5))
    plt.imshow(cv2.cvtColor(img_m, cv2.COLOR_BGR2RGB))
    plt.title(f"Correspondances SIFT — {len(pts1)} points")
    plt.axis('off')
    plt.savefig("matches_sift.png", dpi=150, bbox_inches='tight')
    plt.show()

    return pts1, pts2


# ── ÉTAPE 4 : TRIANGULATION 3D ─────────────────────────────
def trianguler(pts1, pts2, K):
    print("\n--- ÉTAPE 4 : Triangulation 3D ---")

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), [[-BASELINE_MM], [0], [0]]])

    pts4D  = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D  = (pts4D[:3] / pts4D[3]).T

    # Garder uniquement les points devant la caméra
    pts3D  = pts3D[(pts3D[:, 2] > 0) & (pts3D[:, 2] < 10000)]
    print(f"  Points 3D : {len(pts3D)} | Z moyen : {pts3D[:,2].mean():.1f} mm")
    return pts3D


# ── ÉTAPE 5 : VISUALISATION ────────────────────────────────
def visualiser(pts3D):
    print("\n--- ÉTAPE 5 : Visualisation 3D ---")

    z_norm = (pts3D[:, 2] - pts3D[:, 2].min()) / (pts3D[:, 2].ptp() + 1e-6)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(pts3D[:, 0], pts3D[:, 2], pts3D[:, 1],
                     c=z_norm, cmap='plasma', s=2, alpha=0.7)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z - Profondeur (mm)")
    ax.set_zlabel("Y (mm)")
    ax.set_title("Nuage de points 3D — Stéréovision")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Profondeur normalisée")
    plt.savefig("nuage_points_3D.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Visualisation Open3D (optionnel)
    try:
        import open3d as o3d
        nuage        = o3d.geometry.PointCloud()
        nuage.points = o3d.utility.Vector3dVector(pts3D)
        nuage.colors = o3d.utility.Vector3dVector(plt.cm.plasma(z_norm)[:, :3])
        o3d.visualization.draw_geometries([nuage], window_name="Stéréovision 3D")
    except ImportError:
        print("  (Open3D non installé — visualisation matplotlib uniquement)")


# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    K, dist      = calibrer_camera()
    img1, img2, gris1, gris2 = charger_images(K, dist)
    pts1, pts2   = sift_et_matching(gris1, gris2, img1, img2)
    pts3D        = trianguler(pts1, pts2, K)
    visualiser(pts3D)
    print("\n✅ Terminé ! Fichiers : matches_sift.png | nuage_points_3D.png")