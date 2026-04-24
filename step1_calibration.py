"""
ÉTAPE 1 : Calibration de la caméra avec un damier
================================================
But : Obtenir la matrice intrinsèque K et les coefficients de distorsion
"""

import cv2
import numpy as np
import glob
import os

# ─────────────────────────────────────────────
# PARAMÈTRES DU DAMIER
# ─────────────────────────────────────────────
ROWS = 6          # nombre de coins internes en hauteur
COLS = 9          # nombre de coins internes en largeur
SQUARE_SIZE = 25  # taille d'un carré en mm (à adapter selon votre damier)

# Critères d'arrêt pour l'affinage des coins
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ─────────────────────────────────────────────
# PRÉPARATION DES POINTS 3D RÉELS DU DAMIER
# ─────────────────────────────────────────────
# Points dans le repère monde : Z=0 (damier plat)
# Exemple pour ROWS=6, COLS=9 :
# (0,0,0), (25,0,0), (50,0,0), ..., (200,125,0)
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_SIZE

# Listes pour stocker tous les points
objpoints = []  # points 3D dans le monde réel
imgpoints = []  # points 2D dans l'image

# ─────────────────────────────────────────────
# CHARGEMENT DES IMAGES DE CALIBRATION
# ─────────────────────────────────────────────
images = glob.glob('calibration_images/*.jpg') + glob.glob('calibration_images/*.png') + glob.glob('calibration_images/*.jpeg')
# Si vous n'avez pas d'images, utilisez la webcam :
# images = capture_calibration_images()  # voir fonction ci-dessous

print(f"[INFO] {len(images)} images de calibration trouvées")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Recherche des coins du damier
    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)

    if ret:
        objpoints.append(objp)

        # Affinage subpixel des coins (plus de précision)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners_refined)

        # Visualisation (optionnel)
        cv2.drawChessboardCorners(img, (COLS, ROWS), corners_refined, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# ─────────────────────────────────────────────
# CALCUL DE LA CALIBRATION
# ─────────────────────────────────────────────
h, w = gray.shape[:2]

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (w, h), None, None
)

print("\n" + "="*50)
print("✅ RÉSULTATS DE LA CALIBRATION")
print("="*50)
print(f"\n📐 Matrice intrinsèque K :")
print(f"   [ fx  0  cx ]   [ {K[0,0]:.2f}  0  {K[0,2]:.2f} ]")
print(f"   [  0 fy  cy ] = [  0  {K[1,1]:.2f}  {K[1,2]:.2f} ]")
print(f"   [  0  0   1 ]   [  0    0       1   ]")
print(f"\n   fx (focale x) = {K[0,0]:.2f} pixels")
print(f"   fy (focale y) = {K[1,1]:.2f} pixels")
print(f"   cx (centre x) = {K[0,2]:.2f} pixels")
print(f"   cy (centre y) = {K[1,2]:.2f} pixels")
print(f"\n🔍 Coefficients de distorsion :")
print(f"   {dist.ravel()}")

# Erreur de reprojection (doit être < 1 pixel)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print(f"\n📊 Erreur de reprojection moyenne : {mean_error/len(objpoints):.4f} px")
print("   (Bonne calibration si < 0.5 px)")

# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────
np.save('camera_K.npy', K)
np.save('camera_dist.npy', dist)
print("\n💾 Paramètres sauvegardés dans camera_K.npy et camera_dist.npy")


# ─────────────────────────────────────────────
# OPTIONNEL : Capturer les images depuis la webcam
# ─────────────────────────────────────────────
def capture_calibration_images(n=20, save_dir='calibration_images'):
    """Capture n images depuis la webcam pour la calibration"""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    print("Appuyez sur ESPACE pour capturer, Q pour quitter")
    while count < n:
        ret, frame = cap.read()
        cv2.putText(frame, f"Captures: {count}/{n} | ESPACE=capture Q=quitter",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capture calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = f"{save_dir}/calib_{count:03d}.jpg"
            cv2.imwrite(path, frame)
            print(f"  Sauvegardé : {path}")
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
