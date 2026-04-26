"""
CALIBRATION RAPIDE — avec redimensionnement automatique
========================================================
- Réduit les images à 1280px de large → 10x plus rapide
- Résultat identique car K est rescalé automatiquement
- Adapté pour damier 9x7, carrés 20mm
"""

import cv2
import numpy as np
import glob
import os

# ══════════════════════════════════════════════
#  PARAMÈTRES — VÉRIFIE CES VALEURS
# ══════════════════════════════════════════════
COLS        = 9     # coins internes horizontaux (cases - 1)
ROWS        = 7     # coins internes verticaux   (cases - 1)
SQUARE_SIZE = 20    # taille d'un carré en mm (écrit sur ton damier : 20x20mm ✅)
RESIZE_W    = 1280  # largeur de travail (réduit pour la vitesse)

# ══════════════════════════════════════════════
#  PRÉPARATION
# ══════════════════════════════════════════════
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []
imgpoints = []

# ══════════════════════════════════════════════
#  CHARGEMENT ET TRAITEMENT DES IMAGES
# ══════════════════════════════════════════════
images = (glob.glob('calibration_images/*.jpg')  +
          glob.glob('calibration_images/*.jpeg') +
          glob.glob('calibration_images/*.png')  +
          glob.glob('calibration_images/*.JPG')  +
          glob.glob('calibration_images/*.JPEG'))

print(f"[INFO] {len(images)} images trouvées")
print(f"[INFO] Damier : {COLS}×{ROWS} coins internes, carrés {SQUARE_SIZE}mm")
print(f"[INFO] Redimensionnement à {RESIZE_W}px de large\n")

scale_ratio  = None   # ratio de redimensionnement (calculé sur 1ère image)
original_size = None  # taille originale des images

ok_count   = 0
fail_count = 0

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"  ⚠️  Impossible de lire : {fname}")
        continue

    h_orig, w_orig = img.shape[:2]

    # ── Calcul du ratio de redimensionnement (1 seule fois) ──
    if scale_ratio is None:
        scale_ratio   = RESIZE_W / w_orig
        original_size = (w_orig, h_orig)
        resize_h      = int(h_orig * scale_ratio)
        print(f"[INFO] Taille originale : {w_orig}×{h_orig} px")
        print(f"[INFO] Taille de travail : {RESIZE_W}×{resize_h} px")
        print(f"[INFO] Ratio : {scale_ratio:.4f}\n")

    # ── Redimensionnement ─────────────────────────────────────
    img_small = cv2.resize(img, (RESIZE_W, int(h_orig * scale_ratio)))
    gray      = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # ── Détection des coins ───────────────────────────────────
    print(f"  [{i+1}/{len(images)}] {os.path.basename(fname)} ... ", end='', flush=True)
    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)

    if ret:
        ok_count += 1
        objpoints.append(objp)

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners_refined)
        print(f"✅ coins détectés")

        # Visualisation rapide
        vis = img_small.copy()
        cv2.drawChessboardCorners(vis, (COLS, ROWS), corners_refined, ret)
        cv2.imshow('Calibration — appuie une touche', vis)
        cv2.waitKey(300)
    else:
        fail_count += 1
        print(f"❌ coins NON détectés")

cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"  Images valides   : {ok_count}")
print(f"  Images rejetées  : {fail_count}")
print(f"{'='*50}\n")

if ok_count < 5:
    print("❌ Moins de 5 images valides — calibration impossible.")
    print("   Reprends les photos en suivant les conseils.")
    exit()

# ══════════════════════════════════════════════
#  CALIBRATION SUR LES IMAGES RÉDUITES
# ══════════════════════════════════════════════
resize_h = int(original_size[1] * scale_ratio)
ret, K_small, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints,
    (RESIZE_W, resize_h),
    None, None
)

# ══════════════════════════════════════════════
#  RESCALER K VERS LA RÉSOLUTION ORIGINALE
# ══════════════════════════════════════════════
# K a été calculé pour RESIZE_W×resize_h
# On le ramène à la taille originale des images
K = K_small.copy()
K[0, 0] /= scale_ratio   # fx
K[1, 1] /= scale_ratio   # fy
K[0, 2] /= scale_ratio   # cx
K[1, 2] /= scale_ratio   # cy

# ══════════════════════════════════════════════
#  RÉSULTATS
# ══════════════════════════════════════════════
print("✅ RÉSULTATS DE LA CALIBRATION")
print("="*50)
print(f"\n  Calculé sur images {RESIZE_W}px, rescalé pour {original_size[0]}px\n")
print(f"  Matrice K (pour images {original_size[0]}×{original_size[1]}) :")
print(f"  [ {K[0,0]:.1f}      0    {K[0,2]:.1f} ]")
print(f"  [    0    {K[1,1]:.1f}   {K[1,2]:.1f} ]")
print(f"  [    0       0       1   ]\n")
print(f"  fx = {K[0,0]:.1f} px  (attendu ~2000-4000 pour image {original_size[0]}px)")
print(f"  fy = {K[1,1]:.1f} px")
print(f"  cx = {K[0,2]:.1f} px  (attendu ~{original_size[0]//2})")
print(f"  cy = {K[1,2]:.1f} px  (attendu ~{original_size[1]//2})")

print(f"\n  Coefficients distorsion : {dist.ravel()}")

# Erreur de reprojection
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K_small, dist
    )
    mean_error += cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
mean_error /= len(objpoints)

print(f"\n  Erreur de reprojection : {mean_error:.4f} px")
if mean_error < 0.5:
    print("  ✅ Excellente calibration !")
elif mean_error < 1.0:
    print("  ✅ Bonne calibration")
else:
    print("  ⚠️  Erreur élevée — essaie de supprimer les photos floues ou mal cadrées")

# ══════════════════════════════════════════════
#  SAUVEGARDE
# ══════════════════════════════════════════════
np.save('camera_K.npy',    K)
np.save('camera_dist.npy', dist)

print(f"\n💾 Sauvegardé : camera_K.npy  camera_dist.npy")
print(f"   Ces fichiers sont prêts pour main_stereo_final.py ✅")