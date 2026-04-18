
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CHARGEMENT DES IMAGES
# ─────────────────────────────────────────────
img_left  = cv2.imread('image_left_undist.jpg')
img_right = cv2.imread('image_right_undist.jpg')

gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# ─────────────────────────────────────────────
# DÉTECTION DES POINTS SIFT
# ─────────────────────────────────────────────
# nfeatures : nombre max de keypoints à détecter
# contrastThreshold : filtre les points de faible contraste
# edgeThreshold : filtre les points sur les bords
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04)

kp1, des1 = sift.detectAndCompute(gray_left,  None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

print(f"[SIFT] Points détectés :")
print(f"  Image gauche  : {len(kp1)} keypoints")
print(f"  Image droite  : {len(kp2)} keypoints")

# ─────────────────────────────────────────────
# MISE EN CORRESPONDANCE — RATIO TEST DE LOWE
# ─────────────────────────────────────────────
# On utilise FLANN (rapide) ou BFMatcher
# Pour chaque point de l'image gauche, on trouve les 2 meilleurs voisins
# dans l'image droite, puis on applique le ratio test :
#   distance_meilleur / distance_2e_meilleur < seuil (0.75)
# → Rejette les appariements ambigus

FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann   = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Application du ratio test de Lowe (seuil recommandé : 0.7 - 0.8)
RATIO_THRESH = 0.75
good_matches = []
for m, n in matches:
    if m.distance < RATIO_THRESH * n.distance:
        good_matches.append(m)

print(f"\n[Matching] Résultats :")
print(f"  Total appariements bruts : {len(matches)}")
print(f"  Après ratio test (< {RATIO_THRESH}) : {len(good_matches)}")

# ─────────────────────────────────────────────
# FILTRAGE PAR LA MATRICE FONDAMENTALE (RANSAC)
# ─────────────────────────────────────────────
# La contrainte épipolaire dit que pour deux images d'une même scène,
# chaque point p1 dans l'image gauche doit correspondre à un point p2
# dans l'image droite qui se trouve sur une ligne épipolaire précise.
# RANSAC élimine les outliers (faux appariements).

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Calcul de la matrice fondamentale F avec RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                  ransacReprojThreshold=3.0,
                                  confidence=0.99)

# Ne garder que les inliers
inlier_mask  = mask.ravel() == 1
pts1_inliers = pts1[inlier_mask]
pts2_inliers = pts2[inlier_mask]
good_final   = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]

print(f"  Après RANSAC (contrainte épipolaire) : {len(good_final)}")
print(f"\n[Matrice Fondamentale F] :")
print(F)

# ─────────────────────────────────────────────
# SAUVEGARDE DES POINTS CORRESPONDANTS
# ─────────────────────────────────────────────
np.save('pts_left.npy',  pts1_inliers)
np.save('pts_right.npy', pts2_inliers)
np.save('F_matrix.npy',  F)
print("\n💾 Points sauvegardés : pts_left.npy / pts_right.npy")

# ─────────────────────────────────────────────
# VISUALISATION DES APPARIEMENTS
# ─────────────────────────────────────────────
img_matches = cv2.drawMatches(
    img_left,  kp1,
    img_right, kp2,
    good_final[:50],   # afficher seulement les 50 premiers
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0)
)

plt.figure(figsize=(18, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'Mise en correspondance SIFT — {len(good_final)} paires valides')
plt.axis('off')
plt.tight_layout()
plt.savefig('sift_matches.png', dpi=150)
plt.show()
print("✅ Visualisation sauvegardée : sift_matches.png")
