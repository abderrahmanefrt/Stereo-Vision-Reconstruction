import cv2
import numpy as np
import os

# ─────────────────────────────────────────────
# PARAMÈTRE CLEF : LA BASELINE
# ─────────────────────────────────────────────
BASELINE = 150  # mm

def load_or_capture():
    """
    Charge les images locales directement
    """
    LEFT_PATH  = 'image_left_undist4.jpg'
    RIGHT_PATH = 'image_right_undist4.jpg'

    # Vérification que les fichiers existent
    if not os.path.exists(LEFT_PATH):
        print(f"❌ Fichier introuvable : {LEFT_PATH}")
        exit()
    if not os.path.exists(RIGHT_PATH):
        print(f"❌ Fichier introuvable : {RIGHT_PATH}")
        exit()

    img_left  = cv2.imread(LEFT_PATH)
    img_right = cv2.imread(RIGHT_PATH)

    # Vérification que les images sont bien chargées
    if img_left is None:
        print(f"❌ Impossible de lire : {LEFT_PATH}")
        exit()
    if img_right is None:
        print(f"❌ Impossible de lire : {RIGHT_PATH}")
        exit()

    print(f"✅ Image gauche chargée  : {LEFT_PATH}  — taille {img_left.shape}")
    print(f"✅ Image droite chargée  : {RIGHT_PATH} — taille {img_right.shape}")

    return img_left, img_right


def undistort_images(img_left, img_right):
    """
    Corrige la distorsion avec les paramètres de calibration
    """
    K    = np.load('camera_K.npy')
    dist = np.load('camera_dist.npy')

    h, w = img_left.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    img_left_und  = cv2.undistort(img_left,  K, dist, None, newK)
    img_right_und = cv2.undistort(img_right, K, dist, None, newK)

    print(f"\n[INFO] Distorsion corrigée.")
    print(f"Nouvelle matrice K :\n{newK}")

    return img_left_und, img_right_und, newK


# ─────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Charger les images locales
    img_left, img_right = load_or_capture()

    # 2. Correction de distorsion
    img_left_u, img_right_u, K_new = undistort_images(img_left, img_right)

    # 3. Sauvegarde
    cv2.imwrite('image_left_undist4.jpg',  img_left_u)
    cv2.imwrite('image_right_undist4.jpg', img_right_u)
    print("\n💾 Images sauvegardées : image_left_undist4.jpg / image_right_undist4.jpg")

    # 4. Vérification visuelle côte à côte
    comparison = np.hstack([img_left_u, img_right_u])
    cv2.imshow('Paire stéréo (gauche | droite)', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()