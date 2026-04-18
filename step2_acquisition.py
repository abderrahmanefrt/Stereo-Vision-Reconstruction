"""
ÉTAPE 2 : Acquisition des deux images stéréo
============================================
But : Capturer image gauche (I1) et image droite (I2)
      avec un déplacement horizontal connu = baseline b

IMPORTANT : Le mouvement doit être une TRANSLATION PURE
            (pas de rotation, pas de changement de hauteur)
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# PARAMÈTRE CLEF : LA BASELINE
# ─────────────────────────────────────────────
# b = distance de translation horizontale entre les deux prises de vue
# Mesurez-la physiquement avec une règle !
# Exemple : b = 10 cm = 100 mm
BASELINE = 100  # mm — à adapter à votre mesure réelle

def capture_stereo_pair():
    """
    Capture les deux images stéréo depuis la webcam.
    Workflow :
      1. Placer la caméra à gauche → appuyer ESPACE
      2. Déplacer la caméra de BASELINE mm vers la droite
      3. Appuyer ESPACE pour la 2e image
    """
    cap = cv2.VideoCapture(0)
    images = []
    names = ["GAUCHE (position initiale)", "DROITE (après translation)"]

    for i, name in enumerate(names):
        print(f"\n📸 Prêt pour l'image {name}")
        if i == 1:
            print(f"   ➡️  Déplacez la caméra de {BASELINE} mm vers la droite")
        print("   Appuyez sur ESPACE pour capturer...")

        while True:
            ret, frame = cap.read()
            label = f"Image {i+1}/2 : {name} | ESPACE=capturer"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Acquisition stéréo", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                images.append(frame.copy())
                print(f"   ✅ Image {i+1} capturée !")
                break

    cap.release()
    cv2.destroyAllWindows()
    return images[0], images[1]


def load_or_capture():
    """
    Charge des images existantes ou lance la capture
    """
    import os
    if os.path.exists('image_left.jpg') and os.path.exists('image_right.jpg'):
        print("[INFO] Chargement des images existantes...")
        img_left  = cv2.imread('image_left.jpg')
        img_right = cv2.imread('image_right.jpg')
    else:
        print("[INFO] Capture des images depuis la webcam...")
        img_left, img_right = capture_stereo_pair()

    return img_left, img_right


def undistort_images(img_left, img_right):
    """
    Corrige la distorsion lens avec les paramètres de calibration
    """
    K    = np.load('camera_K.npy')
    dist = np.load('camera_dist.npy')

    h, w = img_left.shape[:2]
    # Calcul de la nouvelle matrice de caméra optimale
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    img_left_und  = cv2.undistort(img_left,  K, dist, None, newK)
    img_right_und = cv2.undistort(img_right, K, dist, None, newK)

    print(f"[INFO] Distorsion corrigée. Nouvelle matrice K :")
    print(newK)

    return img_left_und, img_right_und, newK


# ─────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────
if __name__ == '__main__':
    img_left, img_right = load_or_capture()

    # Correction de distorsion
    img_left_u, img_right_u, K_new = undistort_images(img_left, img_right)

    # Sauvegarde
    cv2.imwrite('image_left_undist.jpg',  img_left_u)
    cv2.imwrite('image_right_undist.jpg', img_right_u)

    # Vérification visuelle
    comparison = np.hstack([img_left_u, img_right_u])
    cv2.imshow('Paire stéréo (gauche | droite)', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n✅ Images sauvegardées : image_left_undist.jpg / image_right_undist.jpg")
