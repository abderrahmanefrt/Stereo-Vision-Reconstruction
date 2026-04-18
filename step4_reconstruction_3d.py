"""
ÉTAPE 4 : Reconstruction 3D par Triangulation
==============================================
But : Calculer les coordonnées (X, Y, Z) de chaque point 3D
      à partir des paires de points 2D correspondants.

PRINCIPE MATHÉMATIQUE :
─────────────────────────────────────────────────────────────
Pour un système stéréo avec translation pure sur X (baseline b) :

  Modèle de projection (caméra sténopé) :
    u = fx * X/Z + cx    (équation de projection sur axe x)
    v = fy * Y/Z + cy    (équation de projection sur axe y)

  La DISPARITÉ d est la différence horizontale entre les deux projections :
    d = u_gauche - u_droite

  Calcul de la profondeur Z :
    Z = fx * b / d          ← Plus d est grand, plus l'objet est proche !

  Puis X et Y :
    X = (u_gauche - cx) * Z / fx
    Y = (v_gauche - cy) * Z / fy

  Méthode générale (triangulation DLT) :
    Pour chaque paire (p1, p2), on résout le système Ax = 0
    par décomposition SVD.
─────────────────────────────────────────────────────────────
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
K        = np.load('camera_K.npy')
pts_left  = np.load('pts_left.npy')   # shape (N, 2)
pts_right = np.load('pts_right.npy')  # shape (N, 2)

BASELINE = 100.0  # mm — doit correspondre à votre mesure réelle

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

print(f"[INFO] Paramètres caméra :")
print(f"  fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
print(f"  Baseline b = {BASELINE} mm")
print(f"  Nombre de paires de points : {len(pts_left)}")

# ─────────────────────────────────────────────
# MATRICES DE PROJECTION
# ─────────────────────────────────────────────
# Caméra gauche : repère de référence, R=I, t=[0,0,0]
# Caméra droite : translation de -b sur l'axe X
#   (la caméra s'est déplacée vers la droite de b mm)

R = np.eye(3)
t = np.array([[-BASELINE], [0.0], [0.0]])  # translation en mm

# Matrices de projection P = K * [R | t]
P1 = K @ np.hstack([np.eye(3),   np.zeros((3, 1))])  # caméra gauche
P2 = K @ np.hstack([R,           t])                  # caméra droite

print(f"\n[Matrices de projection]")
print(f"P1 =\n{P1}")
print(f"\nP2 =\n{P2}")

# ─────────────────────────────────────────────
# MÉTHODE 1 : Formule directe (translation pure)
# ─────────────────────────────────────────────
def reconstruct_direct(pts_l, pts_r, fx, fy, cx, cy, b):
    """
    Reconstruction simple quand le mouvement est une translation pure sur X.
    Formules directes (plus intuitives pédagogiquement).
    """
    points_3d = []
    for (ul, vl), (ur, vr) in zip(pts_l, pts_r):
        # Disparité
        d = ul - ur
        if abs(d) < 0.5:   # éviter la division par zéro
            continue

        # Profondeur
        Z = fx * b / d

        # Coordonnées 3D
        X = (ul - cx) * Z / fx
        Y = (vl - cy) * Z / fy

        if Z > 0:   # garder uniquement les points devant la caméra
            points_3d.append([X, Y, Z])

    return np.array(points_3d)

# ─────────────────────────────────────────────
# MÉTHODE 2 : Triangulation OpenCV (DLT + SVD)
# ─────────────────────────────────────────────
def reconstruct_opencv(pts_l, pts_r, P1, P2):
    """
    Triangulation par la méthode DLT (Direct Linear Transform).
    Résout le système Ax=0 par SVD — plus robuste que la formule directe.
    """
    # OpenCV attend des arrays (2, N)
    pts1_T = pts_l.T.astype(np.float64)   # (2, N)
    pts2_T = pts_r.T.astype(np.float64)   # (2, N)

    # triangulatePoints retourne des coordonnées homogènes (4, N)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)

    # Conversion en coordonnées cartésiennes (diviser par W)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T   # (N, 3)

    # Filtrer les points devant la caméra (Z > 0)
    mask = pts_3d[:, 2] > 0
    return pts_3d[mask]

# ─────────────────────────────────────────────
# RECONSTRUCTION
# ─────────────────────────────────────────────
print("\n[RECONSTRUCTION 3D]")

pts3d_direct = reconstruct_direct(pts_left, pts_right, fx, fy, cx, cy, BASELINE)
pts3d_opencv = reconstruct_opencv(pts_left, pts_right, P1, P2)

print(f"  Méthode directe  : {len(pts3d_direct)} points 3D")
print(f"  Méthode OpenCV   : {len(pts3d_opencv)} points 3D")

# ─────────────────────────────────────────────
# FILTRAGE DES OUTLIERS
# ─────────────────────────────────────────────
def filter_outliers(pts, z_thresh=3.0):
    """Supprime les points dont la profondeur est aberrante (> z_thresh sigma)"""
    if len(pts) == 0:
        return pts
    median = np.median(pts[:, 2])
    std    = np.std(pts[:, 2])
    mask   = np.abs(pts[:, 2] - median) < z_thresh * std
    return pts[mask]

pts3d = filter_outliers(pts3d_opencv)
print(f"  Après filtrage outliers : {len(pts3d)} points 3D")

# ─────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────
print(f"\n[STATISTIQUES DES POINTS 3D]")
print(f"  X : [{pts3d[:,0].min():.1f} ; {pts3d[:,0].max():.1f}] mm")
print(f"  Y : [{pts3d[:,1].min():.1f} ; {pts3d[:,1].max():.1f}] mm")
print(f"  Z : [{pts3d[:,2].min():.1f} ; {pts3d[:,2].max():.1f}] mm")
print(f"  Profondeur médiane : {np.median(pts3d[:,2]):.1f} mm")

# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────
np.save('points_3d.npy', pts3d)
np.savetxt('points_3d.txt', pts3d, fmt='%.3f', header='X Y Z (mm)')
print("\n💾 Sauvegardé : points_3d.npy / points_3d.txt")
