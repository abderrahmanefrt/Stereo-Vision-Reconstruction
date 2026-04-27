# 📸 3D Reconstruction using Stereo Vision

## 🎯 Objective

This project implements a complete pipeline for **3D reconstruction from stereo images** (left/right).
It estimates **depth (Z)** and generates a **3D point cloud in millimeters**.

---

## 🧠 Principle

The system is based on **stereo vision**:

* Two images of the same scene are captured
* Corresponding points are detected
* The horizontal difference (**disparity**) is used to compute depth

[
Z = \frac{f \cdot B}{d}
]

Where:

* `Z` = depth
* `f` = focal length (pixels)
* `B` = baseline (mm)
* `d` = disparity

---

## ⚙️ Project Pipeline

### 1. 📷 Image Loading

* Check file existence
* Load images using OpenCV
* Convert to landscape orientation

---

### 2. 📐 Camera Calibration

* Load intrinsic matrix `K`
* Convert portrait → landscape
* Parameters:

  * focal lengths (`fx`, `fy`)
  * optical center (`cx`, `cy`)

---

### 3. 🔍 Feature Detection & Matching (SIFT + FLANN)

* Detect keypoints using SIFT
* Match descriptors using FLANN
* Apply Lowe’s ratio test (0.70)

---

### 4. 🧮 Essential Matrix Estimation

```python
cv2.findEssentialMat(...)
```

* Computed using RANSAC
* Encodes the geometric relationship between views

---

### 5. 🧭 Camera Pose Recovery

```python
cv2.recoverPose(...)
```

* Recovers:

  * Rotation `R`
  * Translation `t`

---

### 6. 🔄 Epipolar Rectification

```python
cv2.stereoRectifyUncalibrated(...)
```

* Aligns points horizontally
* Reduces vertical disparity
* Update intrinsic matrices:

```python
K_rect = H @ K
```

---

### 7. 📉 Disparity Computation

```python
disp = pts_l[:,0] - pts_r[:,0]
```

* Horizontal difference between matched points

---

### 8. 📦 3D Triangulation

```python
cv2.triangulatePoints(P1, P2, ...)
```

Using:

* `P1 = K[I | 0]`

* `P2 = K[R | t]`

* Output is in homogeneous coordinates → converted to 3D

---

### 9. 📏 Real-World Scaling

```python
scale = BASELINE / ||t||
pts3d *= scale
```

* Converts reconstruction to **real-world units (mm)**

---

### 10. 🧹 Point Cloud Filtering

* Remove:

  * points with Z < 0 (behind camera)
  * NaN / infinite values
* Apply outlier filtering (3σ rule)

---

### 11. 📊 Visualization

* 2D projections (X-Z, Z-Y)
* 3D scatter plot
* Export:

  * `resultat_3d.png`
  * `nuage_points.ply`

---

## 📁 Required Files

* `image_left_undist4.jpg`
* `image_right_undist4.jpg`
* `camera_K.npy`
* `camera_dist.npy` (optional)

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 📌 Outputs

* 3D point cloud
* Generated files:

  * `points_3d.npy`
  * `points_3d.txt`
  * `nuage_points.ply`
  * `resultat_3d.png`

---

## ⚠️ Notes

* Accuracy depends on:

  * image quality
  * calibration precision
  * number of matches
* Translation vector `t` is normalized → scaling with baseline is required

---

## 🚀 Possible Improvements

* Use full stereo calibration (`stereoCalibrate`)
* Compute dense disparity maps (StereoBM / StereoSGBM)
* Add color/texture to point cloud
* Apply bundle adjustment for optimization

---

## 👨‍💻 Author

This project was developed as part of studies in **computer vision and stereo vision**.
