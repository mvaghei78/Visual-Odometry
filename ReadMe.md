Visual odometry is a process used to estimate the motion of a camera through consecutive frames. Below is a general roadmap for visual odometry, including the purpose of each step, and the input and output of each step.
## Overview
This project is a simple implementation of visual odometry using various computer vision techniques.

## Directory Structure
- `config/`: Configuration files.
- `dataset/`: All datasets you want to use for this project should place in this folder.
- `lib/`: Libraries used in visual odometry.
- `result/`: Result files of this project save here (like .ply file for 3D representation)
- `utils/`: Utility scripts for visualization, generate outputs and dataset processing.

### General Steps of Visual Odometry

1. **Feature Detection and Matching**
2. **Estimate the Essential Matrix**
3. **Decompose the Essential Matrix to Get Rotation (R) and Translation (t)**
4. **Triangulate Points**
5. **Pose Estimation and Optimization**
6. **Scale Estimation (for monocular systems)**
7. **Update Pose and Map**

### Detailed Roadmap

#### Step 1: Feature Detection and Matching

**Purpose:** Identify and match key points between consecutive frames to establish correspondences.

**Input:** Two consecutive image frames.

**Output:** Matched feature points between the two frames.

**Process:**
- Detect features (e.g., using SIFT, ORB, or FAST).
- Match features (e.g., using FLANN or brute-force matcher).
- Filter matches (e.g., using Lowe’s ratio test or RANSAC to remove outliers).

#### Step 2: Estimate the Essential Matrix

**Purpose:** Compute the essential matrix that encapsulates the relative rotation and translation between two frames.

**Input:** Matched feature points, camera intrinsic matrix (K).

**Output:** Essential matrix (E).

**Process:**
- Use the normalized coordinates of the matched points.
- Compute the essential matrix using the eight-point algorithm or RANSAC.

#### Step 3: Decompose the Essential Matrix to Get Rotation (R) and Translation (t)

**Purpose:** Extract the possible rotation and translation from the essential matrix.

**Input:** Essential matrix (E).

**Output:** Four possible solutions for (R) and (t).

**Process:**
- Decompose the essential matrix using SVD.
- Compute the possible rotation and translation matrices.

#### Step 4: Triangulate Points

**Purpose:** Estimate the 3D coordinates of the matched feature points.

**Input:** Matched feature points, intrinsic matrix K, rotation R, translation t.

**Output:** 3D points in space.

**Process:**
- Form the projection matrices for the two frames.
- Use triangulation to get the 3D coordinates.

#### Step 5: Pose Estimation and Optimization

**Purpose:** Refine the estimated pose to reduce error.

**Input:** 3D points, initial (R) and (t).

**Output:** Optimized rotation (R) and translation (t).

**Process:**
- Use techniques such as bundle adjustment to minimize reprojection error.
- Apply optimization algorithms like Levenberg-Marquardt.

#### Step 6: Scale Estimation (for Monocular Systems)

**Purpose:** Estimate the scale of translation for monocular systems (optional for stereo or depth camera systems).

**Input:** 3D points, initial scale.

**Output:** Estimated scale.

**Process:**
- Use known measurements or additional sensors (e.g., IMU) to estimate scale.

#### Step 7: Update Pose and Map

**Purpose:** Update the camera pose and the map with the new 3D points.

**Input:** Optimized (R) and (t), previous camera pose.

**Output:** Updated camera pose, updated map of 3D points.

**Process:**
- Update the pose of the camera.
- Add new 3D points to the map.
- Integrate into SLAM (Simultaneous Localization and Mapping) if needed.

### Summary of the Visual Odometry Pipeline

1. **Feature Detection and Matching:** Detect and match features between consecutive frames.
2. **Estimate the Essential Matrix:** Compute the essential matrix from matched points.
3. **Decompose the Essential Matrix:** Extract possible rotations and translations.
4. **Triangulate Points:** Estimate 3D coordinates of matched points.
5. **Pose Estimation and Optimization:** Refine the camera pose using optimization techniques.
6. **Scale Estimation:** Estimate scale for monocular systems.
7. **Update Pose and Map:** Update the camera pose and the 3D map.

Each step builds upon the previous one, starting from detecting and matching features to finally updating the camera pose and map. This pipeline is essential for applications like robot navigation, autonomous driving, and augmented reality.