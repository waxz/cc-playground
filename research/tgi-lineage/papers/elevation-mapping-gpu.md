# GPU elevation mapping: Elevation Mapping for Locomotion and Navigation using GPU

| | |
|---|---|
| Authors | Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022 |
| Link | https://arxiv.org/abs/2204.12876 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Added at depth 3 because every perceptive controller in this group deploys on a robot-centric 2.5D elevation map, and its errors define the sim-to-real gap that observation-noise models must cover. [Miki et al.](miki-wild.md), [attention-based map encoding](attention-map-encoding.md), [BeamDojo](beamdojo.md) (via robot-centric elevation mapping fed by FAST-LIO) and [Watch Your Step](watch-your-step.md) (its reference [29]) rely on this pipeline or its CPU predecessor by Fankhauser et al. (2014, 2018); the model-based planners [TAMOLS](tamols.md) and [perceptive NMPC](perceptive-nmpc.md) consume its post-processed layers. The paper positions the elevation map between a 2D occupancy grid, which cannot represent slopes and steps, and a 3D voxel grid, which needs far more memory at the fine resolution that foothold selection requires. The CPU predecessor updated each cell's height in Kalman-filter fashion but could not process dense point clouds fast enough for agile locomotion; a GPU mapper by Pan et al. targeted long-range navigation with down-sampled clouds. The contribution is engineering: moving the whole pipeline to the GPU and adding the features that field deployment, notably the DARPA Subterranean Challenge, showed to be necessary.

## 2. Method

**Pipeline.** Point clouds and the robot pose (from leg odometry, or from an extended Kalman filter fusing IMU and LiDAR SLAM) are transferred to GPU memory, transformed into the map frame, and processed by custom CUDA kernels written with CuPy, which shares GPU memory with PyTorch; ROS handles communication. Per-point kernels compute the height-drift error, update heights and perform ray casting; per-cell kernels then run overlap clearance, traversability estimation, normal computation and optional filters. The map is copied to CPU memory and published as a GridMap message only at a user-defined rate, and heavier post-processing runs on request.

**Height update.** Each cell's height $h$ and variance $\sigma_m^2$ are updated per point in Kalman-filter form ([heightmaps](../math-cookbook.md#m-terrain-lookup)),

$$h \leftarrow \frac{\sigma_p^2 h + \sigma_m^2 p_z}{\sigma_m^2 + \sigma_p^2},\qquad \sigma_m^2 \leftarrow \frac{\sigma_m^2\sigma_p^2}{\sigma_m^2 + \sigma_p^2},$$

with sensor noise $\sigma_p^2 = \alpha_d d^2$ growing quadratically with range $d$, a constant variance growth $\sigma_t^2$ per unit time for cells that are not updated, and Mahalanobis-distance outlier rejection. Two additions concern geometry that a 2.5D map represents poorly. An exclusion region, a ramp defined by an angle and offsets around the sensor, discards points above it so that ceilings and overhangs near the robot do not appear as walls, while sloped ground ahead is still mapped. At vertical edges, where points of different heights fall into one cell and averaging would lower the estimate, points below the current estimate are ignored once a cell has received more than a threshold number of points, which sharpens edges.

**Drift compensation and visibility cleanup.** State-estimation drift, mainly vertical, creates step-like artefacts. For each point the error to the current map is computed on cells of high traversability (to avoid rough or steep areas), and the mean error is added to the whole elevation layer, shifting the map to match the newest scan. Visibility cleanup casts a ray from the sensor to every point; a cell traversed by the ray below its estimated height minus its standard deviation is cleared, which removes moved obstacles. To avoid jitter at grazing rays, a cell is cleared only if it has not been updated recently and the ray satisfies $\lvert r\cdot n\rvert > \alpha_n$ with the surface normal $n$. The same rays define an upper-bound layer for unobserved cells, since the ground cannot lie above a ray that passed through; this distinguishes occluded but probably safe regions (small inclinations of the bound) from real drops. Overlap clearance removes stale heights from another floor near the robot.

**Post-processing.** A small CNN computes a traversability value for navigation at the full update rate. For locomotion, the package provides a minimum-based inpainting filter that fills occluded cells with the minimum along the occlusion border, Gaussian, box-blur and median smoothing (median being reported as better at rejecting artefacts), and segmentation of the map into planar regions with polygonal boundaries for optimisation-based controllers.

## 3. Evidence

On a 10 m × 10 m map at 4 cm resolution, the full pipeline on a Jetson Xavier took 6.857 ms for 43 017 Bpearl LiDAR points: 1.194 ms for point transformation and drift-error counting, 0.742 ms for drift compensation, 0.648 ms for height update and ray casting, 0.003 ms for overlap clearance, 4.102 ms for the traversability CNN and 0.168 ms for normals. Map update rates on the Jetson were 49.4 Hz for voxel-filtered RealSense clouds (6276 points, 60 Hz sensor), 16.1 Hz for raw RealSense clouds (407 040 points) and 19.99 Hz for the 20 Hz Bpearl. Against the CPU baseline on a desktop and on the Jetson, processing time grew much more slowly with the number of points, and the baseline lagged on the onboard computer. Qualitative comparisons on ANYmal data show smaller gaps under strong height drift with compensation, cleaner maps from per-scan visibility cleanup (the baseline cleans at a lower rate), and an overhang excluded where the baseline drew a wall. The system ran on four ANYmals of team CERBERUS in the SubT final; it fed the RL controller of [Miki et al.](miki-wild.md), where drift compensation reduced step-like artefacts during slipping, and two model-based planners (foothold scores from normals and roughness; a virtual floor from box-blur, dilation and median filters in [TAMOLS](tamols.md)) as well as a plane-segmentation-based MPC.

## 4. Critical analysis

The evaluation is largely qualitative for map quality and quantitative only for processing time; there is no ground-truth comparison of height error, so the accuracy of the map, the quantity that matters for footholds, is not measured. The 2.5D representation cannot encode overhangs or walls, and the exclusion region is a heuristic that trades representing low ceilings against representing obstacles near the robot. Drift compensation shifts the whole map by one scalar, which corrects vertical drift but not tilt or horizontal drift, and it relies on flat, traversable cells being present. Smoothing and inpainting deliberately blur edges and invent heights in occluded cells.

These residual errors are exactly what [BeamDojo](beamdojo.md)'s foothold-extension and map-rotation noise and [Miki et al.](miki-wild.md)'s per-foot offsets and belief encoder imitate, and what led [Zhang et al.](risky-terrains.md) to bypass perception with motion capture. On sparse terrain a smoothed or drifted edge is the difference between a supported and an unsupported step, and a policy trained on exact simulated maps must learn robustness to errors of the kind documented here, typically centimetres of offset and blurred rims.

## 5. Cross-references

- Builds on: Fankhauser et al. robot-centric probabilistic elevation mapping (2014, 2018), cited by [Watch Your Step](watch-your-step.md).
- Extended by / used in: [Miki et al.](miki-wild.md), [ANYmal parkour](anymal-parkour.md) (as baseline), [attention-based map encoding](attention-map-encoding.md), [BeamDojo](beamdojo.md), [Watch Your Step](watch-your-step.md), [TAMOLS](tamols.md), [perceptive NMPC](perceptive-nmpc.md).
- Contrasts with: learned 3D reconstruction in [ANYmal parkour](anymal-parkour.md); depth-image policies in [Extreme Parkour](extreme-parkour.md); signed distance fields as full 3D terrain representations ([SDFs](../math-cookbook.md#m-terrain-lookup)).

## 6. Relevance to TGI

TGI's deployed policy, inherited from Watch Your Step, samples an elevation map of this kind in 7×7 patches around predicted footholds, so its observation noise (per-episode planar and vertical drift and per-point noise added in simulation and in imagination) targets this pipeline's error modes, in particular the scalar vertical offset that drift compensation leaves and the edge blurring of smoothing filters. In training, TGI's world model, rewards, terminations and critic read the simulator's exact 4 cm heightmap through differentiable lookups instead, and the policy's own scan is deliberately non-differentiable. The elevation map therefore enters TGI only at the policy input, not in the gradient signal, and the robustness of the deployed policy to real mapping errors rests entirely on how well the injected noise matches the artefacts described here.
