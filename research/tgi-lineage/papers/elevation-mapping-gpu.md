# GPU elevation mapping: Elevation Mapping for Locomotion and Navigation using GPU

| | |
|---|---|
| Authors | Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022 |
| Link | https://arxiv.org/abs/2204.12876 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Added at depth 3 because every perceptive controller in this group deploys on a robot-centric 2.5D elevation map, and its errors define the sim-to-real gap that noise models must cover. [Miki et al.](miki-wild.md), [attention-based map encoding](attention-map-encoding.md), [BeamDojo](beamdojo.md) (via robot-centric elevation mapping fed by FAST-LIO) and [Watch Your Step](watch-your-step.md) (its reference [29]) rely on this pipeline or its predecessor by Fankhauser et al. (2014). The CPU predecessor was too slow for fast locomotion with dense point clouds.

## 2. Method

Point clouds are transformed into the map frame on the GPU (CuPy kernels). Each cell's height $h$ and variance $\sigma_m^2$ are updated in Kalman-filter fashion per point,

$$h \leftarrow \frac{\sigma_p^2 h + \sigma_m^2 p_z}{\sigma_m^2 + \sigma_p^2},\qquad \sigma_m^2 \leftarrow \frac{\sigma_m^2\sigma_p^2}{\sigma_m^2 + \sigma_p^2},$$

with sensor noise $\sigma_p^2 = \alpha_d d^2$ growing with range $d$, Mahalanobis outlier rejection and a time-growing variance. Added features: an exclusion region for overhangs, height-drift compensation (shifting the map by the mean error on flat cells), ray-casting visibility cleanup and an upper-bound layer for occluded cells, overlap clearance for multi-floor scenes, a CNN traversability filter, and locomotion post-processing (min-inpainting, smoothing, plane segmentation) ([heightmaps](../math-cookbook.md#m-terrain-lookup)).

## 3. Evidence

With a 10 m × 10 m map at 4 cm resolution on a Jetson Xavier, the full pipeline took about 6.9 ms for 43 000 lidar points, with the traversability CNN the largest share (4.1 ms); map update rates were 19.99 Hz with dome lidars and 16–49 Hz with RealSense cameras depending on filtering. Qualitative comparisons show drift compensation, faster visibility cleanup and overhang exclusion reducing artefacts relative to the CPU baseline; the system ran on four ANYmals in the DARPA SubT challenge.

## 4. Critical analysis

The evaluation is largely qualitative and focused on processing time. The 2.5D representation cannot encode overhangs or walls, and map quality is bounded by pose estimation: drift compensation is a heuristic over flat cells. These residual errors, together with smoothing that blurs edges, are exactly what [BeamDojo](beamdojo.md)'s "foothold extension" and "map rotation" noise and [Miki et al.](miki-wild.md)'s per-foot offsets imitate, and what made [Zhang et al.](risky-terrains.md) bypass perception with motion capture. On sparse terrain, a smoothed or drifted edge is the difference between a supported and an unsupported step.

## 5. Cross-references

- Builds on: Fankhauser et al. robot-centric elevation mapping (2014, cited by [Watch Your Step](watch-your-step.md)).
- Extended by / used in: [Miki et al.](miki-wild.md), [ANYmal parkour](anymal-parkour.md) (as baseline), [attention-based map encoding](attention-map-encoding.md), [BeamDojo](beamdojo.md), [Watch Your Step](watch-your-step.md).
- Contrasts with: learned 3D reconstruction in [ANYmal parkour](anymal-parkour.md); depth-image policies in [Extreme Parkour](extreme-parkour.md).

## 6. Relevance to TGI

TGI's deployed policy, inherited from Watch Your Step, samples an elevation map of this kind around predicted footholds, so its observation noise models target this pipeline's error modes. In training, TGI's world model and rewards read the simulator's exact heightmap through differentiable lookups instead; the elevation map therefore enters TGI only at the policy input, not in the gradient signal.
