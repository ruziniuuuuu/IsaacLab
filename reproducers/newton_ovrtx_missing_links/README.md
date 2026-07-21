# Isaac Lab Newton + OVRTX missing-link reproducer

This is a standalone minimal reproducer for robot links intermittently disappearing
when Newton/MJWarp physics, OVRTX camera rendering, and the Newton visualizer are
used together. It contains the robot USD and does **not** import or register
GalbotLab, its tasks, planners, randomizers, or environment wrapper.

## Reproduction

Run from an environment where Isaac Lab, Newton, OVRTX, and the Newton visualizer
are installed:

```bash
python repro.py \
  --num-envs 4 \
  --steps 5000 \
  --seed 17
```

The default matches the reported backend composition:

- Newton/MJWarp physics with five substeps;
- four cloned environments with synchronized robot-root and joint motion;
- one fixed third-person RGB camera per environment, framing the whole robot and
  rendered by OVRTX;
- an outstretched initial arm pose so every arm link is exposed instead of
  self-occluded;
- one static emissive-gray backdrop per environment so link silhouettes are not
  contaminated by lighting or shadows;
- `OVRTXRendererCfg(use_ovrtx_cloning=False)`;
- Newton visualizer with the four OVRTX images in its tiled-camera panel.

Close the Newton window to stop early. The script derives a denoised robot
silhouette from every RGB image and compares environments whose robot and camera
poses are synchronized. A large difference at the same pixel positions for three
consecutive frames prints `[REPRODUCED]`, saves evidence under `captures/`, and
returns exit code 2. This differential detector is intentionally small, but it
cannot prove that environment zero is complete and cannot detect a link that
disappears from every environment at once. Inspect the saved RGB frames; the
corrected evidence uses a separate CPU-transform reference for that reason.

The script always saves the final RGB frame, camera poses, Newton body/joint state,
USD visibility state, and `run_info.json` containing package/GPU versions, the
robot USD hash, the environment-variable value, and whether the installed Isaac
Lab OVRTX adapter actually reads that variable. A rendering failure is only
reported when Newton state is finite and synchronized, so physics divergence is
not counted as this bug. Viewer-enabled captures also include the Newton geometry
framebuffer.

For an automated offscreen run:

```bash
python repro.py --headless --steps 1000 --stop-on-failure
```

Useful one-variable controls are:

```bash
# Keep OVRTX cameras but remove the Newton viewer.
python repro.py --viewer none --headless

# Keep the Newton viewer but switch on OVRTX's env_0 cloning path.
python repro.py --ovrtx-cloning

# Add periodic full-scene resets as a lifecycle stress test.
python repro.py --reset-every 500

# Force CPU-side transform reads when the installed adapter supports this variable.
ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=0 python repro.py
```

Isaac Lab `v3.0.0-beta2.patch1` does not read that variable: for OVRTX 0.3 it
passes `read_gpu_transforms=True` directly. In that release, setting the variable
alone is a no-op. The reproducer prints a warning and records
`honors_read_gpu_transforms_env: false` when it detects this situation. Apply the
included narrow local patch first:

```bash
patch -p1 -d /path/to/IsaacLab < \
  patches/isaaclab_3_0_beta2_read_gpu_transforms_env.patch
ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=0 python repro.py --headless
```

Current Isaac Lab `develop` already contains an environment-variable helper; do
not apply this backport there.

## Expected and actual behavior

Expected: every robot link remains present in every OVRTX image; synchronized
environments produce the same robot geometry within normal path-tracing noise.

Reported actual behavior: intermittently, one or more robot links disappear from
one environment. The corresponding RGB image diverges from environment zero, and
the saved `stage_visibility.json` can show that USD visibility remained unchanged
at the time of the rendering failure.

Because the symptom is probabilistic, repeat the program in fresh processes when
estimating its frequency:

```bash
for trial in $(seq 1 50); do
  echo "trial=${trial}"
  python repro.py --headless --stop-on-failure || break
done
```

## Environment used when this reproducer was prepared

- `isaaclab==6.1.14`
- `isaaclab-newton==0.13.6`
- `isaaclab-ov==0.4.2`
- `isaaclab-visualizers==0.1.0`
- `newton==1.2.1`
- `ovrtx==0.3.0.312915`
- `torch==2.10.0+cu128`
- `warp-lang==1.13.0`
- NVIDIA GeForce RTX 5090 D v2, driver 590.48.01

The script records the receiving machine's actual versions on every run.

## Confirmed sample

`evidence/` contains a corrected step-7 comparison between an effective
`read_gpu_transforms=False` run and an unmodified production-adapter run where the
same environment variable was ignored. The CPU frame is complete; the GPU/Fabric
frame visibly loses a left-arm link. See
[`evidence/README.md`](evidence/README.md) for images, metrics, and trial counts.

## Files

- `repro.py`: the complete reproducer and differential detector.
- `assets/galbot_one_golf/`: self-contained robot USD, meshes, materials, and
  textures referenced with relative paths.
- `assets/backdrop.usda`: a tiny emissive backdrop used by the silhouette detector.
- `evidence/`: a confirmed local failure and its minimal diagnostic evidence.
- `patches/`: a narrow local backport that makes the CPU-transform environment
  variable effective on Isaac Lab `v3.0.0-beta2.patch1`.
- `assets/LICENSE`: Apache-2.0 license shipped with the robot description source.

The reproducer's Python code may be submitted under the Isaac Lab issue's
preferred license. The bundled robot asset remains governed by `assets/LICENSE`.
