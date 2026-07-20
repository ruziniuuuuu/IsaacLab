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
- one static emissive-gray backdrop per environment so link silhouettes are not
  contaminated by lighting or shadows;
- `OVRTXRendererCfg(use_ovrtx_cloning=False)`;
- Newton visualizer with the four OVRTX images in its tiled-camera panel.

Close the Newton window to stop early. The script derives a denoised robot
silhouette from every RGB image and compares it with environment zero because all
four robot poses and camera poses are identical. A large silhouette difference at
the same pixel positions for three consecutive frames prints `[REPRODUCED]`, saves
evidence under `captures/`, and returns exit code 2. The uniform backdrop, spatial
closing, and persistence filter reject ordinary path-tracing noise. It always
saves the final RGB frame, camera world/relative poses, Newton body/joint state,
USD visibility state, and `run_info.json` containing package/GPU versions and the
robot USD hash. A rendering failure is only reported when all Newton body and joint
states are finite and remain
synchronized across environments; physics divergence is therefore not counted as
the missing-link bug. When the Newton viewer is enabled, each capture also includes
`newton_viewer.png`, the viewer's own geometry framebuffer. The four OVRTX camera
images and derived masks are saved as `rgb_env_*.png` and
`silhouette_env_*.png`; failure captures also include `detection.json`.

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

# Force CPU-side transform reads as an OVRTX adapter diagnostic.
ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS=0 python repro.py
```

## Expected and actual behavior

Expected: every robot link remains present in every OVRTX image; synchronized
environments produce the same robot silhouette within the configured threshold.

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

`evidence/` contains a locally reproduced failure at step 7, including the
reference and affected OVRTX images, binary silhouettes, Newton viewer framebuffer,
USD visibility result, synchronized simulation state, and exact run metadata. See
[`evidence/README.md`](evidence/README.md) for the comparison.

## Files

- `repro.py`: the complete reproducer and differential detector.
- `assets/galbot_one_golf/`: self-contained robot USD, meshes, materials, and
  textures referenced with relative paths.
- `assets/backdrop.usda`: a tiny emissive backdrop used by the silhouette detector.
- `evidence/`: a confirmed local failure and its minimal diagnostic evidence.
- `assets/LICENSE`: Apache-2.0 license shipped with the robot description source.

The reproducer's Python code may be submitted under the Isaac Lab issue's
preferred license. The bundled robot asset remains governed by `assets/LICENSE`.
