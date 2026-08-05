Fixed
^^^^^

* Gave every logical OVRTX camera its own RenderProduct and submitted all products in one sensor
  step, preventing head and wrist cameras from overwriting or discarding each other's DLSS history.
  The adapter now routes outputs by RenderProduct path, uses the simulation render interval, clears
  histories after environment reset, and performs the recommended 40-frame RTPT warm-up.
