Added
^^^^^

* Added a simulation-scoped complete-camera-set render seam for temporal backends. Cameras sharing
  one backend can now register their render data, submit the full set once per physics step, and
  pass the actual render interval instead of an implicit 60 Hz value.

Fixed
^^^^^

* Invalidated complete-set output after camera updates and environment resets, and added a renderer
  reset hook so temporal histories cannot survive randomized scene teleports.
