Changelog
---------

Unreleased
~~~~~~~~~~

Changed
^^^^^^^

* Robot connector visuals now load from native ``.usd`` instead of ``.glb``.
  The cable's SFP / SC / LC connector ``visual`` prims in
  ``assets/robots/ur5e_cable/aic_unified_robot_cable_sdf.usd`` previously
  referenced ``.glb`` meshes via the on-the-fly glTF→USD syntax
  (``...glb:SDF_FORMAT_ARGS:target=usd``), which no longer resolves in the
  current container (no glTF file-format plugin), so the connectors rendered
  as nothing — the cable looked bare in every camera. The ``.glb`` files were
  converted to sibling ``.usd`` (in meters) and the references repointed. No
  Python / MDP / observation / physics change. See ``docs/07_connector_visual_assets.md``.

Added
^^^^^

* Eval tooling (``scripts/``, outside the ``aic_task`` package): a
  third-person **overview camera** in ``scripts/eval_demos.py`` (``--save_videos``
  now writes a whole-robot ``_overview`` mp4 plus a ``_cams`` policy-camera
  strip), and a generic ``extra_sensors`` hook on
  ``PortInsertionEnv.make`` (``scripts/il/env_wrapper.py``) for attaching
  eval-only scene sensors without touching the task config. These do not enter
  any observation group, so the policy and dataset schema are unchanged.

0.1.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab