---
scope: the cable connector visual meshes referenced by the UR5e robot USD — why they are now .usd instead of .glb, and how to regenerate them
audience: AI agents working in this repo
last_verified_commit: 2f86d63
related:
  - 01_package_structure.md
  - 03_port_insertion_overview.md
  - ../aic_task/assets/robots/ur5e_cable/aic_unified_robot_cable_sdf.usd
---

# 07 · Connector Visual Assets

The UR5e cable robot carries three connector parts past the gripper — the
held **SFP module** (the thing actually inserted) plus passive **SC** and
**LC** plugs on the same cable. Their *visual* meshes live as separate files
under
[`assets/robots/ur5e_cable/visuals/`](../aic_task/assets/robots/ur5e_cable/visuals/)
and are pulled into the robot USD by reference. This doc records what those
files are, the format change made on `2f86d63`, and how to regenerate them —
because the change is invisible in the Python layer and easy to undo by
accident.

> Scope note: this is an **asset/data** change. It touches no Python, no MDP
> term, no observation schema, and no physics — only which file the renderer
> reads for each connector's surface mesh. Collision geometry for the
> connector links is authored separately in the robot USD and is unaffected.

## The reference, prim by prim

Inside
[`aic_unified_robot_cable_sdf.usd`](../aic_task/assets/robots/ur5e_cable/aic_unified_robot_cable_sdf.usd)
three `visual` prims each carry a single reference to a connector mesh file:

| Visual prim (in the robot USD) | Mesh file |
| --- | --- |
| `/World/cable/sfp_module/sfp_module_link/visual` | `./visuals/sfp_module_visual.usd` |
| `/World/cable/sc_plug/sc_plug_link/visual` | `./visuals/sc_plug_visual.usd` |
| `/World/cable/lc_plug/lc_plug_link/visual` | `./visuals/lc_plug_visual.usd` |

References are **relative** (`./visuals/...`), so the `visuals/` directory must
travel with the robot USD.

## Why .usd and not .glb (the change on 2f86d63)

The original robot USD referenced the **`.glb`** files directly, using USD's
on-the-fly glTF→USD dynamic-payload syntax:

```
./visuals/sfp_module_visual.glb:SDF_FORMAT_ARGS:target=usd
```

That syntax only resolves if a glTF `SdfFileFormat` plugin is registered in
the running USD runtime. In the current `isaac-lab-base` container it is
**not** — `Sdf.FileFormat.FindByExtension("glb")` returns `None`, and it stays
`None` even after `enable_extension("omni.kit.asset_converter")` and force-
loading that extension's `OmniAsset` plugin. So every scene load logged:

```
[omni.usd] Could not open asset @.../sfp_module_visual.glb@ ...
  -- Cannot determine file format for @...glb:SDF_FORMAT_ARGS:target=usd@
```

and the connectors **rendered as nothing** — the cable appeared as a bare
yellow strand in every camera, including the three policy cameras and any
eval video. (This likely worked at dataset-collection time via a converted-
asset cache that a later container rebuild cleared; the `.glb` files
themselves are intact, the runtime just can no longer parse them.)

The fix converts each `.glb` to a native `.usd` offline (once) and repoints
the reference at the `.usd`. USD loads `.usd` with no plugin, so the
connectors render everywhere again. Both the `.glb` (source of truth) and the
`.usd` (loaded artifact) are kept in `visuals/`.

## Regenerating the .usd files

If you change the meshes or need to rebuild them, convert with
`omni.kit.asset_converter` **in meters**. The unit flag is not optional:

- glTF is authored in meters; the converter's default output is
  centimetres (`metersPerUnit = 0.01`).
- USD references do **not** rescale by `metersPerUnit` — it is advisory. The
  robot stage is `metersPerUnit = 1.0`, so a centimetre-unit connector is read
  as **100× too large** and fills the whole frame.

Convert with `converter_context.use_meter_as_world_unit = True` so the output
is `metersPerUnit = 1.0`. Verify the result: each connector's world-space
bounding box should be a few centimetres (the SFP module is roughly
`0.015 × 0.056 × 0.012 m`), not a few metres. Keep `ignore_materials = False`
so the converter also emits the `visuals/textures/` directory the `.usd`
files reference.

Sketch (run inside the container via `./isaaclab.sh -p`, after `AppLauncher`):

```python
import omni.kit.asset_converter as ac
ctx = ac.AssetConverterContext()
ctx.ignore_materials = False
ctx.use_meter_as_world_unit = True          # <- the load-bearing flag
task = ac.get_instance().create_converter_task(src_glb, dst_usd, None, ctx)
await task.wait_until_finished()
```

Then, on the robot USD, swap each `visual` prim's reference from the `.glb`
to the sibling `.usd` (e.g. via `pxr.Sdf.Reference` with the `.usd` asset
path) and save the root layer.

## Reverting

The robot USD is git-tracked, so the `.glb`-referencing version is one
command away:

```
git checkout source/aic_task/aic_task/assets/robots/ur5e_cable/aic_unified_robot_cable_sdf.usd
```

then delete the generated `visuals/*.usd` and `visuals/textures/`. Note the
revert restores the broken-in-this-container `.glb` references — only do it if
the glTF file-format plugin has been made available again.
