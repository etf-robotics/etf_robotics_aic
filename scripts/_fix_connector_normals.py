"""One-time fix: strip the corrupted ``normals`` on the connector visual USD.

``sfp_module_visual.usd`` carries a broken ``normals`` primvar (faceVarying
count mismatch -> Hydra shades it near-black, darkening the whole eval render).
The valid ``.glb`` source can't load (no glTF plugin), so instead of re-converting
we just remove the bad normals from the ``.usd``: with no authored normals, Hydra
computes them from the geometry at render time -> correct (bright) shading.

Reports every Mesh (points / faceVertexIndices / normals count + interpolation)
before and after so we can see exactly what was removed. Writes a report to
/tmp/fix_normals_report.txt (Kit swallows stdout). pxr needs the sim app, so we
launch AppLauncher headless first.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True, help="Path to the connector .usd to fix.")
parser.add_argument("--apply", action="store_true",
                    help="Actually remove normals + save. Without it, only reports (dry run).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from pxr import Usd, UsdGeom  # noqa: E402

lines = [f"USD: {args.usd}", f"apply: {args.apply}", ""]
stage = Usd.Stage.Open(args.usd)
meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
lines.append(f"found {len(meshes)} Mesh prim(s)")

removed_any = False
for prim in meshes:
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    n_attr = mesh.GetNormalsAttr()
    nrm = n_attr.Get()
    interp = mesh.GetNormalsInterpolation()
    pv_api = UsdGeom.PrimvarsAPI(prim)
    has_pv_normals = pv_api.HasPrimvar("normals")
    lines.append(
        f"\n  {prim.GetPath()}\n"
        f"    points={len(pts) if pts is not None else 0}  "
        f"faceVertexIndices={len(fvi) if fvi is not None else 0}\n"
        f"    normals attr: count={len(nrm) if nrm is not None else 0}  interp={interp}  "
        f"authored={n_attr.HasAuthoredValue()}\n"
        f"    primvars:normals present={has_pv_normals}"
    )
    if args.apply:
        if n_attr.HasAuthoredValue() or prim.HasProperty("normals"):
            prim.RemoveProperty("normals")
            removed_any = True
        for pname in ("primvars:normals", "primvars:normals:indices"):
            if prim.HasProperty(pname):
                prim.RemoveProperty(pname)
                removed_any = True
        lines.append("    -> removed normals (Hydra will recompute)")

if args.apply and removed_any:
    stage.GetRootLayer().Save()
    lines.append("\nSAVED.")
    # Re-open to confirm normals are gone.
    stage2 = Usd.Stage.Open(args.usd)
    leftover = [
        str(p.GetPath()) for p in stage2.Traverse()
        if p.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(p).GetNormalsAttr().HasAuthoredValue()
    ]
    lines.append(f"verify: meshes still carrying authored normals = {leftover or 'none'}")
elif args.apply:
    lines.append("\nnothing to remove (no authored normals found).")

with open("/tmp/fix_normals_report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

app.close()
