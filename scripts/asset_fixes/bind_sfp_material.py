"""Re-bind sfp_module's mesh to its two materials via GeomSubsets.

The .glb->.usd conversion merged Body.005's two glTF primitives (mat .005 over
6684 tris, mat .001 over the next 184) into one USD Mesh and dropped the material
binding -> the connector renders with the default (dark) material. Face order is
preserved (6684+184 == the mesh's 6868 faces), so we recreate two ``materialBind``
GeomSubsets over those face ranges and bind each to its material, plus a whole-mesh
fallback bind to the dominant material.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdShade, UsdGeom, Vt  # noqa: E402

out = []
st = Usd.Stage.Open(a.usd)
mesh_prim = st.GetPrimAtPath("/World/Body_005")
mesh = UsdGeom.Mesh(mesh_prim)
mat5 = UsdShade.Material(st.GetPrimAtPath("/World/Looks/Material_005"))
mat1 = UsdShade.Material(st.GetPrimAtPath("/World/Looks/Material_001"))
out.append(f"mesh valid={bool(mesh_prim)}  mat5 valid={bool(mat5)}  mat1 valid={bool(mat1)}")

# whole-mesh fallback -> dominant material
UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat5)

faces0 = Vt.IntArray(list(range(0, 6684)))      # Material_005
faces1 = Vt.IntArray(list(range(6684, 6868)))   # Material_001
s0 = UsdGeom.Subset.CreateGeomSubset(mesh, "Material_005", UsdGeom.Tokens.face, faces0, "materialBind")
s1 = UsdGeom.Subset.CreateGeomSubset(mesh, "Material_001", UsdGeom.Tokens.face, faces1, "materialBind")
UsdShade.MaterialBindingAPI.Apply(s0.GetPrim()).Bind(mat5)
UsdShade.MaterialBindingAPI.Apply(s1.GetPrim()).Bind(mat1)

st.GetRootLayer().Save()
out.append("SAVED.")

# verify on a reloaded stage
st2 = Usd.Stage.Open(a.usd)
mb = UsdShade.MaterialBindingAPI(st2.GetPrimAtPath("/World/Body_005")).ComputeBoundMaterial()[0]
out.append(f"verify mesh fallback -> {mb.GetPath() if mb else 'NONE'}")
for sp in ("/World/Body_005/Material_005", "/World/Body_005/Material_001"):
    prim = st2.GetPrimAtPath(sp)
    b = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0] if prim else None
    out.append(f"verify subset {sp} -> {b.GetPath() if b else 'MISSING/NONE'}")
open("/tmp/bind.txt", "w").write("\n".join(out) + "\n")
app.close()
