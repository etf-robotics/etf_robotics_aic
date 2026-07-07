import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdGeom, Gf  # noqa: E402
st = Usd.Stage.Open(a.usd); dp = st.GetDefaultPrim(); xf = UsdGeom.Xformable(dp)
existing = xf.GetOrderedXformOps()
sop = xf.AddScaleOp(opSuffix="unitfix"); sop.Set(Gf.Vec3f(0.01, 0.01, 0.01))
xf.SetXformOpOrder([sop] + existing)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
st.GetRootLayer().Save()
open("/tmp/sc.txt","w").write("scaled 0.01 + mpu=1.0\n")
app.close()
