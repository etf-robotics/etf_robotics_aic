# `scripts/il/` — scripted demo collection

What `scripts/collect_demos.py` does and how the LeRobot writer behind it
is wired up. Covers: how to run, how the writer works, and the
environment prep needed inside the `isaac-lab-base` container.

## What got added

| File | Purpose |
|---|---|
| [`scripts/il/writer.py`](writer.py) | `PortInsertionWriter`: per-env frame buffers, success-only commit, LeRobot v3 dataset on disk. |
| [`scripts/collect_demos.py`](../collect_demos.py) | Replaces the previous `# TODO(writer)` stubs with `writer.record(...)` / `writer.commit(...)` / `writer.close()`. Adds `--out_dir`, `--append`, `--task_label` CLI flags. |
| [`scripts/view_demos.py`](../view_demos.py) | Thin wrapper around `lerobot-dataset-viz` that resolves the latest `NNN_*` run, picks an episode, and produces a Rerun `.rrd` file you can open on the host. |
| [`scripts/train_demos.py`](../train_demos.py) | Wrapper around `lerobot.scripts.lerobot_train` that swaps in a phase-aware ACT subclass: action L1 + KLD + auxiliary cross-entropy on `annotation.phase`. The aux head trains during forward but is never read at `select_action()` — eval and deployment are byte-identical to vanilla ACT. |

No changes to the task definition under `source/aic_task/`. The success
mask comes from the named termination term already in
[`builders.py`](../../source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py)
(`"success"` vs. `"failed_stationary"` vs. `"time_out"`), so distinguishing
a real success from a failure or timeout is a one-line query against the
termination manager.

## How to run

From inside the `isaac-lab-base` container (host has no isaaclab):

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/collect_demos.py \
  --headless --enable_cameras --num_envs 1
```

Flags worth knowing:

- `--num_envs N` — parallel envs. The 3060 Ti (8 GB) fits **1** env with
  three 224×224 RGB cameras; bigger GPUs handle more.
- `--out_dir <path>` — defaults to `<repo_root>/datasets/port_insertion`
  (resolved from `collect_demos.py`'s own location, so it's stable
  regardless of the launcher's working directory). Each run lands in a
  `NNN_<timestamp>/` subdir.
- `--append` — reopen the most recent `NNN_*` run and keep adding
  episodes instead of starting a new one.
- `--task_label "AIC-Port-Insertion-v0"` — written to LeRobot's per-episode
  `task` field. Defaults to the Gym ID.
- `--seed N` — seeds the planner's RNG (approach jitter).
- `--headless --enable_cameras` — required. Without `--enable_cameras`
  the camera obs terms blow up at sensor init.

Stop the run with Ctrl-C (or `pkill -INT -f collect_demos` inside the
container). In-flight episodes are dropped on shutdown; only fully
completed successful ones are on disk.

## How the writer works

Per step, before `env.step`:

```python
action, _ = executor.step(obs)
writer.record(obs, action)         # (s_t, a_t) into per-env buffer
obs, _, terminated, truncated, _ = env.step(action)
```

Per step, after `env.step`:

```python
done = (terminated | truncated).nonzero(as_tuple=False).flatten()
if done.numel() > 0:
    success_mask = env.unwrapped.termination_manager.get_term("success")
    writer.commit(done, success_mask)   # success → flush, else drop
```

`get_term("success")` returns a `(num_envs,)` bool of envs whose success
term fired this step. Envs that ended via `failed_stationary` or
`time_out` have `False` there and their buffers are silently discarded.

### Schema inference

`PortInsertionWriter.__init__` peeks at `env.unwrapped.obs_buf["policy"]`
once (right after `env.reset`) to build the LeRobot feature dict:

- Every key in the policy group whose name does **not** end in `_rgb` is
  appended to a single concatenated `observation.state` vector. With the
  current task that's `joint_pos, joint_vel, joint_torque, tcp_pos_b,
  tcp_quat_b, eef_pos_b, eef_quat_b, tcp_lin_vel_b, tcp_ang_vel_b,
  eef_lin_vel_b, eef_ang_vel_b, wrist_wrench, actions` — total 56 floats.
  The feature's `names` field records the source-term order so the
  splitting is reversible.
- Every `*_rgb` key becomes `observation.images.<name>` (e.g.
  `center_camera_rgb` → `observation.images.center`), stored as
  `dtype: "video"` so LeRobot encodes one mp4 per episode per camera.
- `action` is the (6,) DiffIK delta-pose action fed to `env.step`.

The cheatcode obs group (`entrance_pos_b`, `seat_pos_b`,
`insertion_fraction`, …) is **not** saved. Switch on policy-group-only
matches the BC training surface.

### Per-episode `task` string

Goes into the LeRobot `task` field for every frame. Defaults to
`"AIC-Port-Insertion-v0"` (the Gym ID). Override with `--task_label`.

### Run-directory layout

Under `--out_dir`:

```
datasets/port_insertion/
  001_<timestamp>/        # one LeRobot v3 dataset per run
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.center/chunk-000/file-000.mp4
    videos/observation.images.left/chunk-000/file-000.mp4
    videos/observation.images.right/chunk-000/file-000.mp4
  002_<timestamp>/
  ...
```

The `NNN_` prefix auto-increments based on existing subdirs (zero-padded
to 3 digits), so dataset runs sort naturally. `--append` skips the
auto-increment and reopens the highest-numbered run.

### What's intentionally NOT done

- No on-disk locking. The writer assumes one collector process at a time
  per `--out_dir`. Two concurrent collectors writing into the same run
  would race on episode indices.
- No in-flight checkpointing. If the process is killed mid-episode that
  episode's buffer is lost; only fully completed (and successful)
  episodes ever hit disk.
- No reward / cheatcode columns. Add them by extending
  `PortInsertionWriter._state_keys` (and the LeRobot `features` dict) if
  you need them for asymmetric critics.

## Viewing the dataset

### Quick path: synchronized cameras + state + action in Rerun

The repo ships [`scripts/view_demos.py`](../view_demos.py), a thin
wrapper around lerobot's bundled `lerobot-dataset-viz`. It picks the
most recent run, runs the viz in headless save-mode, writes a Rerun
`.rrd` file, and chowns it back to the host user.

```bash
# Inside the container — produce the .rrd
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/view_demos.py --episode 0
```

The output lands at:

```
datasets/port_insertion/<run>/viz/<run>_episode_0.rrd
```

Open it on the host with the Rerun viewer:

```bash
pip install rerun-sdk      # once
rerun datasets/port_insertion/<run>/viz/<run>_episode_0.rrd
```

Rerun gives you the three camera streams playing in sync with the
56-dim state, the 6-dim action, and the per-phase one-hot — scrub the
timeline, zoom plots, isolate a single camera, etc.

Other modes:

- `--episode N` to pick an episode index.
- `--run NNN_<timestamp>` to target a specific run.
- `--mode local` to spawn the Rerun viewer directly (needs X11 into the
  container).
- `--mode distant --grpc_port 9876` to serve over gRPC; connect from the
  host with `rerun rerun+http://localhost:9876/proxy` (port-mapped via
  the container's compose stack).

### File-by-file fallbacks

You don't need Rerun to inspect the data:

- **Videos**: `videos/observation.images.{center,left,right}/chunk-000/file-*.mp4`
  play in vlc/mpv/xdg-open. They concatenate every successful episode
  for that camera back-to-back.
- **Low-dim columns**: `data/chunk-000/file-*.parquet`. Open with pandas
  / pyarrow / duckdb. Columns include `observation.state` (56,),
  `action` (6,), `annotation.phase` (3,), `episode_index`,
  `frame_index`, `timestamp`, `task_index`.
- **Per-episode frame ranges**: `meta/episodes.parquet` (length per
  episode) lets you slice the concatenated mp4s by frame index.
- **Programmatic loader**: `LeRobotDataset(repo_id=run.name, root=run)`
  gives a torch-style `__getitem__` that decodes mp4 frames into tensors
  for you — pair it with the writer's `_state_keys` list to split
  `observation.state` back into named subvectors.

### Permissions footgun

Episodes are written by the container as `root:root`. The host user
can't read them by default. One-time fix after each collection run:

```bash
docker exec isaac-lab-base \
  chown -R 1000:1000 /workspace/isaaclab/etf_robotics_aic/datasets/port_insertion
```

(`1000` = your host UID; check with `id -u`.) `view_demos.py` chowns
its own output (`viz/*.rrd`) automatically so the loop above only needs
to be run after a `collect_demos.py` session.

## Environment prep — read before installing lerobot

The container ships with isaaclab pinned to `numpy<2`; `lerobot` (>= 0.4)
declares `numpy>=2`. The install will succeed but breaks two things you
have to fix back manually. **Don't skip this.**

```bash
# 1) Install lerobot (upgrades numpy to 2.x, packaging to 25.x,
#    huggingface_hub etc; uninstalls the old pip-prebundled packaging).
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install lerobot
```

Now two breakages need to be undone:

**(a) `packaging` was uninstalled out of the vendored location** that
isaaclab's bundled torch reads. Symptom:
`FileNotFoundError: '/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py'`
and a cascade of "Failed to import python module isaacsim.core.*"
errors. pip itself can also break (its own `_vendor/packaging/_structures.py`
is a symlink into the same wiped dir). Repair by copying the just-installed
`packaging-25.0` files **dereferenced** into the vendored slot:

```bash
docker exec isaac-lab-base bash -c "
  rm -rf /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging &&
  cp -rL /isaac-sim/kit/python/lib/python3.11/site-packages/packaging \
         /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging"
```

`cp -rL` is important — the source dir contains symlinks that point back
into the location you're restoring, so a plain `cp -r` creates broken
self-references.

**(b) numpy 2.x makes `omni.syntheticdata` fail at camera init** with
`TypeError: Unable to write from unknown dtype, kind=f, size=0`. Pin
numpy back below 2 — lerobot 0.4 works fine at runtime with numpy 1.26:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install "numpy<2"
```

**(c) torchcodec must match isaac-sim's torch.** lerobot pulls in
`torchcodec` (it's the default video decoder for `LeRobotDataset`).
The latest torchcodec is built against torch ≥ 2.10; isaac-sim ships
torch 2.7, so importing the bundled torchcodec dies with
`undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev`. Pin to the
0.4.x line, which matches torch 2.7's ABI:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install "torchcodec==0.4.0"
```

This is only needed for **reading** the dataset (training, viz). The
writer doesn't touch torchcodec, so you can skip step (c) on a collector
machine if you only ever produce data there.

Then verify all four stacks still import:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -c "
import numpy, torch, isaaclab
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchcodec.decoders import VideoDecoder
print('numpy', numpy.__version__, 'torch', torch.__version__, 'OK')"
```

### lerobot import path

The 0.4.x layout is `lerobot.datasets.lerobot_dataset`, **not**
`lerobot.common.datasets.lerobot_dataset` (the old `common.` path was
removed). [`writer.py`](writer.py) imports the new path; if the
container ever ends up with an older lerobot, that import is the only
line that needs touching.

## Known footguns

- **Stale GPU processes after a crash.** Isaac Sim doesn't always release
  GPU memory when its Python process dies. `nvidia-smi` lists the
  zombies. Clean with `docker exec isaac-lab-base pkill -9 -f kit/python`
  before the next attempt — on an 8 GB card the next run will OOM
  otherwise.
- **`--out_dir` was relative in the very first version.** It's now
  derived from the script's own location, so launching with
  `docker exec -w /workspace/isaaclab ...` no longer writes into
  container scratch space — episodes land under
  `etf_robotics_aic/datasets/port_insertion/` on the mounted host
  filesystem.
- **LeRobot's `create()` calls `mkdir(exist_ok=False)`** on whatever you
  pass as `root`. The writer therefore creates `out_dir/` itself but
  hands the *per-run subdir* (`out_dir/NNN_<ts>`) to LeRobot. If you
  pre-create the run dir manually, `create()` will raise.
