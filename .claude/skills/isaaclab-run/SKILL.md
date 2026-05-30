---
name: isaaclab-run
description: How to actually run isaaclab — Python, scripts, or env-build smoke tests — for this repo. The host has no isaaclab install; everything runs inside the already-running `isaac-lab-base` Docker container, reached via `docker exec` (no sudo). Invoke whenever you need to import `aic_task`, exercise `build_observation_cfg`, run `direct_entrance_approach.py`, or otherwise verify a change by executing code rather than reading it. Skip for static reads / greps / file edits.
---

# isaaclab-run

The host has no isaaclab. Anything that needs to actually execute
Python with isaaclab in scope must go through the running container.

## Container access

Container `isaac-lab-base` is already up and long-lived. The user is in
the `docker` group, so `docker exec` and `docker ps` work **without
sudo**. Always prefer `docker exec` over the sudo-gated
`container.py enter` — the latter opens an interactive shell unsuitable
for tool calls.

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p <args>
```

- `-w /workspace/isaaclab` sets the workdir; the repo is mounted at
  `/workspace/isaaclab/etf_robotics_aic/`.
- `./isaaclab.sh -p` is the python passthrough. `isaaclab` is **not**
  on PATH in a fresh non-interactive shell — use the script path.
- Anything after `-p` is forwarded to python (`-c "..."`, a script
  path + its argv, etc.).

## The `pxr` trap (read before writing a `-c` snippet)

Most `isaaclab.*` modules import `pxr` at module-load time, but `pxr`
is only on the python path once `AppLauncher` has started Isaac Sim.
So this **will** fail:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \
  -c "from isaaclab.envs import ManagerBasedRLEnvCfg"
# ModuleNotFoundError: No module named 'pxr'
```

Plain `-c` works only for top-level `import aic_task` sanity. For
anything that touches `isaaclab.envs`, `isaaclab.sim`,
`isaaclab.controllers`, etc., you must launch the sim app first.
The fastest way is to use an existing entry script — it already does
the AppLauncher dance.

## Fastest path to validate a port-insertion task change

Use the existing
[`scripts/direct_entrance_approach.py`](../../../etf_robotics_aic/scripts/direct_entrance_approach.py).
It launches AppLauncher, registers the Gym IDs, and calls
`gym.make("AIC-Port-Insertion-v0")` — which fully exercises the
observation/event/termination managers (this is where body-name
resolution and obs-function errors surface):

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/direct_entrance_approach.py \
  --headless --enable_cameras --num_envs 1
```

- `--headless` skips opening a Kit window.
- `--enable_cameras` is **required** because the policy obs group
  includes three `TiledCameraCfg` sensors; without it, sim init raises
  `RuntimeError: A camera was spawned without the --enable_cameras flag`.
- `--num_envs 1` keeps GPU/RAM pressure low for a smoke test.

The script runs a goal-driven controller indefinitely; for a smoke
test, stop after you see the post-init `[INFO]` lines from the
observation manager (or the failing traceback). To exit, interrupt the
process or set a step cap if you need a clean termination.

## Interactive shell (only when the user asks)

The documented entry is `sudo ~/IsaacLab/docker/container.py enter base`.
Sudo will prompt for the user's password — never put the password in a
command (even via `sudo -S`). Have the user type it themselves. Inside
that interactive shell, `isaaclab` *is* on PATH, so `isaaclab -p ...`
works.

## X11

Only relevant if launching the sim GUI (drop `--headless`). Headless
imports, env-build smoke tests, and BC dataset recording do not need
X11. If the user actually needs the GUI and forwarding into the
container isn't working, ask — `~/IsaacLab/docker/x11.yaml` is the
compose patch they may need to layer in via `container.py`.

## What this skill is NOT for

- Anything read-only (file reads, greps, git inspection) — those run
  on the host, no container needed.
- Long training runs — the user starts those themselves; this skill
  is for quick verification, not multi-hour jobs.
- Modifying container state (installing packages, editing config).
  If the user requests that, confirm first — the container is shared
  across their work, not throwaway.
