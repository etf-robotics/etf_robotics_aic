# Installing lerobot into the `isaac-lab-base` container

Recipe for adding [lerobot](https://github.com/huggingface/lerobot) (dataset writer
backend) to a fresh `isaac-lab-base` Docker container.

> These changes live in the **running container's writable layer**, not the image.
> If the container is recreated (`docker compose down`/`up`, `docker rm`), re-run all
> three steps — or bake them into a Dockerfile layer (see [Making it permanent](#making-it-permanent)).

## Prerequisites

- `isaac-lab-base` container running (`docker ps --filter name=isaac-lab-base`)
- Host on kernel `6.17.0-29-generic` with NVIDIA driver `580.159.03` (verified working setup)

## Three steps, in order

Skipping any of them breaks at runtime.

### 1. Install lerobot

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install lerobot
```

Lands **lerobot 0.4.4**. As a side effect it bumps `numpy` → 2.x, `packaging` → 25.0,
plus updates to `huggingface_hub`, `wandb`, `protobuf`, `pyarrow`.

### 2. Restore the vendored `packaging` directory the install wiped

Pip's uninstall of the old `packaging 23.0` also removes files from
`/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging/`, which isaac-sim's
bundled torch (and pip itself) reads via symlink. Without this step you get
`FileNotFoundError: ..._vendor/packaging/_structures.py` everywhere and pip itself is broken.

```bash
docker exec isaac-lab-base bash -c "
  rm -rf /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging &&
  cp -rL /isaac-sim/kit/python/lib/python3.11/site-packages/packaging \
         /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging"
```

`-L` is critical — the source tree contains symlinks back into the location you're
restoring, so a plain `cp -r` creates broken self-references.

### 3. Pin numpy back below 2

lerobot only declared `numpy>=2` for install; it runs fine on 1.26. But
`omni.syntheticdata` (camera init) fails on numpy 2.x with
`TypeError: Unable to write from unknown dtype, kind=f, size=0`.

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install "numpy<2"
```

## Final pinned versions

| Package | Version | Why |
|---|---|---|
| lerobot | 0.4.4 | dataset writer backend |
| numpy | 1.26.4 | required by isaac-sim's `omni.syntheticdata`; lerobot tolerates it at runtime |
| packaging | 25.0 | lerobot pulled this in; also restored into `omni.isaac.core_archive/pip_prebundle/packaging/` for isaac-sim's vendored torch |
| torch | 2.7.0+cu128 | unchanged from base container |

A `rerun-sdk requires numpy>=2` pip warning after step 3 is expected and harmless.

## Smoke test (run before `collect_demos`)

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -c "
import numpy, torch, isaaclab
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('numpy', numpy.__version__, 'torch', torch.__version__, 'OK')"
```

Expected: `numpy 1.26.4 torch 2.7.0+cu128 OK`.

- `FileNotFoundError` mentioning `_vendor/packaging/_structures.py` → step 2 didn't take.
- isaac-sim imports but the camera errors at run time → step 3 didn't take.

## Making it permanent

To survive container recreation, bake the steps into a Dockerfile layer on top of the
base image, e.g.:

```dockerfile
FROM isaac-lab-base

RUN ./isaaclab.sh -p -m pip install lerobot && \
    rm -rf /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging && \
    cp -rL /isaac-sim/kit/python/lib/python3.11/site-packages/packaging \
           /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging && \
    ./isaaclab.sh -p -m pip install "numpy<2"
```
