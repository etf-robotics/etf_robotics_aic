# 2. Instalacija zavisnosti (LeRobot, torchcodec, ffmpeg, wandb)

Sve ispod se instalira **u kontejner** (writable layer) i **nestaje pri
rekreaciji kontejnera** — tada ponovi sve korake ovim redom.

## LeRobot (dataset writer + treneri)

Tri koraka, redosled je bitan — detaljno objašnjenje svakog u
[lerobot-install.md](lerobot-install.md) (engleski, sa smoke testom):

```bash
# 1) instalacija (lerobot 0.4.4)
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -m pip install lerobot

# 2) vrati vendored packaging koji je pip uninstall obrisao (inače je i pip slomljen)
docker exec isaac-lab-base bash -c "
  rm -rf /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging &&
  cp -rL /isaac-sim/kit/python/lib/python3.11/site-packages/packaging \
         /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging"

# 3) numpy nazad ispod 2 (omni.syntheticdata puca na numpy 2.x)
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -m pip install "numpy<2"
```

Smoke test:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -c "
import numpy, torch, isaaclab
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('numpy', numpy.__version__, 'torch', torch.__version__, 'OK')"
# očekivano: numpy 1.26.4 torch 2.7.0+cu128 OK
```

## ffmpeg (treba torchcodec-u)

`apt-get install ffmpeg` puca na `tzdata` postinst-u jer je `/etc/localtime`
read-only bind mount. Zaobilazak:

```bash
docker exec isaac-lab-base bash -c '
  printf "#!/bin/sh\nexit 0\n" > /var/lib/dpkg/info/tzdata.postinst.new &&
  cp /var/lib/dpkg/info/tzdata.postinst /tmp/tzdata.postinst.orig 2>/dev/null;
  mv /var/lib/dpkg/info/tzdata.postinst.new /var/lib/dpkg/info/tzdata.postinst &&
  chmod +x /var/lib/dpkg/info/tzdata.postinst &&
  apt-get update && apt-get install -y ffmpeg;
  cp /tmp/tzdata.postinst.orig /var/lib/dpkg/info/tzdata.postinst 2>/dev/null || true'
```

## torchcodec — **presudno za brzinu treninga**

LeRobot za dekodovanje video opservacija bira torchcodec kad god je importabilan,
ali verzija mora da odgovara torchu (**0.4.x ↔ torch 2.7**, 0.5↔2.8, 0.6↔2.9).
Bundled verzija u image-u je ABI-nekompatibilna i pada pri učitavanju.

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p -m pip install torchcodec==0.4.0
```

Izmereno na ovom projektu (bs=64, RTX 5090): **torchcodec 11.7 ms/sample vs
pyav 89.6 ms/sample (7.6×)** — sa torchcodec-om GPU stoji na 96–99 %, ETA punog
DP treninga pada sa ~23 h na ~8 h. `pyav` backend radi bez ičega ekstra, ali je
rezervna opcija samo ako torchcodec zezne.

## wandb

Login je u host fajlu `~/.netrc` koji je bind-mountovan u kontejner — radi odmah.
Ako ikad treba ponovo: `./isaaclab.sh -p -m wandb login <key>` (CLI `wandb`
nije na PATH-u).

- **Entity je `sm220315d-etf-`** (sa crticom na kraju!), ne username.
- Projekti: `aic-act-phase-port-insertion` (ACT), `aic-dp-phase-port-insertion` (DP).

## Trajno rešenje

Da instalacija preživi rekreaciju, ispeci korake u Dockerfile sloj povrh base
image-a — skica u [lerobot-install.md](lerobot-install.md#making-it-permanent)
(dodaj i ffmpeg + torchcodec).
