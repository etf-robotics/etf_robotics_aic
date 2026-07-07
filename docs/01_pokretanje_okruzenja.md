# 1. Pokretanje okruženja (Docker + Isaac Lab)

Na hostu **nema** instalacije Isaac Lab-a — sve se izvršava u Docker kontejneru
`isaac-lab-base`. Ovaj repo (`etf_robotics_aic/`) živi unutar klona
[IsaacLab](https://github.com/isaac-sim/IsaacLab) repoa i montiran je u kontejner
na putanji `/workspace/isaaclab/etf_robotics_aic/`.

## Pokretanje kontejnera

Kontejner se diže preko Isaac Lab-ovog wrappera (traži sudo lozinku):

```bash
cd ~/IsaacLab
sudo docker/container.py start base
```

To koristi `docker/docker-compose.yaml` + `docker/x11.yaml` (X11 patch za GUI).
U našem `docker-compose.yaml` su već dodate dve bitne izmene:

- **`shm_size: '8gb'`** — podrazumevanih 64 MB `/dev/shm` obara multi-worker
  dataloader tokom treninga (`Bus error`).
- **Bind mount `~/.netrc` (read-only)** — wandb login preživljava rekreaciju
  kontejnera. (Ne stavljati `WANDB_API_KEY` u `docker/.env.base` — taj fajl je
  u git-u i ključ bi procureo!)

Provera da kontejner radi:

```bash
docker ps --filter name=isaac-lab-base
```

## Izvršavanje komandi u kontejneru

Korisnik je u `docker` grupi, pa `docker exec` radi **bez sudo**. Obrazac za sve:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p <skripta> [argumenti]
```

- `-w /workspace/isaaclab` postavlja radni direktorijum (tu je montiran IsaacLab).
- `./isaaclab.sh -p` je python passthrough — bira ispravan Isaac Sim python.
  `isaaclab` **nije** na PATH-u u non-interactive shell-u, zato uvek preko skripte.
- Sve posle `-p` ide pythonu: putanja skripte + njeni argumenti, ili `-c "..."`.

Interaktivni shell (kad zatreba): `sudo ~/IsaacLab/docker/container.py enter base`
— unutra `isaaclab -p ...` radi direktno.

## Zamka: `pxr` import

Većina `isaaclab.*` modula importuje `pxr` pri učitavanju, a `pxr` postoji tek
kad `AppLauncher` pokrene Isaac Sim. Zato ovo **pada**:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p \
  -c "from isaaclab.envs import ManagerBasedRLEnvCfg"   # ModuleNotFoundError: pxr
```

Svaka naša skripta prvo diže `AppLauncher`, pa tek onda importuje ostalo —
koristi postojeće skripte kao šablon ako pišeš novu.

## Smoke test okruženja

Najbrža provera da task radi (gradi ceo env, prolazi kroz sve managere):

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/direct_entrance_approach.py \
  --headless --enable_cameras --num_envs 1
```

- `--headless` — bez Kit prozora.
- `--enable_cameras` — **obavezno**: obs grupa ima tri `TiledCamera` senzora;
  bez flega sim init baca `RuntimeError: A camera was spawned without ...`.
- Skripta vrti goal-driven kontroler beskonačno — za smoke test prekini
  (Ctrl-C) čim prođu `[INFO]` linije observation managera.

## GUI (X11)

Za posmatranje simulacije uživo izostavi `--headless` (npr. `eval_demos.py --gui`).
`DISPLAY` je već podešen u kontejneru (`:1`). Ako X11 ne radi, proveri da je
kontejner podignut sa `x11.yaml` patch-om. Za gamepad ulaz u kontejner vidi
[DOCKER_GAMEPAD_INPUT.md](DOCKER_GAMEPAD_INPUT.md).

## ⚠️ Rekreacija kontejnera briše writable layer

`docker compose down` / `docker rm` briše **sve što nije u image-u ili bind
mount-u**: pip pakete (lerobot, torchcodec), apt pakete (ffmpeg)... Posle
rekreacije ponovi celu instalaciju iz
[02_instalacija_zavisnosti.md](02_instalacija_zavisnosti.md).
(wandb login preživljava zahvaljujući `.netrc` mount-u.)
