# 4. Treniranje politika (ACT i Diffusion Policy)

Obe skripte su tanki wrapperi oko `lerobot-train` koji dodaju **pomoćnu phase
glavu** (3-klasna CE na `annotation.phase`; aktivna samo u treningu, u inference-u
ponašanje identično vanila politici) i prosleđuju sve ostale flagove lerobotu:

- [scripts/train_demos.py](../scripts/train_demos.py) — ACT
- [scripts/train_dp_demos.py](../scripts/train_dp_demos.py) — Diffusion Policy

Pre prvog treninga prođi [02_instalacija_zavisnosti.md](02_instalacija_zavisnosti.md)
(posebno torchcodec — 3× kraći trening).

## Validirane konfiguracije (RTX 5090, dataset 001, 30290 ep.)

### Diffusion Policy — full run (`dp_phase_full_001`, ~8 h)

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/train_dp_demos.py \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \
  --policy.crop_shape='[216, 216]' --policy.crop_is_random=true \
  --dataset.repo_id=aic/port_insertion \
  --dataset.root=etf_robotics_aic/datasets/port_insertion/001_20260612-132535 \
  --dataset.video_backend=torchcodec \
  --output_dir=etf_robotics_aic/outputs/<ime_runa> \
  --batch_size=64 --steps=200000 --num_workers=8 \
  --save_freq=10000 --eval_freq=0 --log_freq=200 \
  --wandb.enable=true --wandb.entity=sm220315d-etf- \
  --wandb.project=aic-dp-phase-port-insertion
```

### ACT — full run (`act_full_5090_run1`, 512k koraka ≈ 4 epohe, ~16 h)

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/train_demos.py \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.optimizer_lr=3e-5 --policy.optimizer_lr_backbone=1e-5 \
  --dataset.repo_id=aic/port_insertion \
  --dataset.root=etf_robotics_aic/datasets/port_insertion/001_20260612-132535 \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --output_dir=etf_robotics_aic/outputs/<ime_runa> \
  --batch_size=64 --steps=512000 --num_workers=16 \
  --save_freq=20000 --eval_freq=0 --log_freq=200 \
  --wandb.enable=true --wandb.entity=sm220315d-etf- \
  --wandb.project=aic-act-phase-port-insertion
```

**Ablacija na podskupu epizoda:** dodaj `--dataset.episodes='[0..999]'` i smanji
korake proporcionalno da epohe budu uporedive (ACT-1000 je koristio
`--steps=17000` ≈ 4 epohe).

## Izlaz

`outputs/<ime_runa>/checkpoints/<step>/` (+ symlink `last`), u svakom
`pretrained_model/model.safetensors` (~207 MB ACT, ~1 GB DP) i `training_state/`
za `--resume`. Praćenje toka bez wandb-a:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/training_status.py --run <ime_runa>
```

(Puca ako `checkpoints/` još ne postoji — prvi checkpoint nastaje tek na `--save_freq`.)

## Zamke (svaka je koštala bar jedan propao run)

- **`--policy.push_to_hub=false` je obavezan** — inače lerobot traži `policy.repo_id`.
- **lerobot odbija postojeći `--output_dir`** (`FileExistsError`) — obriši dir
  ili koristi `--resume=true`.
- **Ne redirektuj log u `outputs/`** (`> outputs/foo.log`) — kontejner te
  direktorijume pravi kao root, host-side redirect padne na `Permission denied`
  pre starta. Loguj u home/scratch.
- **`/dev/shm`**: ako dataloader umire uz `Bus error`, kontejner nije podignut
  sa `shm_size: 8gb` (vidi [01](01_pokretanje_okruzenja.md)).
- **`num_workers`: 16 je sweet spot** na ovoj mašini (9950X). Više od 16 diže
  `updt_s` (core contention), manje ostavlja GPU gladnim.
- **GPU osciluje 0↔97 %?** To je CPU-bound decode/augmentacija, ne broj workera.
  Najskuplji transform je `affine` — `--dataset.image_transforms.tfs.affine.weight=0`
  skoro pinuje GPU i seče ~40 % vremena, a zadržava fotometrijske augmentacije.
  (Košta samo vreme, ne kvalitet modela.)
- **`--policy.optimizer_grad_clip_norm` ne postoji za ACT** — preset već ima
  `grad_clip_norm=10.0`; postoje samo `optimizer_lr`, `optimizer_lr_backbone`,
  `optimizer_weight_decay`.
- **Phase metrike (`phase_acc`, `phase_ce`) se NE vide u wandb-u** — lerobotov
  MetricsTracker loguje fiksan skup ključeva. Aux loss ipak deluje (uračunat u
  `train/loss`); konvergenciju čitaj kao `train/loss − l1_loss − kl_weight·kld_loss`
  (ACT), analogno za DP.
- ACT-u je normalno da `kld_loss` padne na ~0 (kolaps VAE latenta).
