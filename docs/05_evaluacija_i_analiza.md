# 5. Evaluacija i analiza

## Closed-loop evaluacija checkpointa

[scripts/eval_demos.py](../scripts/eval_demos.py) pušta politiku da vozi robota
u istom env-u iz kog je dataset snimljen. **ACT ili DP se prepoznaje automatski**
iz `config.json` checkpointa — ista komanda za obe:

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/eval_demos.py \
  --headless --enable_cameras \
  --ckpt etf_robotics_aic/outputs/<run>/checkpoints/last/pretrained_model \
  --n_episodes 50 --seed 0 --save_videos
```

- Izlaz: `outputs/eval/<NNN>_<timestamp>/` sa `metrics.csv` (per-episode ishodi)
  i, uz `--save_videos`, dva mp4 po epizodi: `_overview` (treće lice, ceo robot —
  za figure) i `_cams` (traka tri policy kamere).
- Default overview kamera (`--overview_eye 0.55 0.55 0.45 --overview_target
  -0.38 -0.48 0.11`) hvata i robota i ploču — verifikovano za figure teze.
- **ACT specifično:** eval ide sa temporal ensemblingom (`--temporal_ensemble_coeff
  0.01`, `n_action_steps=1`) — to je ACT-ispravan closed-loop mod.
- `--no_connector_retarget` — goli vrh kabla kao u treningu (vidi
  [07](07_poznati_problemi.md#konektori-se-ne-renderuju)).
- `--gui` umesto `--headless` — gledaj uživo kroz Isaac Sim viewport.

## Statistika i klasifikacija otkaza

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/analyze_eval.py \
  --eval_dir etf_robotics_aic/outputs/eval/<run_dir>
```

Čita `metrics.csv` i piše `analysis/` izveštaj: agregati + klasifikacija
epizoda (`stalled_far` ≥ 90 mm, `approached` < 60 mm, pragovi podesivi).

## Organizacija kuriranih rezultata (`outputs/eval/`)

Metrika svuda: **d_min = najbliži prilaz vrha konektora sedištu porta (tip→seat)**.
Pazi: stariji runovi su merili TCP umesto vrha — nisu uporedivi (takvi su obrisani).

| Folder | Šta drži |
|---|---|
| `DP_full/`, `ACT_full/` | Kanonski runovi punih politika (30290 ep., N=50, seed 0) |
| `DP_1000ep/`, `ACT_1000/`, `DP_10000/`, `ACT_10000/` | Data-scaling tačke |
| `multiseed/ms_{act,dp}_s{1,2}` | Ponovljeni evalovi punih politika, seedovi 1 i 2 |
| `aux_eksperimenti/` | Kontrole: ACT-full goli vrh; ACT-full checkpoint 420k |
| `demo_dp_success/` | Uspešna DP epizoda (izvor YouTube snimka) |

Skripte u [scripts/thesis/](../scripts/thesis/) čitaju tačno ovu strukturu
(gleđaju run-dir unutar svakog kuriranog foldera, pa preživljavaju re-eval).

## Multiseed protokol

Uspešnost se izveštava preko 3 eval seeda × 50 epizoda: seed 0 = kanonski
kurirani run, seedovi 1 i 2 u `multiseed/`. Agregacija (mean ± std):

```bash
python scripts/thesis/multiseed_agg.py
```

## Ključni nalazi (da ne otkrivaš ponovo)

- **ACT ≈ 0 % uspeha na ovom zadatku za SVE checkpointe** (i 512k i 420k sa
  najnižim L1) — parkira se ~4 cm ispred porta i okine `failed_stationary`.
  Nije do izbora checkpointa. Multiseed: ACT 0.7 ± 0.9 %, DP 1.3 ± 1.9 %.
- **Medijana d_min ~47 mm je plato za sve 4 kombinacije** (arhitektura- i
  data-nezavisan) → ograničenje je završna kontaktna faza (DiffIK bez povratne
  sile), ne model. Podaci popravljaju poravnanje (off-axis), najviše DP-u.
- `dp_10000_run1` je treniran sa `crop_shape=None` (bez random cropa) — razlikuje
  se od DP pilot/full (216×216); imaj u vidu pri poređenju.
- Figure teze se regenerišu skriptama iz `scripts/thesis/` (vidi
  [README](../scripts/thesis/README.md)); izlaz ide u `DIPLOMSKI/images/`.
