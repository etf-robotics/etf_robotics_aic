# 6. Artefakti i arhiva (checkpointi, dataset, snimci)

`outputs/` i `datasets/` su gitignorovani — GitHub nosi samo kod. Ovde je
zapisano šta postoji, šta je arhivirano van mašine i kako se vraća.

## Šta je arhivirano na Hugging Face Hub (nalog `Mihajlo04`)

> Repoi su kreirani kao **privatni** — za predaju nasledniku ili javno deljenje
> prebaci ih na public u Settings, ili dodaj korisnika kao collaboratora.

### Model repo: [`Mihajlo04/aic-port-insertion-policies`](https://huggingface.co/Mihajlo04/aic-port-insertion-policies)

Finalni checkpoint svakog runa iz teze (sa `training_state` — može i `--resume`):

| Putanja u repou | Run iz teze | Veličina |
|---|---|---|
| `dp_phase_full_001/checkpoints/200000` | **DP full** (30290 ep.) — glavni model | 3.1 GB |
| `act_full_5090_run1/checkpoints/512000` | **ACT full** (30290 ep.) | 592 MB |
| `dp_10000_run1/checkpoints/066000` | DP 10k (bez random cropa!) | 3.1 GB |
| `act_10000_run1/checkpoints/169000` | ACT 10k | 592 MB |
| `dp_phase_smoke1k/checkpoints/005000` | DP 1k (pilot) | 3.1 GB |
| `act_1000_run1/checkpoints/017000` | ACT 1k | 592 MB |

### Dataset repo: [`Mihajlo04/aic-port-insertion-eval`](https://huggingface.co/datasets/Mihajlo04/aic-port-insertion-eval)

Kompletan kurirani `outputs/eval/` (~600 MB): `metrics.csv`, `summary.txt`,
`analysis/` i mp4 snimci svih kanonskih runova + multiseed + aux eksperimenti
(organizacija foldera opisana u [05](05_evaluacija_i_analiza.md)).

### Vraćanje na mašinu

```bash
# u kontejneru (huggingface_hub je već tu preko lerobota); hf token po potrebi
docker exec -e HF_TOKEN=<token> -w /workspace/isaaclab isaac-lab-base ./isaaclab.sh -p -c "
from huggingface_hub import snapshot_download
snapshot_download('Mihajlo04/aic-port-insertion-policies',
                  local_dir='etf_robotics_aic/outputs')
snapshot_download('Mihajlo04/aic-port-insertion-eval', repo_type='dataset',
                  local_dir='etf_robotics_aic/outputs')"
```

Time checkpointi sednu na iste putanje koje očekuju `eval_demos.py` i
`scripts/thesis/*` (eval završi u `outputs/eval/`).

## Šta NIJE arhivirano (postoji samo na ovoj mašini)

| Šta | Gde | Zašto nije |
|---|---|---|
| LeRobot dataset (30290 ep., **47 GB**) | `datasets/port_insertion/001_20260612-132535` | Prevelik; regeneriše se sa `collect_demos.py` (~isti kvalitet, drugi seed). Odluka: čuva se samo lokalno. |
| Međukoraci checkpointa (svakih 10–20k) | `outputs/<run>/checkpoints/` | Retko kome trebaju; finalni su na HF. |
| wandb lokalni datastore | `outputs/<run>/wandb/` | Online kopija: wandb entity `sm220315d-etf-`, projekti `aic-{act,dp}-phase-port-insertion`. |

## Snimci i teza

- **YouTube (javno):** uspešna DP epizoda — https://youtu.be/SS5MC-dcSX4
  (izvor: `outputs/eval/demo_dp_success/`, i u HF eval repou)
- **Teza (PDF + LaTeX):** https://github.com/MihStev/DIPLOMSKI (zaseban repo;
  lokalno `DIPLOMSKI/`)
- **Google Drive:** folder [`AIC-arhiva`](https://drive.google.com/drive/folders/1AmU_5tWE0ZTnm1_WRBRRf2me55lSAqJf)
  (nalog mihastevanovic04@gmail.com) — indeks svih lokacija + finalna statistika evaluacije
