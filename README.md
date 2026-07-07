# etf_robotics_aic — UR5e ubacivanje kabla u SFP port (AI for Industry Challenge)

Simulaciono okruženje (Isaac Lab / Isaac Sim) i pipeline imitacionog učenja
(LeRobot: ACT i Diffusion Policy) za zadatak ubacivanja SFP kabla u port
UR5e robotom. Razvijeno kao diplomski rad na ETF-u (Mihajlo Stevanović, 2026).

> **Novi na projektu?** Kreni od [docs/01_pokretanje_okruzenja.md](docs/01_pokretanje_okruzenja.md)
> i idi redom. Sve što je otkriveno teškim putem je zapisano tamo.

## 📦 Arhiva modela i rezultata (Hugging Face, javno)

Veliki artefakti ne stoje u git-u — trajno su arhivirani na HF nalogu autora:

- **Checkpointi svih 6 politika** (DP/ACT × 1k/10k/30k demonstracija):
  [huggingface.co/Mihajlo04/aic-port-insertion-policies](https://huggingface.co/Mihajlo04/aic-port-insertion-policies)
- **Kurirani eval rezultati** (metrike, analize, mp4 snimci):
  [huggingface.co/datasets/Mihajlo04/aic-port-insertion-eval](https://huggingface.co/datasets/Mihajlo04/aic-port-insertion-eval)

Uputstvo za vraćanje na očekivane putanje: [docs/06_artefakti_i_arhiva.md](docs/06_artefakti_i_arhiva.md).

## Pipeline ukratko

```
collect_demos.py          train_demos.py (ACT)         eval_demos.py         analyze_eval.py
(skriptovani oracle)  →   train_dp_demos.py (DP)   →   (closed-loop eval  →  (statistika +
30290 uspešnih epizoda    LeRobot + phase aux head     + mp4 snimci)         klasifikacija otkaza)
```

Zadatak je registrovan kao Gym ID **`AIC-Port-Insertion-v0`**
(paket [source/aic_task/](source/aic_task/), detaljna dokumentacija paketa u
[source/aic_task/docs/](source/aic_task/docs/)).

## Glavni rezultati (detalji u tezi)

- **Uspešnost (N=150, seedovi 0–2):** DP full 1.3 ± 1.9 %, ACT full 0.7 ± 0.9 %.
- **Medijana najbližeg prilaza (d_min) je tvrd plato ~47 mm za sve 4 kombinacije**
  (DP/ACT × {1000, 30290 epizoda}) — usko grlo je završna kontaktna faza
  (DiffIK bez sile), ne kapacitet modela niti količina podataka.
- Više podataka popravlja pre svega **poravnanje** (off-axis grešku), najviše za DP.
- Snimak uspešne DP epizode: https://youtu.be/SS5MC-dcSX4

## Struktura repoa

| Putanja | Šta je |
|---|---|
| `source/aic_task/` | Task paket: scena, MDP termovi, Gym registracija ([dokumentacija](source/aic_task/docs/)) |
| `scripts/` | Pipeline: prikupljanje, trening, eval, analiza (vidi tabelu ispod) |
| `scripts/il/` | Oracle planer (APPROACH→ALIGN→INSERT) + LeRobot writer |
| `scripts/asset_fixes/` | Jednokratne popravke USD asseta ([README](scripts/asset_fixes/README.md)) |
| `scripts/thesis/` | Generatori figura za tezu ([README](scripts/thesis/README.md)) |
| `docs/` | **Tutorijali** — pokretanje, instalacija, trening, eval, arhiva |
| `datasets/` | LeRobot dataseti (gitignorovano, ~47 GB na disku) |
| `outputs/` | Checkpointi + eval rezultati (gitignorovano; arhiva → [docs/06](docs/06_artefakti_i_arhiva.md)) |
| `DIPLOMSKI/` | Teza (zaseban repo: https://github.com/MihStev/DIPLOMSKI, gitignorovano ovde) |

## Glavne skripte

| Skripta | Uloga |
|---|---|
| `scripts/collect_demos.py` | Skriptovano prikupljanje demonstracija (multi-env, čuva samo uspešne epizode) |
| `scripts/train_demos.py` | Trening ACT + pomoćna phase glava (wrapper oko `lerobot-train`) |
| `scripts/train_dp_demos.py` | Trening Diffusion Policy + phase glava |
| `scripts/eval_demos.py` | Closed-loop evaluacija checkpointa (ACT/DP auto-dispatch), mp4 snimci |
| `scripts/analyze_eval.py` | Statistika + klasifikacija otkaza jednog eval runa |
| `scripts/view_demos.py` | Pregled epizoda dataseta u Rerun vieweru |
| `scripts/training_status.py` | Snapshot toka treninga (ETA iz checkpoint mtime-ova) |
| `scripts/direct_entrance_approach.py` | Smoke test env-a: goal-driven kontroler bez politike |
| `scripts/list_envs.py` | Ispis svih registrovanih Isaac Lab env-ova |

## ⚠️ Pre nego što bilo šta commit-uješ

1. **Nikad `git add` nad `*.usd` fajlovima.** `.gitattributes` ih vodi kao LFS,
   ali su commit-ovani kao pravi blobovi — `git add` bi upisao LFS pointer čiji
   sadržaj ne postoji ni na jednom LFS serveru i asset bi bio izgubljen.
   Detalji i bezbedna procedura: [docs/07_poznati_problemi.md](docs/07_poznati_problemi.md).
2. `outputs/`, `datasets/` i `DIPLOMSKI/` su gitignorovani — veliki artefakti se
   čuvaju van GitHub-a ([docs/06](docs/06_artefakti_i_arhiva.md)).

## Tutorijali (docs/)

1. [Pokretanje okruženja (Docker + Isaac Lab)](docs/01_pokretanje_okruzenja.md)
2. [Instalacija zavisnosti (LeRobot, torchcodec, wandb)](docs/02_instalacija_zavisnosti.md)
3. [Prikupljanje demonstracija](docs/03_prikupljanje_demonstracija.md)
4. [Treniranje politika (ACT i DP)](docs/04_treniranje_politika.md)
5. [Evaluacija i analiza](docs/05_evaluacija_i_analiza.md)
6. [Artefakti i arhiva (checkpointi, dataset, snimci)](docs/06_artefakti_i_arhiva.md)
7. [Poznati problemi i zamke](docs/07_poznati_problemi.md)
