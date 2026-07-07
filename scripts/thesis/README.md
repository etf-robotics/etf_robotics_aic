# Skripte za figure i statistiku diplomskog rada

Generišu figure (ćirilica, PDF) i statistiku za tezu iz kuriranih eval rezultata
u `outputs/eval/`. Izlaz ide direktno u `DIPLOMSKI/images/` i `DIPLOMSKI/analiza_eval/`
(DIPLOMSKI je zaseban git repo, ovde gitignorovan).

| Skripta | Šta pravi |
|---|---|
| `analiza_eval_compare.py` | Poređenje 4 runa (DP/ACT × {1000, 30290 epizoda}) — statistika + scaling figura (`rez_serija_a.pdf`, `statistika.txt`) |
| `teza_figure.py` | Per-run analitičke figure za dve full politike (tip→seat metrika, N=50) |
| `teza_montaza.py` | Montaža reprezentativnih stall epizoda ACT i DP (`rez_montaza_otkaza.pdf`) |
| `teza_compute.py` | Iskorišćenost GPU-a / VRAM / snaga iz wandb system stats-a |
| `multiseed_agg.py` | Agregacija full evala preko seedova {0,1,2} → mean ± std |

## Pokretanje

Ne treba im Isaac Sim (samo matplotlib/numpy/pandas), ali putanje očekuju
repo layout, pa ih pokretati iz root-a repoa:

```bash
python scripts/thesis/teza_figure.py
```

Skripte očekuju da `outputs/eval/` sadrži kurirane runove (`ACT_full`, `DP_full`,
`ACT_1000`, `ACT_10000`, `DP_1000ep`, `DP_10000`, `multiseed/`). Ako je `outputs/`
obrisan, prvo povuci arhivu eval rezultata — vidi [docs/06_artefakti.md](../../docs/06_artefakti.md).
