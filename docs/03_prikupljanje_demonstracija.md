# 3. Prikupljanje demonstracija

Demonstracije generiše **skriptovani oracle** (nema teleoperacije): planer
napravi trofaznu TCP trajektoriju APPROACH → ALIGN → INSERT iz trenutne
opservacije, executor je izvrši, a u dataset se upišu **samo epizode koje su
završile `success` terminacijom** (failed_stationary i time_out se odbacuju).

Kod planera: [scripts/il/path_planners/port_insertion.py](../scripts/il/path_planners/port_insertion.py),
writer: [scripts/il/writer.py](../scripts/il/writer.py), config:
[scripts/il/config/oracle.yaml](../scripts/il/config/oracle.yaml).

## Pokretanje

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/collect_demos.py \
  --headless --enable_cameras --num_envs 16 --seed 0
```

- Dataset ide u `datasets/port_insertion/<NNN>_<timestamp>/` (LeRobot v2 format).
- `--num_envs` — envovi rade u lockstepu; resetovani env se re-planira dok
  ostali nastavljaju. 16 envova je razuman default na 5090.
- Ostali parametri (`--standoff_m`, ...): `--help`.

## Format dataseta

Referentni dataset teze je `datasets/port_insertion/001_20260612-132535`
(**30290 uspešnih epizoda**, fps 30):

| Ključ | Sadržaj |
|---|---|
| `observation.state` | 56-dim vektor (bez konstantnih kanala) |
| `action` | 6-dim (DiffIK TCP komanda) |
| `observation.images.{center,left,right}` | 3 kamere, 224×224, AV1 video (GOP~2) |
| `annotation.phase` | one-hot {APPROACH, ALIGN, INSERT} — koristi je pomoćna phase glava u treningu |

> **Napomena:** ovaj dataset je snimljen **bez vidljivog konektora** na vrhu
> kabla (slomljene `.glb` reference — vidi
> [07_poznati_problemi.md](07_poznati_problemi.md#konektori-se-ne-renderuju)).
> Politike su trenirane na golom vrhu; eval podrazumevano prikazuje konektor,
> a `--no_connector_retarget` daje izgled kao u treningu.

## Pregled epizoda (Rerun viewer)

```bash
docker exec -w /workspace/isaaclab isaac-lab-base \
  ./isaaclab.sh -p etf_robotics_aic/scripts/view_demos.py --episode 0
```

Modovi: `--mode save` (default, snimi .rrd fajl pa otvori na hostu),
`--mode local` / `--mode distant` (live server, `--grpc_port`).

Brzi sanity check bez viewera: `datasets/<run>/viz/` sadrži par renderovanih
epizoda, a `meta/info.json` broj epizoda i šemu.
