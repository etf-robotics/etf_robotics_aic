# 7. Poznati problemi i zamke

## ⚠️ `*.usd` fajlovi i git — NAJVAŽNIJE

`.gitattributes` vodi `*.usd` kroz `filter=lfs`, ali su fajlovi istorijski
commit-ovani kao **pravi blobovi, ne LFS pointeri**. Posledice:

- `git status` trajno prikazuje fantomski ` M` na USD fajlovima; `git diff` na
  njima laže. To je normalno — ignoriši.
- **Nikad `git add` nad `*.usd`** — clean filter bi u indeks upisao LFS pointer,
  a pravi sadržaj ne postoji ni na jednom LFS serveru → commit bi izgubio asset.
- Ako baš moraš da stage-uješ USD, koristi `git checkout <ref> -- <putanja>`
  (postavlja indeks na pravi blob), pa commit.

## Konektori se ne renderuju

Kabl USD (`source/aic_task/.../ur5e_cable/aic_unified_robot_cable_sdf.usd`)
referencira vizuale konektora kao **`.glb`**, a ovaj Isaac Sim **nema glTF
plugin** → `Could not open asset ...glb` i vrh kabla je go. Zato je ceo dataset
`001_...` snimljen bez konektora (politike su na to i trenirane).

Popravljene `.usd` konverzije postoje u `visuals/*.usd` (skalirane 0.01 — original
je bio u cm pa su konektori bili 6-metarski džinovi; sfp dodatno: vraćene
originalne normale + GeomSubset material bindinzi). `eval_demos.py` ih učitava
**in-memory retargetom** (`_retarget_connectors_to_usd()`, plus brisanje delimičnog
Body_005 mesha koji je zapečen u kabl USD-u) — disk ostaje netaknut.

- Recepti za popravke: [scripts/asset_fixes/](../scripts/asset_fixes/README.md)
- Snimanje NOVOG dataseta i dalje ide preko `.glb` referenci → bez konektora,
  osim ako se isti retarget doda u collect_demos putanju.

## Crn / taman render

Dva odvojena uzroka, oba viđena:

1. **Korumpirane normale u modifikovanom robot USD-u** (blob 17003126 B):
   `corrupted data in primvar 'normal'` → cela scena crna bez obzira na svetlo.
   Rešeno revertom na origin/main verziju (commit `5b764b1`). Ako ikad
   regenerišeš robot USD, proveri normale pre commita.
2. **`randomize_dome_light` event** re-seeduje kupolu na svakom resetu → poneka
   eval epizoda ispadne mračna. Jednokratni `DomeLight.SetIntensity` ne pomaže
   (Fabric ga pregazi svaki frame) — za konzistentno svetao render forsiraj
   intenzitet **na svakom koraku**.

## Ostalo

- **Rekreacija kontejnera briše pip/apt instalacije** — ponovi
  [02_instalacija_zavisnosti.md](02_instalacija_zavisnosti.md). wandb login i
  `shm_size` preživljavaju (compose + `.netrc` mount).
- **torchcodec mora tačno da prati torch** (0.4↔2.7, 0.5↔2.8...) — pogrešna
  verzija pada uz ABI grešku; bundled verzija u image-u je pogrešna.
- `training_status.py` puca ako `checkpoints/` još ne postoji (pre prvog
  `--save_freq` koraka).
- Stari wandb projekat `port-insertion-act` (run `zmie4c5w`) je sa **drugog
  dataseta** (`003_...`, ~11k ep.) i nedotreniran (0.82 epohe) — nije uporediva
  tačka ni za šta.
- `datasets/_testrun/` je smoke-test artefakt writera, nije pravi dataset.
