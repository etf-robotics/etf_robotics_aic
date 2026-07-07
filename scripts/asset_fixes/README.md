# Asset fix skripte (jednokratne popravke USD asseta)

Ove skripte su **već primenjene** na assete u repou — čuvamo ih kao recept, jer se
isti problemi ponavljaju svaki put kad se konektor konvertuje iz `.glb` u `.usd`
(Isaac Sim asset converter gubi materijale, normale i skalu).

Asseti na koje se odnose žive u:
`source/aic_task/aic_task/assets/robots/ur5e_cable/visuals/`
(`sfp_module_visual.usd`, `sc_plug_visual.usd`, `lc_plug_visual.usd` + njihovi `.glb` izvori).

| Skripta | Problem koji rešava |
|---|---|
| `fix_connector_normals.py` | `.glb`→`.usd` konverzija ostavi pokvaren `normals` primvar (faceVarying count mismatch) → Hydra renderuje mesh skoro crn. Skripta briše autorovane normale, pa se one računaju iz geometrije u render-time-u. |
| `fix_connector_scale.py` | Konvertovan USD je u centimetrima (metersPerUnit=0.01 ili bez skale) → konektor 100× veći. Dodaje scale op 0.01 i postavlja metersPerUnit=1.0. |
| `bind_sfp_material.py` | Konverzija spoji dva glTF primitiva u jedan Mesh i izgubi material binding → konektor taman. Rekreira dva `materialBind` GeomSubset-a preko face opsega (6684 + 184) i vezuje svaki za svoj materijal. |

## Pokretanje

Skripte importuju `pxr`, pa moraju kroz Isaac Sim python **unutar docker kontejnera**
(vidi [docs/01_pokretanje_okruzenja.md](../../docs/01_pokretanje_okruzenja.md)):

```bash
./isaaclab.sh -p etf_robotics_aic/scripts/asset_fixes/fix_connector_normals.py --usd putanja/do/asset.usd
```

> **Upozorenje:** skripte menjaju `.usd` fajl in-place (`RootLayer().Save()`).
> Napravi kopiju pre pokretanja. Nikad ne raditi `git add` nad `*.usd` fajlovima
> bez provere LFS podešavanja — vidi napomenu u glavnom README-u.
