# data/raw/ — moved

Raw data no longer lives here. Every dataset now has its own directory under
`data/datasets/<name>/raw/` (see `configs/datasets/*.toml` and the Datasets
chapter of `docs/OPERATING_MANUAL.md`):

- French Motor (freMTPL2): `data/datasets/french_motor/raw/`
- AllState (sample + full share one raw dir): `data/datasets/allstate/raw/`
- Porto Seguro: `data/datasets/porto_seguro/raw/`

To populate the French raw dir:

```bash
python scripts/generate_synthetic_data.py   # 5,000-row deterministic CI fixture
python scripts/fetch_fremtpl2.py            # real ~678K rows from OpenML
```

Both write into `data/datasets/french_motor/raw/`. The loader auto-discovers
files by filename substring (`freq` / `sev`) and deprioritises anything whose
name contains `synthetic`, so fixtures never shadow real data.
