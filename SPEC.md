# SPEC — New-user onboarding overhaul

**Goal:** a fresh GitHub clone runs end-to-end in under 5 minutes on Python 3.11+, with zero external downloads, on macOS or Linux. Works equally well for a user driving via Claude Code or Codex.

**Non-goal:** changing any project logic. This spec touches only docs, scripts, repo hygiene, and CI. No source under `src/autoresearch/`, no configs, no tests.

---

## 0. Hard constraints (apply to every section)

- **Do not modify:** `src/autoresearch/**`, `configs/**`, `tests/**`, `AGENT.md`, anything under `artifacts/` or `data/` except where explicitly listed below.
- **Do not duplicate content.** If something is moved from README into `docs/CLI.md`, it must be deleted from README.
- **Every command in any new doc must be verified runnable** against the current CLI (see §A for the authoritative command list).
- **All Mermaid diagrams must be fenced with ` ```mermaid ` and tested to render on GitHub.** Keep them small — under ~20 nodes each.
- **No emojis** in any new file unless explicitly listed in this spec.
- **No invented features.** Every claim in any new doc must correspond to behavior that exists in `src/autoresearch/` today.

---

## 1. README rewrite (`README.md`) — replace entirely

Target length: 140–170 lines. Sections in this exact order:

1. **Title + tagline** (one sentence): "Autonomous insurance target-modelling research loop on freMTPL2, driven by an LLM agent (Claude Code or Codex). Burning cost is the default target, with frequency runs available by explicit configuration."
2. **Badges row**: CI status (GitHub Actions), Python 3.11+, License MIT. Use shields.io badge URLs.
3. **Hero diagram (Mermaid)** — system flow. Required nodes:
   - `data/raw/` → `prepare-data` → `data/processed/agent_dataset_search.parquet` and `data/holdout_vault/agent_dataset_holdout.parquet`
   - `agent_dataset_search.parquet` → `experiment_runner` → `artifacts/tracks/<track>/runs/<run-id>/registry.sqlite`
   - `experiment_runner` → `promotion_gate` → `champion`
   - `holdout_vault` → (token-gated) → `milestone evaluation` (dashed edge)
4. **What this is** (3 bullets) / **What this is not** (3 bullets). "Not": a production pricing system, a hosted service, a benchmark of one specific model.
5. **Requirements**: Python 3.11+, ~2 GB free disk, macOS/Linux (Windows via WSL).
6. **Quickstart** — single fenced bash block, must run top-to-bottom on a clean clone:
   ```bash
   git clone <repo-url> && cd Insurance_AutoResearch
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   python scripts/generate_synthetic_data.py
   autoresearch --track demo --run-id quickstart bootstrap-track
   autoresearch --track demo --run-id quickstart run-session-cycles 1
   ```
   Followed by one sentence: "You should see a comparison report under `artifacts/tracks/demo/runs/quickstart/iterations/`."
7. **Run with an agent** — two short subsections, each ~6 lines, linking to:
   - `docs/RUN_WITH_CLAUDE_CODE.md`
   - `docs/RUN_WITH_CODEX.md`
8. **Agent loop diagram (Mermaid)** — required nodes: `handoff` → `agent writes proposal + script` → `proposal inbox` → `ingest` → `run experiment` → `compare to champion` → `promotion gate` → (pass: `new champion`, fail: `rejected`) → loop back to `handoff`.
9. **Repo layout** — fenced text tree showing the top-level directories and the `artifacts/tracks/<track>/runs/<run-id>/` substructure (copy from AGENT.md lines 353–369).
10. **Working with real freMTPL2 data** — 3 lines: run `python scripts/fetch_fremtpl2.py`, then `autoresearch prepare-data`. Note licensing in `data/raw/README.md`.
11. **Where to go next** — bulleted links to:
    - `docs/CLI.md` (full command reference)
    - `docs/architecture.md`
    - `AGENT.md` (operating manual the agent reads)
    - `CONTRIBUTING.md`
12. **License** — one line: "MIT. See `LICENSE`."

**Forbidden in the new README:** any command tables, promotion gate threshold tables, dataset schema tables, list of artifact filenames per experiment, list of file-handoff workflow paths. All of that belongs in `docs/CLI.md` or stays in `AGENT.md`.

---

## 2. Synthetic sample data generator (`scripts/generate_synthetic_data.py`)

**Purpose:** make `autoresearch prepare-data` succeed on a clean clone with zero external downloads.

**Output:** two parquet files in `data/raw/`:
- `freMTPL2freq_synthetic.parquet`
- `freMTPL2sev_synthetic.parquet`

(Filename must contain `freq` and `sev` respectively — the loader at `src/autoresearch/data/loader.py:50` matches on those substrings.)

**Frequency table — exact columns and dtypes** (from `SEMANTIC_NAME_MAP` in `src/autoresearch/data/anonymise.py:11`):
| Column | Dtype | Notes |
|---|---|---|
| `IDpol` | int64 | unique, 1..N |
| `ClaimNb` | int64 | 0–4, Poisson with mean ~0.05 |
| `Exposure` | float64 | uniform(0.05, 1.0) |
| `VehPower` | int64 | uniform int 4..15 |
| `VehAge` | int64 | uniform int 0..20 |
| `DrivAge` | int64 | uniform int 18..90 |
| `BonusMalus` | int64 | uniform int 50..150 |
| `VehBrand` | str | random from `["B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"]` |
| `VehGas` | str | random from `["Regular","Diesel"]` |
| `Area` | str | random from `["A","B","C","D","E","F"]` |
| `Density` | int64 | log-uniform 1..30000 |
| `Region` | str | random from 21 strings `"R11","R21",...` (any 21 distinct values) |

**Severity table — exact columns:**
| Column | Dtype | Notes |
|---|---|---|
| `IDpol` | int64 | one row per claim; references rows in freq where `ClaimNb > 0` |
| `ClaimAmount` | float64 | lognormal, mean ~2000, occasional large values up to ~50000 |

**Volume:** 5,000 frequency rows. Severity rows = sum of `ClaimNb` across the freq table.

**Determinism:** `numpy.random.default_rng(seed=20260526)`. No other randomness sources.

**CLI behavior:**
- Print "Writing data/raw/freMTPL2freq_synthetic.parquet (5000 rows)..." and "Writing data/raw/freMTPL2sev_synthetic.parquet (<N> rows)..."
- Exit 0 on success.
- Refuse to overwrite if files exist unless `--force` is passed; print a helpful message pointing to `--force`.
- Use `argparse`. No third-party dependencies beyond `numpy`, `pandas`, `pyarrow` (all already in `pyproject.toml`).

**Validation requirement:** after the script runs, `autoresearch prepare-data` must succeed unmodified. If it fails, fix the synthetic generator — do not modify the loader.

---

## 3. Real-data fetch script (`scripts/fetch_fremtpl2.py`)

**Purpose:** download real freMTPL2 from OpenML for users who want to run on the actual ~678K-policy dataset.

**Behavior:**
- Fetch OpenML dataset id `41214` (freMTPL2freq) and `41215` (freMTPL2sev) using `sklearn.datasets.fetch_openml(data_id=..., as_frame=True)`. `scikit-learn` is already a dependency.
- Write `data/raw/freMTPL2freq.parquet` and `data/raw/freMTPL2sev.parquet`.
- Print progress: "Fetching freMTPL2freq from OpenML (id 41214)...", row counts, final paths.
- `--force` flag to overwrite. Refuse otherwise.
- If `openml.org` is unreachable, fail with a clear message and a link to `https://www.openml.org/d/41214` for manual download.

**No new dependencies.** Use only what's in `pyproject.toml`.

---

## 4. Data README (`data/raw/README.md`)

~25 lines. Sections:
1. **What goes here**: freMTPL2 frequency and severity files (CSV or parquet). The loader auto-discovers by filename substring `freq` and `sev`.
2. **Two ways to populate this directory:**
   - For testing / smoke runs: `python scripts/generate_synthetic_data.py` (5000 synthetic rows, no network).
   - For real research: `python scripts/fetch_fremtpl2.py` (downloads ~678K rows from OpenML).
3. **Licensing note**: freMTPL2 is distributed under the CASdatasets / OpenML terms; cite Charpentier (2014) for academic use. Synthetic data has no license restriction.
4. **Expected columns** — reference list pointing to `src/autoresearch/data/anonymise.py` for the authoritative mapping.

---

## 5. Agent on-ramps

### 5a. `docs/RUN_WITH_CLAUDE_CODE.md` (~50 lines)

Structure:
1. **Prereqs**: Claude Code installed, repo cloned, quickstart from README completed.
2. **One-time setup** — open the repo in Claude Code (`cd <repo> && claude`).
3. **The first prompt** — copy-paste block:
   ```
   Read AGENT.md, then bootstrap a new run under track "claude" with
   run-id of your choice and run 3 cycles. Use synthetic data — I have
   already run scripts/generate_synthetic_data.py.
   ```
4. **What happens** — 5 bullets describing: bootstrap, baseline runs, champion init, proposal generation, comparison, promotion or rejection.
5. **Where to look afterward**: `artifacts/tracks/claude/runs/<run-id>/RESEARCH_LOG.md`, the latest comparison report HTML.
6. **Common follow-up prompts**: "Continue", "Run 5 more cycles", "Try a GLM next".
7. **Troubleshooting**: pytest failures, integrity manifest changes, holdout token errors.

### 5b. `docs/RUN_WITH_CODEX.md` (~50 lines)

Mirror structure of 5a, but:
- Reference `.codex/config.toml` (already in repo, model = "gpt-5.5", sandbox_mode = "workspace-write", network_access = true).
- First-prompt example uses track `"codex"`.
- Note: Codex sandbox needs network access for the OpenML fetch; the existing config already allows this.

Both files end with the same recommended command pair:
```bash
autoresearch --track <agent> --run-id <name> bootstrap-track
autoresearch --track <agent> --run-id <name> run-session-cycles 3
```

---

## 6. CLI reference (`docs/CLI.md`)

New file. Move every CLI example currently in `README.md` into here. Group commands under these H2 sections, in this order:

1. **Data preparation** — `prepare-data`
2. **Registry & bootstrap** — `init-registry`, `bootstrap-track`, `list-tracks`
3. **Baselines** — `run-baseline`, `run-all-baselines`, `list-experiments`, `init-official-champion`
4. **Comparison & promotion** — `run-repeated-evaluation`, `compare-experiments`, `compare-to-champion`, `list-promotions`, `list-champion-history`, `list-branches`
5. **File-handoff workflow** — `export-context`, `write-proposal-template`, `show-latest-handoff`, `show-proposal-inbox-status`, `ingest-proposals`, `enqueue-proposal`, `run-next-proposal`, `run-latest-proposal-cycle`, `list-proposals`, `inspect-proposal`
6. **Supervised sessions** — `start-session`, `session-status`, `pause-session`, `resume-session`, `stop-session`, `run-session-cycle`, `run-session-cycles`
7. **Milestone / integrity** — `evaluate-milestone`, `update-integrity-manifest`
8. **Cross-track (human-only)** — `compare-tracks`

Each command gets: one-line description, example invocation (with `--track demo --run-id quickstart` where applicable), and what it writes. Cross-reference the authoritative list in §A below.

---

## 7. Repo hygiene

### 7a. `LICENSE`
Standard MIT license text. Copyright line: `Copyright (c) 2026 Alex Hood`.

### 7b. `CONTRIBUTING.md` (~40 lines)
Sections:
1. **Local dev setup** — pointer to README quickstart.
2. **Tests** — `pytest` must pass. Add a smoke test for any new model family or feature builder.
3. **Protected files** — list the four files from AGENT.md §"Safety rules" #2. If you intentionally change one, run `autoresearch update-integrity-manifest` and explain in the PR description.
4. **Adding a model family** — pointer to `src/autoresearch/models/` and the `fit_predict` signature.
5. **Adding a feature builder** — pointer to `src/autoresearch/features/` and the `build_features` signature.
6. **PRs** — keep diffs focused; one logical change per PR.

### 7c. `.github/workflows/ci.yml`
- Trigger: `push` to any branch and `pull_request`.
- Matrix: `python-version: ["3.11", "3.12"]`, `os: ubuntu-latest`.
- Steps:
  1. `actions/checkout@v4`
  2. `actions/setup-python@v5` with cache: pip
  3. `pip install -e ".[dev]"`
  4. `python scripts/generate_synthetic_data.py`
  5. `autoresearch prepare-data`
  6. `pytest --tb=short -q`
- No deploy step. No secrets required.

### 7d. `.gitignore` update
Keep all existing lines. Append:
```
!data/.gitkeep
!data/raw/.gitkeep
!artifacts/.gitkeep
```
The existing `data/*` and `artifacts/*` rules will still exclude everything else.

### 7e. `.gitkeep` files
Create empty files at:
- `data/.gitkeep`
- `data/raw/.gitkeep`
- `artifacts/.gitkeep`

### 7f. Move internal docs
Create `docs/internal/` and `git mv` these files into it:
- `docs/refactor_spec.md` → `docs/internal/refactor_spec.md`
- `docs/codex_build_instructions.md` → `docs/internal/codex_build_instructions.md`
- `docs/auto_research_plan.md` → `docs/internal/auto_research_plan.md`

Do not move: `docs/architecture.md`, `docs/experiment_protocol.md`, `docs/requirements.md`, `docs/autonomous_session_workflow.md`, `docs/RESEARCH_LOG.md`.

---

## 8. Validation gate (must pass before reporting done)

Run these in order from the repo root in a clean venv. Every one must exit 0 / succeed.

1. `pip install -e ".[dev]"`
2. `python scripts/generate_synthetic_data.py` — writes both parquet files.
3. `autoresearch prepare-data` — produces `data/processed/agent_dataset_search.parquet`.
4. `autoresearch --track demo --run-id quickstart bootstrap-track` — exits 0, prints handoff path.
5. `autoresearch --track demo --run-id quickstart run-session-cycles 1` — exits 0.
6. `pytest --tb=short -q` — all tests pass.
7. Open `README.md` and confirm: exactly two ` ```mermaid ` fenced blocks, no command tables, length 140–170 lines.
8. `ls docs/internal/` shows the three moved files; `ls docs/` no longer shows them.
9. `ls data/.gitkeep data/raw/.gitkeep artifacts/.gitkeep data/raw/README.md LICENSE CONTRIBUTING.md .github/workflows/ci.yml scripts/generate_synthetic_data.py scripts/fetch_fremtpl2.py docs/CLI.md docs/RUN_WITH_CLAUDE_CODE.md docs/RUN_WITH_CODEX.md` — all exist.

If any step fails, fix the cause in the new files. Never weaken the spec, never modify out-of-scope source.

---

## Appendix A — Authoritative CLI command list

These are the only commands that may appear in new docs. Source: `src/autoresearch/cli.py`. Do not invent variations.

```
prepare-data
bootstrap-track
init-registry
run-baseline
run-all-baselines
list-experiments
run-repeated-evaluation
compare-experiments
compare-to-champion
list-promotions
init-official-champion
enqueue-proposal
run-next-proposal
list-proposals
list-champion-history
list-branches
inspect-proposal
evaluate-milestone
update-integrity-manifest
export-context
write-proposal-template
ingest-proposals
run-latest-proposal-cycle
show-latest-handoff
show-proposal-inbox-status
start-session
session-status
pause-session
resume-session
stop-session
run-session-cycle
run-session-cycles
compare-tracks
list-tracks
```

All commands accept `--track <name>` and `--run-id <id>` as global flags before the subcommand name.

---

## Appendix B — File manifest

**New files (12):**
- `scripts/generate_synthetic_data.py`
- `scripts/fetch_fremtpl2.py`
- `data/raw/README.md`
- `data/.gitkeep`
- `data/raw/.gitkeep`
- `artifacts/.gitkeep`
- `LICENSE`
- `CONTRIBUTING.md`
- `.github/workflows/ci.yml`
- `docs/CLI.md`
- `docs/RUN_WITH_CLAUDE_CODE.md`
- `docs/RUN_WITH_CODEX.md`

**Modified files (2):**
- `README.md` (full rewrite)
- `.gitignore` (append `.gitkeep` allowlist)

**Moved files (3):**
- `docs/refactor_spec.md` → `docs/internal/refactor_spec.md`
- `docs/codex_build_instructions.md` → `docs/internal/codex_build_instructions.md`
- `docs/auto_research_plan.md` → `docs/internal/auto_research_plan.md`

**Deleted files:** none.

**Untouched (verify by `git status` — these must NOT appear as changed):** everything under `src/`, `configs/`, `tests/`, `AGENT.md`.
