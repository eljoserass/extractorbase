# Installation

This guide sets up the local Python harness, the dataset and, optionally, LM Studio.
For stages, flags and sample selection, use the [CLI guide](cli.md).
The multiline shell examples use Bash on Linux/macOS. The current end-to-end
verification was performed on Linux with CPU BERT and a local LM Studio endpoint.

## 1. Get the repository

With Git installed:

```bash
git clone https://github.com/eljoserass/extractorbase.git
cd extractorbase
```

If the repository requires authentication, use your GitHub credentials, or clone
through SSH when your SSH key is configured:

```bash
git clone git@github.com:eljoserass/extractorbase.git
cd extractorbase
```

If you already have the checkout, start from its root directory, which contains
`pyproject.toml`, `uv.lock` and `cli.py`.

## 2. Install uv

On Linux/macOS, use uv's standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS, Homebrew is another option:

```bash
brew install uv
```

Choose one installation method. Open a new terminal if needed, then check:

```bash
uv --version
```

See the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for Windows and other installation methods.

## 3. Install Python and the project dependencies

From the repository root, choose the dependencies you need:

```bash
# Dummy and LLM methods
uv sync --locked

# Include the BERT method
uv sync --locked --extra bert
```

The repository's `.python-version` selects Python 3.12. uv creates `.venv/` and
can download the required Python when it is missing. To install it explicitly:

```bash
uv python install 3.12
```

`--locked` uses the committed dependency versions and fails if the lockfile needs
updating. The `bert` extra adds PyTorch, Transformers and Accelerate. The current
BERT implementation runs on CPU, and the project uses PyTorch's CPU wheel index.

Development tools (pytest, Ruff and mypy) are included by default. Running plain
`uv sync --locked` after a BERT installation can remove the optional BERT packages;
keep `--extra bert` when syncing an environment that should support BERT.
See uv's [syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/)
for extras and lockfile behavior.

Use `uv run` for project commands; activating `.venv` manually is unnecessary:

```bash
uv run python cli.py --help
uv pip check

# When using BERT
uv run --extra bert python cli.py --help
```

The first setup needs internet access to download Python and libraries. BERT's
first fit also downloads its checkpoint. The optional BERT dependencies and model
weights take substantially more disk space than the dummy/LLM Python setup.

## 4. Add the Label Studio exports

The dataset is local and is not included in Git. Create the directory and copy
your exports into it:

```bash
mkdir -p data
```

Expected files for the documented examples:

```text
data/
├── grupo1.json
└── grupo2.json
```

Each file should be a Label Studio JSON export containing an array of tasks with
`id`, `data.text` and, for fitting/evaluation, `annotations`. The supplied schema
already exists at `schemas/reumalago.xml`.

Check loading and gold selection:

```bash
uv run python cli.py --stage inspect \
  --data data/grupo1.json --data data/grupo2.json \
  --output runs/inspection
```

Inspection prints counts, annotation provenance and data issues. Dataset warnings
are separate from installation errors. Fitting rejects selected examples with
structural annotation errors. `data/` and generated `runs/` files are ignored by Git.

## 5. Optional: configure LM Studio for Gemma

Dummy and BERT work without LM Studio. For the LLM method:

1. Install [LM Studio](https://lmstudio.ai/download).
2. Download and load [Google Gemma 4 E2B](https://lmstudio.ai/models/google/gemma-4-e2b).
3. Start the local API server. In LM Studio's Developer tab, use **Start server**.
4. Confirm the endpoint and the model identifier exposed by the server.

These server controls are described in LM Studio's
[local server documentation](https://lmstudio.ai/docs/developer/core/server).

With the default local port, check the API:

```bash
curl --max-time 10 http://localhost:1234/v1/models
```

The response should list the model you want to use. The harness defaults are:

| Setting | Default |
| --- | --- |
| Endpoint | `http://localhost:1234/v1` |
| Model ID | `google/gemma-4-e2b` |
| Thinking | Disabled per request |

If your server uses another address or model identifier, pass `--endpoint` and
`--model` when fitting. Those values are stored in the artifact and restored for
prediction. LM Studio stays running while the LLM method performs inference.
The current adapter assumes the local server does not require API-token
authentication; it does not expose a configurable token flag.

## 6. Check the complete pipeline

Use dummy to check fitting, persistence, prediction and evaluation without a
model download or an API server. Choose an empty or new artifact directory:

```bash
uv run python cli.py --stage run --method dummy \
  --data data/grupo1.json --offset 0 --count 5 --predict-count 5 \
  --dump runs/install_check/artifact --output runs/install_check/evaluation \
  --yes
```

This fits the first five documents and predicts the next five. `--yes` approves
this scripted smoke test; omit it to exercise the interactive confirmation.
Dummy's empty predictions give F1 zero, as expected. Results appear in the
terminal and in the output directory.

Once that works, follow the [CLI examples](cli.md#examples) for BERT or Gemma.

## Development checks

With the BERT extra installed, run the full checks:

```bash
uv run --extra bert pytest -q
uv run ruff check .
uv run --extra bert mypy data.py evaluator.py cli.py methods
```

The tests use synthetic data, fake LLM inference and a tiny local BERT checkpoint.
They do not require LM Studio or downloading a pretrained model. Without the BERT
extra, BERT-specific tests are skipped.

## Updating an existing checkout

When your local changes are ready for a Git update:

```bash
git pull
uv sync --locked --extra bert
```

Use `uv sync --locked` instead if you only need dummy/LLM. Keep the committed
lockfile to reproduce dependency versions. Python libraries are installed inside
the repository's `.venv/`; model artifacts and source exports remain separate.

## Setup problems

| Problem | Check |
| --- | --- |
| `uv: command not found` | Reopen the terminal or apply the PATH instructions printed by the installer. |
| GitHub clone authentication fails | Use configured GitHub credentials or the SSH clone URL. |
| Lockfile is reported outdated | Check that `pyproject.toml` and `uv.lock` come from the same repository revision. |
| Missing `torch` or `transformers` | Install/run with `--extra bert`. |
| Dataset path does not exist | Copy the exports into `data/` and run commands from the repository root. |
| LM Studio cannot be reached | Start the API server and check its address with `/v1/models`. |
| Requested model is not available | Load it in LM Studio and use its exposed model ID. |
| Artifact directory is not empty | Pick a new path for a new fit, or load the existing artifact for prediction. |
| Help/results are blank but errors print | Follow the [stdout troubleshooting steps](cli.md#troubleshooting). |
