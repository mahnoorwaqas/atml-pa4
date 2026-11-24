# ATML PA4 — reproducible Kaggle experiments

This is the student's **assignment code repository**. The staff's data repository is [AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets).

## Current integration

- **Task 1 implementation:** [task1/run.py](task1/run.py) loads the original **Qwen/Qwen2.5-1.5B base** weights for each condition and evaluates the staff's fixed C4, WikiText-2 and HellaSwag selections.
- **Dataset schemas:** Staff-prepared `data/public_locked/*.jsonl.gz`; generated with `python scripts/setup_data.py --public`. The immutable Hugging Face upstream revision IDs are saved in each run's `source_revisions.json`.
- **Execution:** [GitHub Actions](.github/workflows/kaggle-pa4.yml) submits a **private Kaggle GPU notebook** and retrieves its outputs. [Full matrix](.github/workflows/task1-matrix.yml) starts all 18 Task 1 variants sequentially only after the initial baseline action #45340451212 succeeds; the workflow is also manually dispatchable.
- **Validation:** [Task 1 CI](.github/workflows/task1-ci.yml) compiles the code and parses all generated notebooks before GPU execution.

## Auth

Set GitHub Actions secret **`KAGGLE_API_TOKEN`** in Settings → Secrets and variables → Actions. The token must never be pasted into the repository, a notebook, or a chat. Kaggle username: `mahnoorwaqas21`.

## Task 1 experiment keys

`baseline`; `mag25`, `mag50`, `mag75`; `wanda25`, `wanda50`, `wanda75`; `adaptive50`, `adaptive75`; `structured50`; `int8`, `int4`, `int3`; `awq8`, `awq4`, `awq3`; `mixed4`, `mixed3`.

Each condition starts with independent pretrained weights, uses a seed of 6304, saves its input SHA-256 digests, per-window negative log likelihood, per-item HellaSwag scores, condition configuration, and estimated storage/latency observations. The full HellaSwag 2,000 examples are reserved for the specified baseline/50% pruning/INT4/INT3 conditions. Other conditions use the first 512 staff IDs for diagnostics. Both perplexity suites score exactly 50,000 fixed target tokens each.

**Performance caveat:** Weights are restored as dense FP16 for numerical evaluation. Packed low-bit and sparse bytes are **estimates**, not evidence of GPU low-bit acceleration. The model checkpoint weights are not committed to Git.

## Run locally with prepared staff data

```bash
python task1/run.py --data /path/to/atml-pa4-datasets \
  --method baseline --output /path/to/task1_outputs/baseline
python task1/run.py --data /path/to/atml-pa4-datasets \
  --method wanda --sparsity 0.5 --output /path/to/task1_outputs/wanda50
```

For full method assignments see [the workflow generator](scripts/make_kaggle_kernel.py).

## Critical limitations and ethics

The initial automated baseline and 18-condition matrix are an **attempt to run the Task 1 experimental grid**, not proof that the full assignment protocol has been satisfied. The implementation still needs validation of layerwise output-error probes, uncertainty/paired bootstrap statistics, source-revision consistency across runs, explicit training/calibration and GPU decode accounting, exact low-bit serialized representations, and reproducibility of held-out comparison metrics. If Kaggle runs hit session limits, GPU quotas, or fail on a specific condition, preserve and report them. Do not infer compression speedups from fake quantization.

**Course AI policy:** Material coding assistance was provided by an LLM. The student is responsible for understanding and attributing reused code. The PDF report must be written independently by the student and may not be AI-written.

## Persistent Hugging Face storage

The `HF_TOKEN` GitHub Actions repository secret is used **only from GitHub Actions**.
The [Hugging Face setup workflow](.github/workflows/hf-setup.yml) has authenticated and
created/verified both **private** repositories:

- [`mahnoorwaqas/atml-pa4-results`](https://huggingface.co/datasets/mahnoorwaqas/atml-pa4-results) (dataset)
- [`mahnoorwaqas/atml-pa4-checkpoints`](https://huggingface.co/mahnoorwaqas/atml-pa4-checkpoints) (model)

On successful Kaggle execution, the GitHub Actions job downloads notebook outputs and
runs [`scripts/archive_to_hf.py`](scripts/archive_to_hf.py). Uploads are organized under
`task1/<condition>/github-run-<id>-attempt-<attempt>/`; a remote file listing is
checked before the archive step succeeds. Missing/incomplete outputs make the archive
step fail explicitly, even when notebook submission itself succeeded.

The original baseline run `45340451212` predates this integration and is automatically
backfilled by [`archive-initial-baseline.yml`](.github/workflows/archive-initial-baseline.yml)
**only if that exact run succeeds**. The full 18-condition matrix remains gated on its
successful completion.

**Checkpoints:** Task 1 currently archives result files and metadata, **not complete
compressed weights or resumable optimization checkpoints**. The private checkpoint
repository is prepared for later Tasks 2/3, for which actual model/optimizer/scheduler
state exports must be added and tested. Kaggle temporary files are not durable storage.
