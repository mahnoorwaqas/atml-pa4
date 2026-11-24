# ATML PA4 — Kaggle execution

This is the student's PA4 experiment repository. The source dataset repository is [AbDu11aHHH/atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets).

## Run the Kaggle integration

1. In **Settings → Secrets and variables → Actions**, add a repository secret named **`KAGGLE_API_TOKEN`**, containing the API token generated at https://www.kaggle.com/settings. Never share or commit it.
2. Go to **Actions → Run PA4 on Kaggle → Run workflow**, choose **`dataset-check`** first.
3. Review the Actions output and download its attached run artifact. The notebook is private and has a unique ID per run.
4. Upload the separately provided PA4 source package before choosing `attention-smoke` or `spec-toy`.

| Run option | Purpose | GPU |
|---|---|---|
| `dataset-check` | Validate the staff's bundled data, checksums and unit tests | No |
| `attention-smoke` | Smoke-run the local PA4 attention benchmark (not full assignment coverage) | Yes |
| `spec-toy` | Run a 100,000-draw sampler test from the PA4 code package | No |

**Not yet a completed assignment submission:** this repository initially contains just automation scaffolding. All Task 1/2/3 integrations, official dataset schema adapters, complete experimental matrix, and durable training checkpoint handling must be validated before final experiments. Kaggle sessions can expire; GitHub Actions artifacts have limited retention. No costly GPU task is automatically launched.

The PA4 course rules require that you understand and attribute the code and independently write the PDF report.
