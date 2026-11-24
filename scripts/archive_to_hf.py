#!/usr/bin/env python3
"""Archive one completed Kaggle PA4 run to a private Hugging Face dataset repo.

HF_TOKEN stays on the GitHub Actions runner, never inside the Kaggle notebook.
Every uploaded artifact is namespaced by method and immutable GitHub run ID.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def find_completed_run(root: Path, experiment: str) -> tuple[Path, Path]:
    candidates = sorted(root.rglob("completed.json"))
    matches = []
    for marker in candidates:
        condition_dir = marker.parent
        if condition_dir.name != f"task1_{experiment}":
            continue
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("finished") is True:
            matches.append((condition_dir, marker))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one completed task1_{experiment} run in {root}; "
            f"found {len(matches)}. Incomplete runs must not be archived as complete."
        )
    condition_dir, marker = matches[0]
    required = [
        "config.json", "checkpoint.json", "environment.json", "inputs.json",
        "compression.json", "wiki_summary.json", "c4_summary.json",
        "hellaswag_summary.json", "hellaswag_examples.json",
        "wiki_windows.json", "c4_windows.json", "timing.json",
        "development_summary.json", "dev_windows.json", "protocol.json", "costs.json",
        "data_selection.json",
    ]
    missing = [item for item in required if not (condition_dir / item).is_file()]
    if missing:
        raise RuntimeError(f"Missing result files: {missing}")
    return condition_dir, marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument("--owner", default="mahnoorwaqas")
    args = parser.parse_args()

    if not args.experiment.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid experiment name")
    for value in (args.run_id, args.run_attempt):
        if not value.isdigit():
            raise ValueError("GitHub identifiers must be numeric")
    if len(args.github_sha) != 40:
        raise ValueError("Expected full git SHA")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Missing HF_TOKEN Actions secret")
    result_dir, marker = find_completed_run(args.download_dir, args.experiment)

    api = HfApi(token=token)
    repo_id = f"{args.owner}/atml-pa4-results"
    checkpoints_id = f"{args.owner}/atml-pa4-checkpoints"
    # Private by default on creation. Stop if a repository was accidentally made public.
    for repo, kind in ((repo_id, "dataset"), (checkpoints_id, "model")):
        api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=True)
        details = api.repo_info(repo_id=repo, repo_type=kind)
        if not details.private:
            raise RuntimeError(f"Refusing upload: {repo} is not private")

    prefix = (
        f"task1/{args.experiment}/github-run-"
        f"{args.run_id}-attempt-{args.run_attempt}"
    )
    provenance = {
        "experiment": args.experiment,
        "github_run_id": args.run_id,
        "github_run_attempt": args.run_attempt,
        "github_sha": args.github_sha,
        "kind": "Task 1 metrics; not a model checkpoint",
        "completed_marker": marker.name,
    }
    provenance_path = result_dir / "archive_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    output_root = args.download_dir / "pa4_outputs"
    for file in ("staff_checksum_audit.json", "source_revisions.json"):
        source = output_root / file
        if not source.is_file():
            raise RuntimeError(f"Missing required provenance: {source}")
        (result_dir / file).write_bytes(source.read_bytes())

    # Upload only one condition's measured outputs, not original model weights.
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset",
        folder_path=str(result_dir), path_in_repo=prefix,
        commit_message=f"Archive Task 1 {args.experiment} run {args.run_id}",
    )
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    expected = {f"{prefix}/{p.name}" for p in result_dir.iterdir() if p.is_file()}
    missing = expected - remote
    if missing:
        raise RuntimeError(f"Remote upload verification failed: {sorted(missing)}")
    print(json.dumps({"archive": f"https://huggingface.co/datasets/{repo_id}/tree/main/{prefix}",
                      "verified_files": len(expected), "checkpoints_repo": checkpoints_id}))


if __name__ == "__main__":
    main()
