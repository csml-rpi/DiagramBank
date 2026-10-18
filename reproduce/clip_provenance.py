"""Explicit CLIP selection and per-run provenance for new classification runs.

Historical database labels are never assigned a checkpoint retrospectively.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import uuid

ARCHITECTURE = "ViT-B-32"
DEFAULT_PRETRAINED = "datacomp_xl_s13b_b90k"
PRETRAINED_CHOICES = (DEFAULT_PRETRAINED, "laion2b_s34b_b79k")
CLASS_ORDER = ("diagram", "plot", "photo", "other")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_clip_model(pretrained=DEFAULT_PRETRAINED, backend=None):
    if pretrained not in PRETRAINED_CHOICES:
        raise ValueError(f"Unsupported checkpoint: {pretrained}")
    if backend is None:
        import open_clip as backend
    try:
        model, _, preprocess = backend.create_model_and_transforms(
            ARCHITECTURE, pretrained=pretrained, device="cpu", precision="fp32", require_pretrained=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load explicitly selected {ARCHITECTURE} / {pretrained}; no alternate checkpoint was attempted.") from exc
    model._diagrambank_clip_checkpoint = {"architecture": ARCHITECTURE, "pretrained": pretrained}
    return model, backend.get_tokenizer(ARCHITECTURE), preprocess


def model_state_sha256(model):
    """Hash named tensors, dtypes, shapes, and bytes, independent of dict order."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def get_ensemble_features(model, tokenizer, class_prompts):
    import torch
    class_features = []
    with torch.no_grad():
        for label in CLASS_ORDER:
            features = model.encode_text(tokenizer(class_prompts[label]))
            features /= features.norm(dim=-1, keepdim=True)
            centroid = features.mean(dim=0)
            class_features.append(centroid / centroid.norm())
    return torch.stack(class_features)


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


class ClassificationRun:
    def __init__(self, database, batch_size, source, *, pretrained=DEFAULT_PRETRAINED, run_dir=None):
        if pretrained not in PRETRAINED_CHOICES:
            raise ValueError(f"Unsupported checkpoint: {pretrained}")
        self.database = Path(database).resolve()
        self.pretrained = pretrained
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        self.directory = Path(run_dir) if run_dir else self.database.parent / "classification-runs" / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.image_counts = Counter()
        self.input_keys = set()
        self.valid_image_keys = set()
        source = Path(source).resolve()
        sources = (source, source.with_name("utils.py"), Path(__file__).resolve())
        versions = {}
        for package in ("open-clip-torch", "torch", "torchvision", "Pillow", "duckdb", "numpy"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        self.info = {"schema_version": "1.0", "run_id": run_id, "started_at_utc": utc_now(),
                     "status": "started", "database": str(self.database),
                     "database_sha256_before": sha256_file(self.database),
                     "requested_model": {"architecture": ARCHITECTURE, "pretrained": pretrained},
                     "batch_size": batch_size, "device": "cpu", "precision": "fp32", "require_pretrained": True,
                     "class_order": list(CLASS_ORDER),
                     "scoring": "L2-normalized image embeddings; mean of L2-normalized prompt embeddings, then L2-normalized per class; softmax(100 * cosine_similarity)",
                     "package_versions": versions,
                     "source_sha256": {str(p.relative_to(Path(__file__).resolve().parent)): sha256_file(p) for p in sources}}
        self._save()
        print(f"CLIP checkpoint: {ARCHITECTURE} / {pretrained}")
        print(f"Classification run: {self.directory}")

    @classmethod
    def from_cli(cls, database, batch_size, source):
        parser = argparse.ArgumentParser(description="Classify unlabelled figures with an explicitly selected CLIP checkpoint.")
        parser.add_argument("--clip-pretrained", choices=PRETRAINED_CHOICES, default=DEFAULT_PRETRAINED)
        parser.add_argument("--run-dir", type=Path, help="New directory for run configuration, image hashes, and outputs; must not exist.")
        args = parser.parse_args()
        return cls(database, batch_size, source, pretrained=args.clip_pretrained, run_dir=args.run_dir)

    def _save(self):
        pending = self.directory / "manifest.json.tmp"
        pending.write_text(json.dumps(self.info, indent=2, allow_nan=False) + "\n")
        pending.replace(self.directory / "manifest.json")

    def failed(self, exc):
        self.info.update(status="failed", failed_at_utc=utc_now(), error=f"{type(exc).__name__}: {exc}")
        self._save()

    def record_model(self, model, preprocess, prompts):
        selected = getattr(model, "_diagrambank_clip_checkpoint", None)
        if selected != self.info["requested_model"]:
            raise ValueError("Loaded model identity does not match the requested checkpoint")
        self.info.update(status="model_loaded", loaded_model=selected,
                         model_state_sha256=model_state_sha256(model),
                         model_hash_definition="SHA-256 over sorted state_dict names, dtype/shape headers, and contiguous CPU tensor bytes",
                         preprocessing=str(preprocess), prompts=prompts)
        self._save()

    def record_candidates(self, candidates):
        keys = {(str(pid), str(number)) for pid, number, _ in candidates}
        if len(keys) != len(candidates):
            raise ValueError("Duplicate candidate keys")
        self.input_keys = keys
        write_jsonl(self.directory / "candidates.jsonl", ({"platform_id": str(pid), "figure_number": str(number), "figure_path": path} for pid, number, path in candidates))
        self.info.update(status="inference_started", candidate_records=len(candidates))
        self._save()

    def record_image(self, pid, number, path, *, error=None):
        identity = (str(pid), str(number))
        if identity not in self.input_keys:
            raise ValueError("Image key is absent from the candidate list")
        row = {"platform_id": identity[0], "figure_number": identity[1]}
        if error is None:
            row.update(status="readable", sha256=sha256_file(path))
            self.valid_image_keys.add(identity)
        else:
            row.update(status="skipped", error=error)
        self.image_counts[row["status"]] += 1
        with (self.directory / "image_inputs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def apply_updates(self, con, updates):
        """Save outputs first, then update all selected database rows atomically."""
        keys = {(str(pid), str(number)) for _, _, pid, number in updates}
        if keys != self.valid_image_keys or len(keys) != len(updates):
            raise ValueError("Output keys must match the successfully processed images exactly")
        write_jsonl(self.directory / "predictions.jsonl", ({"platform_id": str(pid), "figure_number": str(number),
                    "figure_type": label, "confidence": confidence} for label, confidence, pid, number in updates))
        self.info.update(status="outputs_saved", image_counts=dict(self.image_counts), output_records=len(updates))
        self._save()
        con.execute("BEGIN TRANSACTION")
        try:
            if updates:
                con.executemany("UPDATE Figures SET figure_type = ?, confidence = ? WHERE platform_id = ? AND figure_number = ?", updates)
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            self.failed(exc)
            raise
        self.info.update(status="database_updated", database_committed_at_utc=utc_now())
        self._save()

    def complete(self):
        if self.info["status"] != "database_updated":
            raise ValueError("Cannot mark a run complete before the database update commits")
        self.info.update(status="completed", completed_at_utc=utc_now(), database_sha256_after=sha256_file(self.database),
                         outputs={p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                                  for p in sorted(self.directory.glob("*.jsonl"))})
        self._save()
