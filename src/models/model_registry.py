"""
models/model_registry.py
─────────────────────────
Lightweight model versioning without MLflow overhead.
In production, swap this for MLflow, W&B, or SageMaker Model Registry.

Version scheme: v{MAJOR}.{MINOR}.{PATCH}
  - MAJOR: breaking schema change
  - MINOR: new features / retraining
  - PATCH: threshold or config change only
"""

import joblib
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import REGISTRY_DIR, ARTIFACTS_DIR, CHAMPION_MODEL_NAME, PIPELINE_NAME, METADATA_NAME

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages model versioning, promotion, and rollback.

    Directory structure:
        models/registry/
            v1.0.0/
                model.joblib
                preprocessing_pipeline.joblib
                metadata.json
            v1.1.0/
                ...
            champion -> symlink to latest promoted version
    """

    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry_index.json"

        if not self.metadata_file.exists():
            self._write_index({"versions": [], "champion": None})

    def _read_index(self) -> dict:
        with open(self.metadata_file) as f:
            return json.load(f)

    def _write_index(self, data: dict):
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _next_version(self) -> str:
        index = self._read_index()
        if not index["versions"]:
            return "v1.0.0"
        latest = index["versions"][-1]["version"]
        parts = latest.lstrip("v").split(".")
        parts[1] = str(int(parts[1]) + 1)  # bump minor
        parts[2] = "0"
        return "v" + ".".join(parts)

    def register_model(
        self,
        pipeline,
        preprocessor_winsorizer,
        metrics: Dict[str, float],
        model_name: str,
        threshold: float,
        feature_names: list,
        notes: str = "",
        version: Optional[str] = None,
    ) -> str:
        """
        Save a new model version to the registry.

        Returns the version string (e.g., "v1.1.0")
        """
        version = version or self._next_version()
        version_dir = self.registry_dir / version
        version_dir.mkdir(exist_ok=True)

        # Serialize model pipeline
        joblib.dump(pipeline, version_dir / "model.joblib")
        joblib.dump(preprocessor_winsorizer, version_dir / "winsorizer.joblib")

        # Save metadata
        metadata = {
            "version": version,
            "model_name": model_name,
            "registered_at": datetime.utcnow().isoformat(),
            "threshold": threshold,
            "metrics": metrics,
            "feature_names": feature_names,
            "notes": notes,
            "status": "candidate",  # candidate → champion → retired
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update index
        index = self._read_index()
        index["versions"].append({
            "version": version,
            "model_name": model_name,
            "registered_at": metadata["registered_at"],
            "roc_auc": metrics.get("roc_auc", None),
            "status": "candidate",
        })
        self._write_index(index)

        logger.info(f"Registered model {model_name} as version {version}")
        return version

    def promote_to_champion(self, version: str):
        """
        Promote a candidate model to champion (production).
        Previous champion is retired (not deleted).

        In production: this triggers a CI/CD pipeline to redeploy the API.
        """
        version_dir = self.registry_dir / version
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found in registry")

        index = self._read_index()

        # Retire current champion
        if index["champion"]:
            old_version = index["champion"]
            self._update_version_status(old_version, "retired")
            logger.info(f"Retired champion {old_version}")

        # Promote new version
        self._update_version_status(version, "champion")
        index["champion"] = version
        self._write_index(index)

        # Copy to well-known paths for the API to load
        shutil.copy(version_dir / "model.joblib", ARTIFACTS_DIR / CHAMPION_MODEL_NAME)
        shutil.copy(version_dir / "winsorizer.joblib", ARTIFACTS_DIR / "winsorizer.joblib")
        shutil.copy(version_dir / "metadata.json", ARTIFACTS_DIR / METADATA_NAME)

        logger.info(f"✅ Promoted {version} to champion. API will load from {ARTIFACTS_DIR}")

    def _update_version_status(self, version: str, status: str):
        version_dir = self.registry_dir / version
        meta_path = version_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["status"] = status
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        index = self._read_index()
        for v in index["versions"]:
            if v["version"] == version:
                v["status"] = status
        self._write_index(index)

    def load_champion(self):
        """Load the current champion model pipeline."""
        model_path = ARTIFACTS_DIR / CHAMPION_MODEL_NAME
        winsorizer_path = ARTIFACTS_DIR / "winsorizer.joblib"
        meta_path = ARTIFACTS_DIR / METADATA_NAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"No champion model found at {model_path}. Run training first."
            )

        pipeline = joblib.load(model_path)
        winsorizer = joblib.load(winsorizer_path)
        with open(meta_path) as f:
            metadata = json.load(f)

        logger.info(f"Loaded champion model: {metadata['version']} ({metadata['model_name']})")
        return pipeline, winsorizer, metadata

    def list_versions(self) -> list:
        return self._read_index()["versions"]

    def get_champion_version(self) -> Optional[str]:
        return self._read_index().get("champion")
