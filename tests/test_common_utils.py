from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from repository.analysis.common_utils import (
    ACTIVE_WORKSPACE_ENV_VAR,
    activate_workspace_path,
    capture_saved_artifacts,
    save_artifact,
)


def test_save_artifact_writes_into_active_workspace(tmp_path: Path) -> None:
    with activate_workspace_path(str(tmp_path)):
        artifact = save_artifact("reports/summary.json", {"status": "ok", "count": 3})

    target = tmp_path / "reports" / "summary.json"
    assert artifact["path"] == str(target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok", "count": 3}


def test_capture_saved_artifacts_tracks_written_files(tmp_path: Path) -> None:
    with activate_workspace_path(str(tmp_path)):
        with capture_saved_artifacts() as saved:
            save_artifact("notes/run.txt", "completed")

    assert len(saved) == 1
    assert saved[0]["relative_path"] == "notes/run.txt"
    assert os.getenv(ACTIVE_WORKSPACE_ENV_VAR) is None
