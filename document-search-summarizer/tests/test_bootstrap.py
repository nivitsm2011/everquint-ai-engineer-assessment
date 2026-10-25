"""Tests for config inheritance (hosted config) and the first-start bootstrap check."""

from __future__ import annotations

from pathlib import Path

from src.bootstrap import REQUIRED, ensure_ready, is_ready
from src.config import PROJECT_ROOT, Config, load_config, read_yaml


def test_cloud_config_extends_base_and_uses_separate_folders() -> None:
    cloud = Config(read_yaml(PROJECT_ROOT / "config" / "config.cloud.yaml"))
    base = load_config()
    assert cloud.get("corpus.train_docs") == 2000 and cloud.get("llm.provider") == "openai"
    assert cloud.get("search.rrf_k") == base.get("search.rrf_k")  # inherited from config.yaml
    assert cloud.get("embedding.model_name") == base.get("embedding.model_name")
    for key in ("paths.documents_file", "paths.bm25_index", "paths.faiss_index", "paths.cache_dir"):
        assert cloud.get(key) != base.get(key)  # never overwrites the full-corpus files


def test_read_yaml_extends_deep_merges(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("x: {p: 1, q: 2}\ny: 5\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nx: {q: 9}\n", encoding="utf-8")
    assert read_yaml(tmp_path / "b.yaml") == {"x": {"p": 1, "q": 9}, "y": 5}


def test_ensure_ready_is_a_noop_when_files_exist(tmp_path: Path) -> None:
    data = {"paths": {k.split(".")[1]: str(tmp_path / k) for k in REQUIRED}}
    cfg = Config(data)
    assert not is_ready(cfg)
    for key in REQUIRED:
        cfg.path(key).write_text("x")
    assert is_ready(cfg) and ensure_ready(cfg) is False
