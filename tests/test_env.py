"""`.env` loading: fills what is missing, never overrides what the shell set."""

from __future__ import annotations

import os

from synapse import load_env


def test_dotenv_fills_missing_vars_and_shell_values_win(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\n"
        "SYNAPSE_TEST_FROM_FILE=file-value\n"
        'SYNAPSE_TEST_OVERRIDDEN="file-value"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SYNAPSE_NO_DOTENV")
    # The local-testing case: the shell points somewhere other than `.env`.
    monkeypatch.setenv("SYNAPSE_TEST_OVERRIDDEN", "shell-value")
    try:
        load_env()
        assert os.environ["SYNAPSE_TEST_FROM_FILE"] == "file-value"
        assert os.environ["SYNAPSE_TEST_OVERRIDDEN"] == "shell-value"
    finally:
        os.environ.pop("SYNAPSE_TEST_FROM_FILE", None)


def test_opt_out_loads_nothing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SYNAPSE_TEST_SKIPPED=x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNAPSE_NO_DOTENV", "1")
    load_env()
    assert "SYNAPSE_TEST_SKIPPED" not in os.environ
