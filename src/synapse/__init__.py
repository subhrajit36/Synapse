"""Synapse: Graph-based resume-JD matching with explainable skill reasoning."""

import os
from pathlib import Path

__version__ = "0.2.0"  # Phase C1 complete, C2 in progress


def load_env() -> None:
    """Load `.env` into the environment, never replacing what is already set.

    Runs on first import of the package, before any submodule reads its
    `os.getenv` defaults, so the server, the pipeline CLI and the migration
    scripts all see the same settings without exporting them by hand.

    A variable already in the environment wins. That is what makes a per-shell
    override work - e.g. pointing NEO4J_* at a local container for testing while
    `.env` still names AuraDB - and it is why Render's dashboard variables take
    precedence in production, where there is no `.env` at all.

    The file is looked for from the working directory upwards, then at the
    project root. `SYNAPSE_NO_DOTENV=1` disables loading; the test suite sets
    it so tests never pick up real credentials.
    """
    if os.getenv("SYNAPSE_NO_DOTENV"):
        return
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is pinned in requirements
        return
    path = find_dotenv(usecwd=True) or Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(path, override=False)


load_env()
