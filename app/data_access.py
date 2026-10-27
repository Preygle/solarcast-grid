"""Cached, read-only access to project artefacts for the dashboard. Missing artefacts raise
``ArtefactMissing`` with the command that produces them; pages show that message instead of failing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
PRODUCERS = {
    "data/processed/hourly.parquet": "python scripts/build_dataset.py",
    "results/eda": "python scripts/run_eda.py",
    "results/model_comparison.csv": "python scripts/train_models.py",
    "models/registry.json": "python scripts/train_models.py",
    "results/grid_scenarios/grid_summary.json": "python scripts/run_grid_analysis.py",
    "results/grid_model_comparison.csv": "python scripts/run_receiving_power.py",
    "results/optimization_comparison.csv": "python scripts/run_optimization.py",
}


class ArtefactMissing(FileNotFoundError):
    pass


def _need(rel: str) -> Path:
    p = ROOT / rel
    if not p.exists():
        prod = next((v for k, v in PRODUCERS.items() if rel.startswith(k)), "see README")
        raise ArtefactMissing(f"`{rel}` not found. Produce it with `{prod}`.")
    return p


@st.cache_data
def hourly() -> pd.DataFrame:
    return pd.read_parquet(_need("data/processed/hourly.parquet"))


@st.cache_data
def csv(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(_need(rel), **kw)


@st.cache_data
def js(rel: str) -> dict:
    return json.loads(_need(rel).read_text())


@st.cache_data
def predictions(task: str) -> pd.DataFrame:
    return pd.read_parquet(_need(f"results/predictions/{task}.parquet"))


def image(rel: str) -> Path:
    return _need(rel)


@st.cache_resource
def forecaster():
    from src.inference.forecast import Forecaster
    _need("models/registry.json")
    return Forecaster()


@st.cache_resource
def receiving_models() -> dict:
    import joblib
    from src.receiving_power.model import TARGETS
    return {t: joblib.load(_need(f"models/receiving_power/{t}.joblib")) for t in TARGETS}


def guard(fn):
    """Run a page section; show a clear message if an artefact is missing."""
    try:
        fn()
    except ArtefactMissing as e:
        st.warning(str(e))
