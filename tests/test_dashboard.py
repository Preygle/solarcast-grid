"""Headless dashboard checks (streamlit.testing). Skipped when trained artefacts are absent."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "registry.json").exists(), reason="artefacts not built")


def _page(name):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(name).run()
    return at


@pytest.mark.parametrize("name", ["Overview", "Forecasting", "Duck curve", "Grid & power delivery", "Model evaluation"])
def test_pages_render_without_exceptions(name):
    at = _page(name)
    assert not at.exception, [e.value for e in at.exception]


def test_grid_page_shows_simulated_and_predicted_values():
    at = _page("Grid & power delivery")
    labels = [m.label for m in at.metric]
    assert any("Received at bus 0" in l for l in labels) and any("losses" in l for l in labels)
    assert all("model:" in (m.delta or "") or "Voltage" in m.label for m in at.metric)


@pytest.mark.skipif(not (ROOT / "results" / "optimization_comparison.csv").exists(), reason="optimisation not run")
def test_optimisation_button_runs_lp_and_ac_validation():
    at = _page("Optimisation")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 2  # saved comparison + live result table
