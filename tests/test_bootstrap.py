import numpy as np

from cfgmp_eval.bootstrap import macro_score, paired_bootstrap, summarize


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    tags = np.array(["a"] * n + ["b"] * (n // 4))
    base = rng.random(len(tags))
    return tags, base


def test_macro_score_weights_tasks_equally():
    tags = np.array(["a", "a", "a", "b"])
    assert macro_score(tags, np.array([1.0, 1.0, 1.0, 0.0])) == 0.5


def test_identical_settings_have_zero_paired_difference():
    tags, base = _toy()
    boots = paired_bootstrap(tags, {"x": base, "y": base.copy()}, n_boot=500)
    assert np.allclose(boots["x"], boots["y"])


def test_paired_interval_is_tighter_than_unpaired():
    tags, base = _toy()
    scores = {"cfg": base, "new": np.clip(base + 0.02, 0, 1)}
    table = summarize(tags, scores, ["cfg"], n_boot=2000)
    d = table["new"]["diff_vs_cfg"]
    paired_width = d["ci"][1] - d["ci"][0]
    marginal_width = table["new"]["ci"][1] - table["new"]["ci"][0]
    assert d["resolved"] and paired_width < marginal_width / 5
