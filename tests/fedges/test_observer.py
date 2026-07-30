from fedcausal.fedges.model import BNModel
from fedcausal.fedges.observer import MetricsObserver, RoundStats
from fedcausal.graph import DAG


def test_metrics_observer_records_structural_rows():
    truth = DAG.from_edges(3, [(0, 1), (1, 2)])
    obs = MetricsObserver(truth)
    obs.on_round("build", 1, 0, BNModel(DAG.from_edges(3, [(0, 1)]), score=-5.0))
    obs.on_round("server", 1, -1, BNModel(DAG.from_edges(3, [(0, 1), (1, 2)])))
    assert [r.stage for r in obs.rows] == ["build", "server"]
    build, server = obs.rows
    assert build.score == -5.0 and build.shd is not None and build.edges == 1
    # server model has no data-derived score; SHD vs truth is still computable
    assert server.score is None
    assert server.shd == 0 and server.smhd == 0


def test_metrics_observer_without_truth_skips_shd():
    obs = MetricsObserver(None)
    obs.on_round("build", 1, 0, BNModel(DAG.from_edges(2, [(0, 1)]), score=-1.0))
    assert obs.rows[0].shd is None and obs.rows[0].smhd is None
    assert isinstance(obs.rows[0], RoundStats)
