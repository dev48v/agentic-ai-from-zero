from agentfuse import CanaryFuse, VersionStats, percentile


def _stats(version, n, latency, cost, errors=0):
    s = VersionStats(version)
    for i in range(n):
        s.record(latency, cost, error=i < errors)
    return s


def test_routing_is_deterministic_not_random():
    fuse = CanaryFuse("v1", "v2", candidate_pct=20)
    first = [fuse.route(f"req-{i}") for i in range(50)]
    second = [fuse.route(f"req-{i}") for i in range(50)]
    assert first == second


def test_routing_respects_the_percentage_roughly():
    fuse = CanaryFuse("v1", "v2", candidate_pct=20)
    routed = [fuse.route(f"req-{i:04d}") for i in range(1000)]
    share = routed.count("v2") / len(routed)
    assert 0.15 <= share <= 0.25


def test_zero_percent_sends_nothing_to_the_candidate():
    fuse = CanaryFuse("v1", "v2", candidate_pct=0)
    assert {fuse.route(f"r{i}") for i in range(100)} == {"v1"}


def test_a_healthy_candidate_is_promoted():
    fuse = CanaryFuse("v1", "v2", 20)
    v = fuse.evaluate(_stats("v1", 20, 1000, 0.001), _stats("v2", 10, 1050, 0.0011))
    assert v.decision == "promote" and fuse.active == "v2"


def test_cost_blowout_rolls_back_even_when_latency_and_errors_are_fine():
    """Project 11's real result: the candidate was 2.20x the baseline cost per request and
    was pulled automatically. Nothing else about it looked wrong."""
    fuse = CanaryFuse("v1", "v2", 20)
    v = fuse.evaluate(_stats("v1", 20, 1000, 0.001), _stats("v2", 10, 1000, 0.0022))
    assert v.rolled_back
    assert [g.name for g in v.failed] == ["cost-per-request"]
    assert fuse.active == "v1"
    assert any("ROLLBACK" in e for e in fuse.events)


def test_latency_regression_rolls_back():
    fuse = CanaryFuse("v1", "v2", 20)
    v = fuse.evaluate(_stats("v1", 20, 1000, 0.001), _stats("v2", 10, 2000, 0.001))
    assert v.rolled_back and "latency-p95" in [g.name for g in v.failed]


def test_added_errors_roll_back():
    fuse = CanaryFuse("v1", "v2", 20)
    v = fuse.evaluate(_stats("v1", 20, 1000, 0.001),
                      _stats("v2", 10, 1000, 0.001, errors=3))
    assert v.rolled_back and "error-rate" in [g.name for g in v.failed]


def test_too_little_traffic_is_a_rollback_not_a_promote():
    """'We saw no problem in two requests' is not 'there is no problem'."""
    fuse = CanaryFuse("v1", "v2", 20, min_candidate_requests=3)
    v = fuse.evaluate(_stats("v1", 20, 1000, 0.001), _stats("v2", 2, 900, 0.0009))
    assert v.rolled_back and [g.name for g in v.failed] == ["sample-size"]


def test_no_baseline_is_a_rollback():
    fuse = CanaryFuse("v1", "v2", 20)
    v = fuse.evaluate(VersionStats("v1"), _stats("v2", 10, 900, 0.0009))
    assert v.rolled_back and [g.name for g in v.failed] == ["baseline"]


def test_percentile_needs_no_numpy():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 95) == 0.0


def test_invalid_percentage_is_rejected():
    try:
        CanaryFuse("v1", "v2", 120)
    except ValueError:
        return
    raise AssertionError("candidate_pct=120 should be rejected")
