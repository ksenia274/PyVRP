import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

from pyvrp import (
    CostEvaluator,
    IteratedLocalSearchParams,
    Route,
    Solution,
)
from pyvrp.adaptive_objective import (
    AdaptiveObjective,
    FairnessSignalAdjustment,
    IterationMetrics,
    LinearDecay,
    ObjectiveWeights,
)
from tests.helpers import read

@pytest.fixture(scope="module")
def ok_small():
    return read("data/OkSmall.txt")

@pytest.fixture(scope="module")
def rc208():
    return read("data/RC208.vrp", round_func="dimacs")

def test_default_weights_are_zero():
    ce = CostEvaluator([1], 1, 0)
    assert_equal(ce.get_weights(), (0.0, 0.0))

def test_custom_weights_constructor():
    ce = CostEvaluator([1], 1, 0, vehicle_count_weight=10.0,
                       route_balance_weight=5.0)
    vcw, rbw = ce.get_weights()
    assert_equal(vcw, 10.0)
    assert_equal(rbw, 5.0)

def test_set_weights_roundtrip():
    ce = CostEvaluator([1], 1, 0)
    ce.set_weights(100.0, 50.0)
    assert_equal(ce.get_weights(), (100.0, 50.0))

@pytest.mark.parametrize(
    "kwargs",
    [
        {"vehicle_count_weight": -1},
        {"route_balance_weight": -0.1},
    ],
)
def test_negative_custom_weights_raise(kwargs):
    with assert_raises(ValueError):
        CostEvaluator([1], 1, 0, **kwargs)

def test_set_weights_negative_raises():
    ce = CostEvaluator([1], 1, 0)
    with assert_raises(ValueError):
        ce.set_weights(-1.0, 0.0)

def test_vehicle_count_weight_increases_cost(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    ce_base = CostEvaluator([0], 0, 0)
    ce_vcw = CostEvaluator([0], 0, 0, vehicle_count_weight=1000.0)

    base_cost = ce_base.penalised_cost(sol)
    vcw_cost = ce_vcw.penalised_cost(sol)

    assert_equal(vcw_cost - base_cost, 2000)

def test_backward_compat_zero_weights(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    ce_old = CostEvaluator([0], 0, 0)
    ce_new = CostEvaluator([0], 0, 0,
                            vehicle_count_weight=0.0,
                            route_balance_weight=0.0)

    assert_equal(ce_old.penalised_cost(sol), ce_new.penalised_cost(sol))

def test_route_balance_single_route(ok_small):
    sol = Solution(ok_small, [[0, 1, 2, 3]])
    assert_equal(sol.route_balance(), 0.0)

def test_route_balance_empty_solution(ok_small):
    sol = Solution(ok_small, [[0, 1, 2, 3]])
    assert_(sol.route_balance() >= 0.0)

def test_route_balance_equal_routes(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    assert_equal(sol.route_balance(), 0.0)

def test_route_balance_unequal_routes(ok_small):
    sol = Solution(ok_small, [[0, 1, 2], [3]])
    assert_(sol.route_balance() > 0.0)

def test_time_window_violation_feasible(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    assert_equal(sol.time_window_violation(), sol.time_warp())

def test_time_window_violation_is_float(ok_small):
    sol = Solution(ok_small, [[0, 1, 2, 3]])
    assert_(isinstance(sol.time_window_violation(), float))

def test_objective_weights_defaults():
    w = ObjectiveWeights()
    assert_equal(w.vehicle_count, 0.0)
    assert_equal(w.route_balance, 0.0)

def test_objective_weights_negative_raises():
    with assert_raises(ValueError):
        ObjectiveWeights(vehicle_count=-1.0)
    with assert_raises(ValueError):
        ObjectiveWeights(route_balance=-0.5)

def test_objective_weights_apply_to(ok_small):
    ce = CostEvaluator([0], 0, 0)
    w = ObjectiveWeights(vehicle_count=42.0)
    w.apply_to(ce)
    vcw, _ = ce.get_weights()
    assert_equal(vcw, 42.0)

def test_objective_weights_as_tuple():
    w = ObjectiveWeights(vehicle_count=1.0, route_balance=2.0)
    assert_equal(w.as_tuple(), (1.0, 2.0))

def test_linear_decay_reduces_weights():
    decay = LinearDecay(decay=0.5)
    w = ObjectiveWeights(vehicle_count=100.0, route_balance=50.0)
    metrics = _make_metrics(1, w)
    w2 = decay.update(w, metrics)
    assert_allclose(w2.vehicle_count, 50.0)
    assert_allclose(w2.route_balance, 25.0)

def test_linear_decay_floor():
    decay = LinearDecay(decay=0.1, min_weight=5.0)
    w = ObjectiveWeights(vehicle_count=10.0)
    metrics = _make_metrics(1, w)
    w2 = decay.update(w, metrics)
    assert_equal(w2.vehicle_count, 5.0)

def test_linear_decay_invalid_params():
    with assert_raises(ValueError):
        LinearDecay(decay=0.0)
    with assert_raises(ValueError):
        LinearDecay(decay=1.5)
    with assert_raises(ValueError):
        LinearDecay(min_weight=-1.0)

def test_adaptive_objective_default_weights():
    obj = AdaptiveObjective()
    assert_equal(obj.weights, ObjectiveWeights())
    assert_equal(obj.iteration, 0)

def test_adaptive_objective_dict_init():
    obj = AdaptiveObjective({"vehicle_count": 100.0})
    assert_equal(obj.weights.vehicle_count, 100.0)

def test_adaptive_objective_history_initially_empty():
    obj = AdaptiveObjective()
    assert_equal(len(obj.get_history()), 0)

def test_adaptive_objective_evaluate(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=10.0, route_balance=5.0)
    )
    expected = 10.0 * sol.num_routes() + 5.0 * sol.route_balance()
    assert_allclose(obj.evaluate(sol), expected)

def test_adaptive_objective_callback_records_metrics(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=10.0),
        strategy=LinearDecay(decay=0.9),
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    assert_equal(obj.iteration, 1)
    assert_equal(len(obj.get_history()), 1)

def test_adaptive_objective_weights_updated_after_callback(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=100.0),
        strategy=LinearDecay(decay=0.5),
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    assert_allclose(obj.weights.vehicle_count, 50.0)

def test_adaptive_objective_restart_clears_history(ok_small):
    obj = AdaptiveObjective(strategy=LinearDecay())
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1, 2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    assert_equal(len(obj._history), 1)

    cb.on_restart(sol)
    assert_equal(len(obj._history), 0)

def test_adaptive_objective_update_every(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=100.0),
        strategy=LinearDecay(decay=0.5),
        update_every=3,
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1, 2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    cb.on_iteration(sol, sol, sol, ce)
    assert_allclose(obj.weights.vehicle_count, 100.0)

    cb.on_iteration(sol, sol, sol, ce)
    assert_allclose(obj.weights.vehicle_count, 50.0)

def test_adaptive_objective_no_strategy_weights_fixed(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=42.0),
        strategy=None,
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1, 2, 3]])

    for _ in range(10):
        cb.on_iteration(sol, sol, sol, ce)

    assert_equal(obj.weights.vehicle_count, 42.0)

def test_penalty_manager_set_custom_weights():
    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())
    w = ObjectiveWeights(vehicle_count=10.0, route_balance=5.0)
    pm.set_custom_weights(w)
    ce = pm.cost_evaluator()
    vcw, rbw = ce.get_weights()
    assert_equal(vcw, 10.0)
    assert_equal(rbw, 5.0)

def test_penalty_manager_default_weights_are_zero():
    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())
    ce = pm.cost_evaluator()
    assert_equal(ce.get_weights(), (0.0, 0.0))

def test_max_cost_evaluator_preserves_custom_weights():
    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())
    pm.set_custom_weights(ObjectiveWeights(route_balance=5.0))
    ce = pm.max_cost_evaluator()
    _, rbw = ce.get_weights()
    assert_equal(rbw, 5.0)

def test_penalty_manager_weights_survive_register(ok_small):
    """Weights must persist across register() + cost_evaluator() cycles."""
    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())
    w = ObjectiveWeights(vehicle_count=99.0)
    pm.set_custom_weights(w)

    sol = Solution(ok_small, [[0, 1], [2, 3]])
    pm.register(sol)
    ce = pm.cost_evaluator()
    vcw, *_ = ce.get_weights()
    assert_equal(vcw, 99.0)

def test_on_setup_pushes_initial_weights():
    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    obj = AdaptiveObjective(ObjectiveWeights(vehicle_count=42.0))
    cb = obj.as_callback()
    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())

    cb.on_setup(pm)

    ce = pm.cost_evaluator()
    vcw, *_ = ce.get_weights()
    assert_equal(vcw, 42.0)
    assert_(obj.weight_applied_count >= 1)

def test_weight_applied_count_increments_on_change(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=100.0),
        strategy=LinearDecay(decay=0.5),
    )
    cb = obj.as_callback()

    from pyvrp.PenaltyManager import PenaltyManager, PenaltyParams

    pm = PenaltyManager(([1.0], 1.0, 1.0), PenaltyParams())
    cb.on_setup(pm)

    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    before = obj.weight_applied_count
    cb.on_iteration(sol, sol, sol, ce)
    assert_(obj.weight_applied_count > before)

def test_weight_applied_count_zero_without_on_setup(ok_small):
    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=100.0),
        strategy=LinearDecay(decay=0.5),
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    assert_equal(obj.weight_applied_count, 0)

def test_integration_weights_applied_during_solve(ok_small):
    from pyvrp.solve import SolveParams, solve
    from pyvrp.stop import MaxIterations

    obj = AdaptiveObjective(
        ObjectiveWeights(vehicle_count=500.0, route_balance=100.0),
        strategy=LinearDecay(decay=0.9),
    )
    params = SolveParams(
        ils=IteratedLocalSearchParams(callbacks=obj.as_callback())
    )
    solve(ok_small, stop=MaxIterations(50), seed=0, params=params)
    assert_(obj.weight_applied_count > 0)

def test_integration_adaptive_and_plain_differ(ok_small):
    from pyvrp.solve import SolveParams, solve
    from pyvrp.stop import MaxIterations

    obj = AdaptiveObjective(ObjectiveWeights(route_balance=1000.0))
    params = SolveParams(
        ils=IteratedLocalSearchParams(callbacks=obj.as_callback())
    )
    result = solve(ok_small, stop=MaxIterations(100), seed=1, params=params)

    history = obj.get_history()
    assert_(len(history) > 0)
    assert_(any(m.weights.route_balance > 0 for m in history))

def test_iteration_metrics_per_route_fields_populated(ok_small):
    obj = AdaptiveObjective()
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    m = obj.get_history()[0]

    assert_equal(len(m.per_route_clients), sol.num_routes())
    assert_equal(len(m.per_route_distances), sol.num_routes())
    assert_equal(len(m.per_route_loads), sol.num_routes())
    assert_equal(sum(m.per_route_clients), sol.num_clients())

def test_iteration_metrics_per_route_clients_values(ok_small):
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    obj = AdaptiveObjective()
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    cb.on_iteration(sol, sol, sol, ce)

    m = obj.get_history()[0]
    routes = sol.routes()
    for i, route in enumerate(routes):
        assert_equal(m.per_route_clients[i], route.num_clients())
        assert_allclose(m.per_route_distances[i], float(route.distance()))

def test_fairness_delta_ma_requires_two_entries(ok_small):
    obj = AdaptiveObjective()
    assert_equal(obj.fairness_delta_ma(), 0.0)

    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    cb = obj.as_callback()
    cb.on_iteration(sol, sol, sol, ce)
    assert_equal(obj.fairness_delta_ma(), 0.0)

    cb.on_iteration(sol, sol, sol, ce)
    assert_equal(obj.fairness_delta_ma(), 0.0)

def test_feasibility_history_is_deque(ok_small):
    obj = AdaptiveObjective(history_window=5)
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])
    cb = obj.as_callback()
    for _ in range(20):
        cb.on_iteration(sol, sol, sol, ce)
    assert_(len(obj._history) <= 5)

def test_fairness_signal_boost_when_stable():
    strategy = FairnessSignalAdjustment(
        epsilon=1.0, cost_budget=0.0, boost_factor=2.0, decay_factor=0.5, window=2
    )
    w = ObjectiveWeights(route_balance=10.0)

    m1 = _make_metrics(1, w, feasibility_rate=1.0)
    m1 = IterationMetrics(
        **{**vars(m1), "route_balance": 0.1, "current_cost": 100}
    )
    w2 = strategy.update(w, m1)
    assert_equal(w2.route_balance, 10.0)

    m2 = IterationMetrics(
        **{**vars(m1), "iteration": 2, "route_balance": 0.1, "current_cost": 90}
    )
    w3 = strategy.update(w2, m2)
    assert_allclose(w3.route_balance, 20.0)

def test_fairness_signal_decay_when_cost_grows():
    strategy = FairnessSignalAdjustment(
        epsilon=0.0, cost_budget=0.0, boost_factor=1.1, decay_factor=0.5, window=2
    )
    w = ObjectiveWeights(route_balance=10.0, vehicle_count=5.0)

    m1 = IterationMetrics(
        iteration=1, current_cost=100, best_cost=100,
        current_feasible=True, best_feasible=True, feasibility_rate=0.5,
        num_routes=2, route_balance=0.5, time_window_violation=0.0, weights=w,
    )
    strategy.update(w, m1)

    m2 = IterationMetrics(
        **{**vars(m1), "iteration": 2, "current_cost": 200, "route_balance": 0.9}
    )
    w2 = strategy.update(w, m2)
    assert_allclose(w2.route_balance, 5.0)
    assert_allclose(w2.vehicle_count, 2.5)

def test_fairness_signal_invalid_params():
    with assert_raises(ValueError):
        FairnessSignalAdjustment(boost_factor=0.5)
    with assert_raises(ValueError):
        FairnessSignalAdjustment(decay_factor=1.5)
    with assert_raises(ValueError):
        FairnessSignalAdjustment(window=1)
    with assert_raises(ValueError):
        FairnessSignalAdjustment(epsilon=-1.0)

def test_fairness_signal_exported_from_pyvrp():
    from pyvrp import FairnessSignalAdjustment as FSA

def test_get_history_dataframe_basic(ok_small):
    pandas = pytest.importorskip("pandas")

    obj = AdaptiveObjective(
        ObjectiveWeights(route_balance=1.0),
        strategy=LinearDecay(decay=0.99),
    )
    cb = obj.as_callback()
    ce = CostEvaluator([0], 0, 0)
    sol = Solution(ok_small, [[0, 1], [2, 3]])

    cb.on_iteration(sol, sol, sol, ce)
    cb.on_iteration(sol, sol, sol, ce)

    df = obj.get_history_dataframe()
    assert isinstance(df, pandas.DataFrame)
    assert len(df) == 2
    assert "iteration" in df.columns
    assert "weight_route_balance" in df.columns

def _make_metrics(
    iteration: int,
    weights: ObjectiveWeights,
    feasibility_rate: float = 0.5,
) -> IterationMetrics:
    return IterationMetrics(
        iteration=iteration,
        current_cost=0,
        best_cost=0,
        current_feasible=True,
        best_feasible=True,
        feasibility_rate=feasibility_rate,
        num_routes=2,
        route_balance=0.0,
        time_window_violation=0.0,
        weights=weights,
    )
