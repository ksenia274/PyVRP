from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from pyvrp._pyvrp import CostEvaluator, Solution
    from pyvrp.IteratedLocalSearch import IteratedLocalSearchCallbacks
    from pyvrp.PenaltyManager import PenaltyManager


@dataclass
class ObjectiveWeights:

    vehicle_count: float = 0.0
    route_balance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vehicle_count", "route_balance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")

    def as_tuple(self) -> tuple[float, float]:
        return (self.vehicle_count, self.route_balance)

    def apply_to(self, cost_evaluator: "CostEvaluator") -> None:
        cost_evaluator.set_weights(self.vehicle_count, self.route_balance)


@dataclass
class IterationMetrics:
    iteration: int
    current_cost: int
    best_cost: int
    current_feasible: bool
    best_feasible: bool
    feasibility_rate: float
    num_routes: int
    route_balance: float
    time_window_violation: float
    weights: ObjectiveWeights

    per_route_clients: list[int] = field(default_factory=list)
    per_route_distances: list[float] = field(default_factory=list)
    per_route_loads: list[list[float]] = field(default_factory=list)


class AdaptationStrategy(ABC):
    @abstractmethod
    def update(
        self,
        weights: ObjectiveWeights,
        metrics: IterationMetrics,
    ) -> ObjectiveWeights: ...


class LinearDecay(AdaptationStrategy):
    """Multiplies all weights by `decay` each iteration, clamped at `min_weight`."""

    def __init__(self, decay: float = 0.9999, min_weight: float = 0.0):
        if not (0 < decay <= 1):
            raise ValueError("decay must be in (0, 1].")
        if min_weight < 0:
            raise ValueError("min_weight must be >= 0.")
        self._decay = decay
        self._min = min_weight

    def update(
        self, weights: ObjectiveWeights, metrics: IterationMetrics
    ) -> ObjectiveWeights:
        def _d(v: float) -> float:
            return max(v * self._decay, self._min)

        return ObjectiveWeights(
            vehicle_count=_d(weights.vehicle_count),
            route_balance=_d(weights.route_balance),
        )


class FairnessSignalAdjustment(AdaptationStrategy):
    """Boosts `route_balance` when fairness is stable, decays all weights when cost grows."""

    def __init__(
        self,
        epsilon: float = 0.01,
        cost_budget: float = 0.0,
        boost_factor: float = 1.1,
        decay_factor: float = 0.9,
        window: int = 10,
        min_weight: float = 0.0,
        max_weight: float = 1e9,
    ):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0.")
        if boost_factor <= 1:
            raise ValueError("boost_factor must be > 1.")
        if not (0 < decay_factor < 1):
            raise ValueError("decay_factor must be in (0, 1).")
        if window < 2:
            raise ValueError("window must be >= 2.")

        self._epsilon = epsilon
        self._cost_budget = cost_budget
        self._boost = boost_factor
        self._decay = decay_factor
        self._window = window
        self._min = min_weight
        self._max = max_weight

        self._balance_buf: deque[float] = deque(maxlen=window)
        self._cost_buf: deque[int] = deque(maxlen=window)

    def _clip(self, v: float) -> float:
        return max(self._min, min(v, self._max))

    def update(
        self, weights: ObjectiveWeights, metrics: IterationMetrics
    ) -> ObjectiveWeights:
        self._balance_buf.append(metrics.route_balance)
        self._cost_buf.append(metrics.current_cost)

        if len(self._balance_buf) < 2:
            return weights

        balances = list(self._balance_buf)
        fairness_delta = sum(
            abs(balances[i] - balances[i - 1]) for i in range(1, len(balances))
        ) / (len(balances) - 1)

        costs = list(self._cost_buf)
        cost_growth = sum(
            costs[i] - costs[i - 1] for i in range(1, len(costs))
        ) / (len(costs) - 1)

        if fairness_delta < self._epsilon and cost_growth < self._cost_budget:
            return ObjectiveWeights(
                vehicle_count=weights.vehicle_count,
                route_balance=self._clip(weights.route_balance * self._boost),
            )

        if cost_growth > self._cost_budget:
            return ObjectiveWeights(
                vehicle_count=self._clip(weights.vehicle_count * self._decay),
                route_balance=self._clip(weights.route_balance * self._decay),
            )

        return weights


class AdaptiveObjective:

    def __init__(
        self,
        initial_weights: ObjectiveWeights | dict[str, float] | None = None,
        strategy: AdaptationStrategy | None = None,
        update_every: int = 1,
        history_window: int = 100,
    ):
        if isinstance(initial_weights, dict):
            self._weights = ObjectiveWeights(**initial_weights)
        elif initial_weights is None:
            self._weights = ObjectiveWeights()
        else:
            self._weights = initial_weights

        self._strategy = strategy
        self._update_every = max(1, update_every)
        self._history: deque[bool] = deque(maxlen=history_window)
        self._history_window = history_window
        self._iteration = 0
        self._metrics_log: list[IterationMetrics] = []
        self._weight_applied_count: int = 0

    @property
    def weights(self) -> ObjectiveWeights:
        return self._weights

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def weight_applied_count(self) -> int:
        return self._weight_applied_count

    def fairness_delta_ma(self, k: int = 10) -> float:
        """Mean absolute change of `route_balance` over the last k iterations."""
        log = self._metrics_log
        if len(log) < 2:
            return 0.0
        recent = log[-k:]
        if len(recent) < 2:
            return 0.0
        deltas = [
            abs(recent[i].route_balance - recent[i - 1].route_balance)
            for i in range(1, len(recent))
        ]
        return sum(deltas) / len(deltas)

    def evaluate(self, solution: "Solution") -> float:
        w = self._weights
        value = 0.0
        if w.vehicle_count:
            value += w.vehicle_count * solution.num_routes()
        if w.route_balance:
            value += w.route_balance * solution.route_balance()
        return value

    def get_history(self) -> list[IterationMetrics]:
        return list(self._metrics_log)

    def get_history_dataframe(self) -> "pd.DataFrame":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for get_history_dataframe()."
            ) from exc

        rows = [
            {
                "iteration": m.iteration,
                "current_cost": m.current_cost,
                "best_cost": m.best_cost,
                "current_feasible": m.current_feasible,
                "best_feasible": m.best_feasible,
                "feasibility_rate": m.feasibility_rate,
                "num_routes": m.num_routes,
                "route_balance": m.route_balance,
                "time_window_violation": m.time_window_violation,
                "weight_vehicle_count": m.weights.vehicle_count,
                "weight_route_balance": m.weights.route_balance,
            }
            for m in self._metrics_log
        ]
        return pd.DataFrame(rows)

    def _on_iteration(
        self,
        current: "Solution",
        candidate: "Solution",
        best: "Solution",
        cost_evaluator: "CostEvaluator",
    ) -> None:
        self._iteration += 1

        self._history.append(candidate.is_feasible())
        feasibility_rate = sum(self._history) / len(self._history)

        routes = current.routes()
        per_route_clients = [r.num_clients() for r in routes]
        per_route_distances = [float(r.distance()) for r in routes]
        per_route_loads = [
            [float(v) for v in r.delivery()] for r in routes
        ]

        metrics = IterationMetrics(
            iteration=self._iteration,
            current_cost=cost_evaluator.penalised_cost(current),
            best_cost=cost_evaluator.penalised_cost(best),
            current_feasible=current.is_feasible(),
            best_feasible=best.is_feasible(),
            feasibility_rate=feasibility_rate,
            num_routes=current.num_routes(),
            route_balance=current.route_balance(),
            time_window_violation=current.time_window_violation(),
            weights=ObjectiveWeights(*self._weights.as_tuple()),
            per_route_clients=per_route_clients,
            per_route_distances=per_route_distances,
            per_route_loads=per_route_loads,
        )
        self._metrics_log.append(metrics)

        if (
            self._strategy is not None
            and self._iteration % self._update_every == 0
        ):
            self._weights = self._strategy.update(self._weights, metrics)

    def as_callback(self) -> "IteratedLocalSearchCallbacks":
        from pyvrp.IteratedLocalSearch import IteratedLocalSearchCallbacks

        obj = self

        class _Callback(IteratedLocalSearchCallbacks):
            def __init__(self) -> None:
                self._pm: "PenaltyManager | None" = None

            def on_setup(self, pm: "PenaltyManager") -> None:
                self._pm = pm
                pm.set_custom_weights(obj._weights)
                obj._weight_applied_count += 1

            def on_iteration(
                self,
                current: "Solution",
                candidate: "Solution",
                best: "Solution",
                cost_evaluator: "CostEvaluator",
            ) -> None:
                prev_weights = obj._weights.as_tuple()
                obj._on_iteration(current, candidate, best, cost_evaluator)
                if self._pm is not None:
                    self._pm.set_custom_weights(obj._weights)
                    if obj._weights.as_tuple() != prev_weights:
                        obj._weight_applied_count += 1

            def on_restart(self, best: "Solution") -> None:
                obj._history.clear()

        return _Callback()
