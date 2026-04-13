from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from pyvrp._pyvrp import CostEvaluator, Solution
    from pyvrp.IteratedLocalSearch import IteratedLocalSearchCallbacks


@dataclass
class ObjectiveWeights:
    """
    Custom objective weights (additive on top of base cost, all >= 0).

    Parameters
    ----------
    vehicle_count
        Penalty per route used.
    route_balance
        Penalty proportional to the coefficient of variation of clients per
        route.
    dist
        Extra cost per unit of raw distance.
    time
        Extra cost per unit of raw duration.
    """

    vehicle_count: float = 0.0
    route_balance: float = 0.0
    dist: float = 0.0
    time: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vehicle_count", "route_balance", "dist", "time"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.vehicle_count, self.route_balance, self.dist, self.time)

    def apply_to(self, cost_evaluator: CostEvaluator) -> None:
        cost_evaluator.set_weights(
            self.vehicle_count,
            self.route_balance,
            self.dist,
            self.time,
        )


@dataclass
class IterationMetrics:
    """Per-iteration snapshot passed to adaptation strategies."""

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


class AdaptationStrategy(ABC):
    @abstractmethod
    def update(
        self,
        weights: ObjectiveWeights,
        metrics: IterationMetrics,
    ) -> ObjectiveWeights: ...


class LinearDecay(AdaptationStrategy):
    """
    Multiplies all weights by ``decay`` each iteration.

    Parameters
    ----------
    decay
        Multiplicative factor in (0, 1]. Default 0.9999.
    min_weight
        Floor value; weights are never reduced below this. Default 0.0.
    """

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
            dist=_d(weights.dist),
            time=_d(weights.time),
        )


class AdaptiveAdjustment(AdaptationStrategy):
    """
    Increases weights when the rolling feasibility rate is below
    ``target_feasibility``; decreases them otherwise.

    Parameters
    ----------
    target_feasibility
        Target fraction of feasible candidates. Default 0.5.
    increase_factor
        Multiplier when below target (must be > 1). Default 1.1.
    decrease_factor
        Multiplier when at/above target (must be in (0, 1)). Default 0.9.
    min_weight
        Weight lower bound. Default 0.0.
    max_weight
        Weight upper bound. Default 1e9.
    """

    def __init__(
        self,
        target_feasibility: float = 0.5,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.9,
        min_weight: float = 0.0,
        max_weight: float = 1e9,
    ):
        if not (0 <= target_feasibility <= 1):
            raise ValueError("target_feasibility must be in [0, 1].")
        if increase_factor <= 1:
            raise ValueError("increase_factor must be > 1.")
        if not (0 < decrease_factor < 1):
            raise ValueError("decrease_factor must be in (0, 1).")
        self._target = target_feasibility
        self._inc = increase_factor
        self._dec = decrease_factor
        self._min = min_weight
        self._max = max_weight

    def update(
        self, weights: ObjectiveWeights, metrics: IterationMetrics
    ) -> ObjectiveWeights:
        factor = (
            self._inc if metrics.feasibility_rate < self._target else self._dec
        )

        def _adj(v: float) -> float:
            return max(self._min, min(v * factor, self._max))

        return ObjectiveWeights(
            vehicle_count=_adj(weights.vehicle_count),
            route_balance=_adj(weights.route_balance),
            dist=_adj(weights.dist),
            time=_adj(weights.time),
        )


class MultiObjectiveScalarization(AdaptationStrategy):
    """
    Periodically resamples weights uniformly from the unit simplex to
    approximate Pareto front exploration.

    Parameters
    ----------
    objectives
        Which weights to randomise. Valid: ``"vehicle_count"``,
        ``"route_balance"``, ``"dist"``, ``"time"``.
        Default: ``["vehicle_count", "route_balance"]``.
    scale
        Overall scale factor for drawn weights. Default 1000.0.
    update_interval
        Iterations between resamples. Default 500.
    rng_seed
        Optional seed. Default ``None``.
    """

    _VALID = frozenset({"vehicle_count", "route_balance", "dist", "time"})

    def __init__(
        self,
        objectives: list[str] | None = None,
        scale: float = 1000.0,
        update_interval: int = 500,
        rng_seed: int | None = None,
    ):
        import random

        self._objectives = objectives or ["vehicle_count", "route_balance"]
        for obj in self._objectives:
            if obj not in self._VALID:
                raise ValueError(
                    f"Unknown objective '{obj}'. Valid: {sorted(self._VALID)}."
                )
        self._scale = scale
        self._interval = update_interval
        self._rng = random.Random(rng_seed)

    def update(
        self, weights: ObjectiveWeights, metrics: IterationMetrics
    ) -> ObjectiveWeights:
        if metrics.iteration % self._interval != 0:
            return weights

        n = len(self._objectives)
        raw = [self._rng.expovariate(1.0) for _ in range(n)]
        total = sum(raw)
        new_vals = {
            obj: (v / total) * self._scale
            for obj, v in zip(self._objectives, raw)
        }

        return ObjectiveWeights(
            vehicle_count=new_vals.get("vehicle_count", weights.vehicle_count),
            route_balance=new_vals.get("route_balance", weights.route_balance),
            dist=new_vals.get("dist", weights.dist),
            time=new_vals.get("time", weights.time),
        )


class AdaptiveObjective:
    """
    Manages adaptive objective weights inside the iterated local search.

    Parameters
    ----------
    initial_weights
        Starting weights. Can be :class:`ObjectiveWeights` or a dict with
        any subset of keys ``{"vehicle_count", "route_balance", "dist",
        "time"}``.
    strategy
        Adaptation strategy. ``None`` keeps weights fixed.
    update_every
        Iterations between strategy calls. Default 1.
    history_window
        Window size for the rolling feasibility rate. Default 100.
    """

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
        self._history: list[bool] = []
        self._history_window = history_window
        self._iteration = 0
        self._metrics_log: list[IterationMetrics] = []

    @property
    def weights(self) -> ObjectiveWeights:
        return self._weights

    @property
    def iteration(self) -> int:
        return self._iteration

    def evaluate(self, solution: Solution) -> float:
        """Return the custom-weighted contribution for *solution*."""
        w = self._weights
        value = 0.0
        if w.vehicle_count:
            value += w.vehicle_count * solution.num_routes()
        if w.route_balance:
            value += w.route_balance * solution.route_balance()
        if w.dist:
            value += w.dist * solution.distance()
        if w.time:
            value += w.time * solution.duration()
        return value

    def get_history(self) -> list[IterationMetrics]:
        return list(self._metrics_log)

    def get_history_dataframe(self) -> "pd.DataFrame":
        """Return iteration history as a ``pandas.DataFrame``."""
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
                "weight_dist": m.weights.dist,
                "weight_time": m.weights.time,
            }
            for m in self._metrics_log
        ]
        return pd.DataFrame(rows)

    def _on_iteration(
        self,
        current: Solution,
        candidate: Solution,
        best: Solution,
        cost_evaluator: CostEvaluator,
    ) -> None:
        self._iteration += 1

        self._history.append(candidate.is_feasible())
        if len(self._history) > self._history_window:
            self._history.pop(0)
        feasibility_rate = sum(self._history) / len(self._history)

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
        )
        self._metrics_log.append(metrics)

        if (
            self._strategy is not None
            and self._iteration % self._update_every == 0
        ):
            self._weights = self._strategy.update(self._weights, metrics)
            self._weights.apply_to(cost_evaluator)

    def as_callback(self) -> "IteratedLocalSearchCallbacks":
        from pyvrp.IteratedLocalSearch import IteratedLocalSearchCallbacks

        obj = self

        class _Callback(IteratedLocalSearchCallbacks):
            def on_iteration(
                self,
                current: Solution,
                candidate: Solution,
                best: Solution,
                cost_evaluator: CostEvaluator,
            ) -> None:
                if obj._iteration == 0:
                    obj._weights.apply_to(cost_evaluator)
                obj._on_iteration(current, candidate, best, cost_evaluator)

            def on_restart(self, best: Solution) -> None:  # type: ignore[override]
                obj._history.clear()

        return _Callback()
