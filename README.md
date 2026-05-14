# PyVRP — Adaptive Objective Fork

A fork of [PyVRP](https://github.com/PyVRP/PyVRP) that adds adaptive objective
weights to the iterated local search.

## What this fork adds

Two custom solution-level weights in `CostEvaluator`:

- `vehicle_count` — additive penalty per route in the solution.
- `route_balance` — additive penalty proportional to imbalance across routes
  (mean absolute deviation of route distances from their mean: `Σ|d_i - d̄| / n`).

A Python-side adaptation system in `pyvrp/adaptive_objective.py`:

- `ObjectiveWeights` — the two weights as a dataclass.
- `IterationMetrics` — per-iteration snapshot with per-route data.
- `LinearDecay` — multiplies weights by a constant factor each iteration.
- `FairnessSignalAdjustment` — boosts `route_balance` when fairness is stable
  and cost is not growing; decays weights when cost grows.
- `AdaptiveObjective` — coordinator that wires a strategy to the ILS via
  callbacks.

A small extension to `PenaltyManager` to persist custom weights and the
route-distance target across the internal `CostEvaluator` re-creations that
happen each iteration.

A `Solution.route_balance()` method returning the mean absolute deviation of
route distances, and a `Solution.time_window_violation()` accessor.

## Installation

Requires Python 3.12 and a C++ compiler in PATH (MinGW on Windows, gcc on
Linux/macOS).

```
pip install git+https://github.com/ksenia274/PyVRP.git@main
```

To build from source:

```
git clone https://github.com/ksenia274/PyVRP.git
cd PyVRP
pip install -e . --no-build-isolation
```

## Quickstart

```python
from pyvrp import (
    Model,
    AdaptiveObjective,
    ObjectiveWeights,
    LinearDecay,
)
from pyvrp.stop import MaxIterations

model = Model.from_solomon("path/to/instance.txt")

adaptive = AdaptiveObjective(
    initial_weights=ObjectiveWeights(route_balance=10.0),
    strategy=LinearDecay(decay=0.999),
)

result = model.solve(
    stop=MaxIterations(5000),
    callbacks=adaptive.as_callback(),
)

history = adaptive.get_history_dataframe()
```

A complete example: [`examples/adaptive_vrp.py`](examples/adaptive_vrp.py).

## How weights work

`vehicle_count` is applied in `penalisedCost<Solution>` — the function that
scores a candidate solution at the end of each ILS iteration.

`route_balance` uses a decomposable MAD proxy: the penalty `Σ|d_i - target|`
is additive per route, so the incremental cost of a local-search move is
`rbw * (|new_dist - target| - |old_dist - target|)`. This delta is computed
inside `CostEvaluator::deltaCost`, making the balance signal visible to all
local-search operators (2-opt, relocate, swap, etc.), not just the acceptance
step.

The target distance is a snapshot of the current-solution mean route distance
taken at the start of each ILS iteration. It is stored in `CostEvaluator` via
`set_target_route_dist` and updated by `AdaptiveObjective` callbacks
(`on_start`, `on_iteration`, `on_restart`).

The weights live in `PenaltyManager` rather than directly in `CostEvaluator`
because PyVRP re-creates the `CostEvaluator` on every iteration to update
penalty coefficients. `PenaltyManager.set_custom_weights()` and
`PenaltyManager.set_target_route_dist()` persist these values across
re-creations.

## Writing your own strategy

Subclass `AdaptationStrategy` and implement `update`:

```python
from pyvrp import AdaptationStrategy, ObjectiveWeights

class StepwiseStrategy(AdaptationStrategy):
    """Switches between two weight configurations at iteration N."""

    def __init__(self, switch_at: int = 1000):
        self.switch_at = switch_at

    def update(self, weights, metrics):
        if metrics.iteration < self.switch_at:
            return ObjectiveWeights(route_balance=1.0)
        return ObjectiveWeights(route_balance=10.0)
```

Pass it to `AdaptiveObjective` like any built-in strategy:

```python
from pyvrp import AdaptiveObjective

adaptive = AdaptiveObjective(
    initial_weights=ObjectiveWeights(route_balance=1.0),
    strategy=StepwiseStrategy(switch_at=1000),
)
```

`metrics` (an `IterationMetrics` instance) gives access to per-iteration data:
current cost, best cost, feasibility, number of routes, route balance, and
per-route distances and loads. Use any of these to drive your weight updates.

## Differences from upstream

- New file: `pyvrp/adaptive_objective.py`.
- `CostEvaluator`: extra weight fields `vehicleCountWeight_`,
  `routeBalanceWeight_`, `targetRouteDist_`; methods `set_weights()`,
  `get_weights()`, `set_target_route_dist()`, `target_route_dist()`;
  MAD balance delta wired into both `deltaCost` overloads.
- `Route::Proposal`: new `rawDistance()` method (required by `deltaCost`
  to compute the balance delta without the cost/excess split).
- `Solution`: `route_balance()` returns mean absolute deviation of route
  distances (not CV by clients); `time_window_violation()` accessor.
- `PenaltyManager`: `set_custom_weights()`, `set_target_route_dist()`,
  pass-through in both `cost_evaluator()` and `max_cost_evaluator()`.
- `IteratedLocalSearch`: `callbacks` field in `ILSParams`.
- `pyvrp.__init__`: exports `AdaptiveObjective`, `AdaptationStrategy`,
  `ObjectiveWeights`, `IterationMetrics`, `LinearDecay`,
  `FairnessSignalAdjustment`, `Route`, `ActivityType`.

