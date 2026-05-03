# PyVRP — Adaptive Objective Fork

A fork of [PyVRP](https://github.com/PyVRP/PyVRP) that adds adaptive objective
weights to the iterated local search.

## What this fork adds

Two custom solution-level weights in `CostEvaluator`:

- `vehicle_count` — additive penalty per route in the solution.
- `route_balance` — additive penalty proportional to imbalance across routes
  (coefficient of variation of clients-per-route: stddev / mean).

A Python-side adaptation system in `pyvrp/adaptive_objective.py`:

- `ObjectiveWeights` — the two weights as a dataclass.
- `IterationMetrics` — per-iteration snapshot with per-route data.
- `LinearDecay` — multiplies weights by a constant factor each iteration.
- `FairnessSignalAdjustment` — boosts `route_balance` when fairness is stable
  and cost is not growing; decays weights when cost grows.
- `AdaptiveObjective` — coordinator that wires a strategy to the ILS via
  callbacks.

A small extension to `PenaltyManager` to persist custom weights across the
internal `CostEvaluator` re-creations that happen each iteration.

A `Solution.route_balance()` method returning the coefficient of variation
(`stddev / mean`) of clients-per-route, and a `Solution.time_window_violation()` accessor.

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

Weights are applied in `penalisedCost<Solution>` — the function that scores
a candidate solution at the end of each ILS iteration. They influence which
candidate the iterated local search accepts.

Weights are not visible to the local search operators (2-opt, relocate, swap),
which use incremental `deltaCost` and bypass the full solution cost. This
means weights bias the *acceptance* step of ILS, not the *internal moves*.

The weights live in `PenaltyManager` rather than directly in `CostEvaluator`
because PyVRP re-creates the `CostEvaluator` on every iteration to update
penalty coefficients. `PenaltyManager.set_custom_weights()` persists the
weights across these re-creations.

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
- `CostEvaluator`: two extra weight fields, `set_weights()`, `get_weights()`.
- `Solution`: `route_balance()`, `time_window_violation()`.
- `PenaltyManager`: `set_custom_weights()`, custom-weight pass-through in
  both `cost_evaluator()` and `max_cost_evaluator()`.
- `IteratedLocalSearch`: `callbacks` field in `ILSParams`.
- `pyvrp.__init__`: exports `AdaptiveObjective`, `AdaptationStrategy`,
  `ObjectiveWeights`, `IterationMetrics`, `LinearDecay`,
  `FairnessSignalAdjustment`, `Route`, `ActivityType`.

