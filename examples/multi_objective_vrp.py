"""
Multi-objective VRP example
============================
Compares three objective configurations on the same instance:

1. **Baseline** — standard PyVRP cost (no custom weights).
2. **Fleet minimisation** — large vehicle-count penalty with
   :class:`~pyvrp.AdaptiveAdjustment` strategy.
3. **Balanced routes** — route-balance penalty with
   :class:`~pyvrp.MultiObjectiveScalarization` strategy.

Run:
    python examples/multi_objective_vrp.py
"""

from pathlib import Path

from pyvrp import read, solve
from pyvrp.adaptive_objective import (
    AdaptiveAdjustment,
    AdaptiveObjective,
    MultiObjectiveScalarization,
    ObjectiveWeights,
)
from pyvrp.IteratedLocalSearch import IteratedLocalSearchParams
from pyvrp.solve import SolveParams
from pyvrp.stop import MaxIterations

DATA = Path(__file__).parent.parent / "tests" / "data" / "RC208.vrp"
STOP = MaxIterations(3_000)
SEED = 0


def run(label: str, objective: AdaptiveObjective | None, seed: int) -> None:
    data = read(DATA, round_func="round")

    if objective is not None:
        params = SolveParams(
            ils=IteratedLocalSearchParams(callbacks=objective.as_callback()),
        )
    else:
        params = SolveParams()

    result = solve(data, stop=STOP, seed=seed, params=params)
    best = result.best

    rb = best.route_balance()
    print(
        f"{label:<30} | cost={result.cost():>10} | routes={best.num_routes():>3}"
        f" | feasible={best.is_feasible()} | balance_cv={rb:.3f}"
    )


def main() -> None:
    print(f"{'Config':<30} | {'cost':>10} | routes | feasible | balance_cv")
    print("-" * 70)

    run("Baseline", objective=None, seed=SEED)

    fleet_obj = AdaptiveObjective(
        initial_weights=ObjectiveWeights(vehicle_count=1000.0),
        strategy=AdaptiveAdjustment(
            target_feasibility=0.5,
            increase_factor=1.05,
            decrease_factor=0.95,
        ),
    )
    run("Fleet minimisation", fleet_obj, SEED)

    balance_obj = AdaptiveObjective(
        initial_weights=ObjectiveWeights(route_balance=500.0),
        strategy=MultiObjectiveScalarization(
            objectives=["vehicle_count", "route_balance"],
            scale=500.0,
            update_interval=300,
            rng_seed=SEED,
        ),
    )
    run("Balanced routes", balance_obj, SEED)


if __name__ == "__main__":
    main()
