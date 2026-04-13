"""
Adaptive VRP example
====================
Demonstrates how to use :class:`~pyvrp.AdaptiveObjective` with
:class:`~pyvrp.LinearDecay` to start with a strong vehicle-count penalty
that gradually fades as the search progresses.

Run:
    python examples/adaptive_vrp.py
"""

from pathlib import Path

from pyvrp import read, solve
from pyvrp.adaptive_objective import AdaptiveObjective, LinearDecay, ObjectiveWeights
from pyvrp.IteratedLocalSearch import IteratedLocalSearchParams
from pyvrp.solve import SolveParams
from pyvrp.stop import MaxIterations

DATA = Path(__file__).parent.parent / "tests" / "data" / "RC208.vrp"


def main() -> None:
    data = read(DATA, round_func="round")

    objective = AdaptiveObjective(
        initial_weights=ObjectiveWeights(vehicle_count=500.0, route_balance=100.0),
        strategy=LinearDecay(decay=0.9999, min_weight=0.0),
        history_window=200,
    )

    params = SolveParams(
        ils=IteratedLocalSearchParams(callbacks=objective.as_callback()),
    )

    result = solve(
        data,
        stop=MaxIterations(5_000),
        seed=42,
        collect_stats=True,
        display=True,
        params=params,
    )

    print(f"\nBest cost : {result.cost()}")
    print(f"Feasible  : {result.best.is_feasible()}")
    print(f"Routes    : {result.best.num_routes()}")
    print(f"Iterations: {result.num_iterations}")

    history = objective.get_history()
    if history:
        first, last = history[0], history[-1]
        print(f"\nWeight evolution (vehicle_count):")
        print(f"  iter {first.iteration:>6}: {first.weights.vehicle_count:.4f}")
        print(f"  iter {last.iteration:>6}: {last.weights.vehicle_count:.4f}")
        print(f"\nFinal feasibility rate: {last.feasibility_rate:.1%}")


if __name__ == "__main__":
    main()
