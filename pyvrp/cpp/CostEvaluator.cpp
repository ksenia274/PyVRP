#include "CostEvaluator.h"

#include <stdexcept>

using pyvrp::CostEvaluator;

CostEvaluator::CostEvaluator(std::vector<double> loadPenalties,
                             double twPenalty,
                             double distPenalty,
                             double vehicleCountWeight,
                             double routeBalanceWeight)
    : loadPenalties_(std::move(loadPenalties)),
      twPenalty_(twPenalty),
      distPenalty_(distPenalty),
      vehicleCountWeight_(vehicleCountWeight),
      routeBalanceWeight_(routeBalanceWeight)
{
    for (auto const penalty : loadPenalties_)
        if (penalty < 0)
            throw std::invalid_argument("load_penalties must be >= 0.");

    if (twPenalty_ < 0)
        throw std::invalid_argument("tw_penalty must be >= 0.");

    if (distPenalty_ < 0)
        throw std::invalid_argument("dist_penalty must be >= 0.");

    if (vehicleCountWeight_ < 0)
        throw std::invalid_argument("vehicle_count_weight must be >= 0.");

    if (routeBalanceWeight_ < 0)
        throw std::invalid_argument("route_balance_weight must be >= 0.");
}

void CostEvaluator::setWeights(double vehicleCountWeight,
                               double routeBalanceWeight)
{
    if (vehicleCountWeight < 0)
        throw std::invalid_argument("vehicle_count_weight must be >= 0.");
    if (routeBalanceWeight < 0)
        throw std::invalid_argument("route_balance_weight must be >= 0.");

    vehicleCountWeight_ = vehicleCountWeight;
    routeBalanceWeight_ = routeBalanceWeight;
}

std::tuple<double, double> CostEvaluator::getWeights() const
{
    return {vehicleCountWeight_, routeBalanceWeight_};
}
