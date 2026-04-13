#include "CostEvaluator.h"

#include <stdexcept>

using pyvrp::CostEvaluator;

CostEvaluator::CostEvaluator(std::vector<double> loadPenalties,
                             double twPenalty,
                             double distPenalty,
                             double vehicleCountWeight,
                             double routeBalanceWeight,
                             double distWeight,
                             double timeWeight)
    : loadPenalties_(std::move(loadPenalties)),
      twPenalty_(twPenalty),
      distPenalty_(distPenalty),
      vehicleCountWeight_(vehicleCountWeight),
      routeBalanceWeight_(routeBalanceWeight),
      distWeight_(distWeight),
      timeWeight_(timeWeight)
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

    if (distWeight_ < 0)
        throw std::invalid_argument("dist_weight must be >= 0.");

    if (timeWeight_ < 0)
        throw std::invalid_argument("time_weight must be >= 0.");
}

void CostEvaluator::setWeights(double vehicleCountWeight,
                               double routeBalanceWeight,
                               double distWeight,
                               double timeWeight)
{
    if (vehicleCountWeight < 0)
        throw std::invalid_argument("vehicle_count_weight must be >= 0.");
    if (routeBalanceWeight < 0)
        throw std::invalid_argument("route_balance_weight must be >= 0.");
    if (distWeight < 0)
        throw std::invalid_argument("dist_weight must be >= 0.");
    if (timeWeight < 0)
        throw std::invalid_argument("time_weight must be >= 0.");

    vehicleCountWeight_ = vehicleCountWeight;
    routeBalanceWeight_ = routeBalanceWeight;
    distWeight_ = distWeight;
    timeWeight_ = timeWeight;
}

std::tuple<double, double, double, double> CostEvaluator::getWeights() const
{
    return {vehicleCountWeight_, routeBalanceWeight_, distWeight_, timeWeight_};
}
