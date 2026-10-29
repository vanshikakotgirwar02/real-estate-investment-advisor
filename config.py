CURRENT_YEAR = 2026
HORIZON_YEARS = 5

# growth rate = city base + infrastructure bonus - age penalty, kept between 4% and 14%
CITY_BASE_GROWTH = {"Bangalore": 0.10, "Hyderabad": 0.10, "Pune": 0.09, "Mumbai": 0.09,
                    "Delhi": 0.085, "Chennai": 0.08, "Kolkata": 0.07}
DEFAULT_BASE_GROWTH = 0.08
INFRA_BONUS_MAX = 0.03
AGE_PENALTY_PER_YEAR = 0.0005
GROWTH_MIN, GROWTH_MAX = 0.04, 0.14

GOOD_SCORE_THRESHOLD = 3       # good investment if it earns at least 3 of 5 points
MAX_AGE_FOR_POINT = 15