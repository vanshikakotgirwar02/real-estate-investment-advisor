import pandas as pd
import config as C

TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}


def compute_stats(df):
    """Reference numbers learned from the data (the app needs these later)."""
    pps = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    stats = {
        "city_median_pps": pps.groupby(df["City"]).median().to_dict(),
        "global_median_pps": float(pps.median()),
        "schools_max": float(df["Nearby_Schools"].max()),
        "hospitals_max": float(df["Nearby_Hospitals"].max()),
    }
    stats["infra_median"] = float(engineer(df, stats | {"infra_median": 0})["Infra_Score"].median())
    return stats


def engineer(df, stats):
    """Add the new helpful columns."""
    df = df.copy()
    df["Age_of_Property"] = C.CURRENT_YEAR - df["Year_Built"]
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]        # Rs per sq ft
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
    df["Amenity_Count"] = df["Amenities"].str.split(",").str.len()
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
    df["School_Density_Score"] = df["Nearby_Schools"] / stats["schools_max"]
    hospitals = df["Nearby_Hospitals"] / stats["hospitals_max"]
    df["Infra_Score"] = (df["School_Density_Score"] + hospitals + (df["Transport_Score"] - 1) / 2) / 3
    city_median = df["City"].map(stats["city_median_pps"]).fillna(stats["global_median_pps"])
    df["Price_vs_City_Median"] = df["Price_per_SqFt"] / city_median
    return df


def make_targets(df, stats):
    """Create Good_Investment (yes/no) and Future_Price_Lakhs (number)."""
    df = df.copy()
    df["Invest_Score"] = (
        (df["Price_vs_City_Median"] <= 1).astype(int)
        + (df["BHK"] >= 3).astype(int)
        + (df["Availability_Status"] == "Ready_to_Move").astype(int)
        + (df["Infra_Score"] >= stats["infra_median"]).astype(int)
        + (df["Age_of_Property"] <= C.MAX_AGE_FOR_POINT).astype(int)
    )
    df["Good_Investment"] = (df["Invest_Score"] >= C.GOOD_SCORE_THRESHOLD).astype(int)

    base = df["City"].map(C.CITY_BASE_GROWTH).fillna(C.DEFAULT_BASE_GROWTH)
    rate = base + C.INFRA_BONUS_MAX * df["Infra_Score"] - C.AGE_PENALTY_PER_YEAR * df["Age_of_Property"].clip(lower=0)
    df["Growth_Rate"] = rate.clip(C.GROWTH_MIN, C.GROWTH_MAX)
    df["Future_Price_Lakhs"] = df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** C.HORIZON_YEARS
    return df

NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Floor_No",
           "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space",
           "Amenity_Count", "Transport_Score", "School_Density_Score", "Infra_Score", "Price_vs_City_Median"]

CATEGORICAL = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
               "Security", "Facing", "Owner_Type", "Availability_Status"]

FEATURES = NUMERIC + CATEGORICAL