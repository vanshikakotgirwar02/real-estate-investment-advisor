import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/housing_processed.csv")


def save(name):
    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}.png", dpi=110)
    plt.close()


# Q1. Distribution of property prices
sns.histplot(df["Price_in_Lakhs"], bins=50, kde=True)
plt.title("Q1 Distribution of price (Lakhs)")
print("Q1 mean:", df["Price_in_Lakhs"].mean().round(1), "median:", df["Price_in_Lakhs"].median().round(1))
save("q01_price_distribution")

# Q2. Distribution of property sizes
sns.histplot(df["Size_in_SqFt"], bins=50, kde=True)
plt.title("Q2 Distribution of size (sq ft)")
save("q02_size_distribution")

# Q3. Price per sq ft by property type
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt")
plt.title("Q3 Price per sq ft by property type (Rs)")
print(df.groupby("Property_Type")["Price_per_SqFt"].median().round(0))
save("q03_pps_by_type")

# Q4. Size vs price
sns.scatterplot(data=df.sample(5000, random_state=1), x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3)
plt.title("Q4 Size vs price")
print("Q4 correlation:", df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]).round(3))
save("q04_size_vs_price")

# Q5. Outliers in price per sq ft and size
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df["Price_per_SqFt"], ax=ax[0])
sns.boxplot(x=df["Size_in_SqFt"], ax=ax[1])
q1, q3 = df["Price_per_SqFt"].quantile([0.25, 0.75])
n_out = (df["Price_per_SqFt"] > q3 + 1.5 * (q3 - q1)).sum()
print("Q5 outliers (1.5xIQR):", n_out)
save("q05_outliers")

# Q6. Average price per sq ft by state
s = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
s.plot.barh(figsize=(7, 7)); plt.title("Q6 Avg price per sq ft by state (Rs)")
print("Q6 highest:", s.index[-1], round(s.iloc[-1]), "| lowest:", s.index[0], round(s.iloc[0]))
save("q06_pps_by_state")

# Q7. Average property price by city
c = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
c.plot.barh(figsize=(7, 10)); plt.title("Q7 Avg price by city (Lakhs)")
print("Q7 highest:", c.index[-1], round(c.iloc[-1], 1), "| lowest:", c.index[0], round(c.iloc[0], 1))
save("q07_price_by_city")

# Q8. Median age of properties by locality (20 oldest shown)
a = df.groupby("Locality")["Age_of_Property"].median().sort_values().tail(20)
a.plot.barh(figsize=(7, 7)); plt.title("Q8 Median property age by locality (20 oldest)")
print("Q8 oldest locality median age:", a.iloc[-1])
save("q08_age_by_locality")

# Q9. BHK distribution across cities (share of each BHK in each city)
share = pd.crosstab(df["City"], df["BHK"], normalize="index")
plt.figure(figsize=(6, 12)); sns.heatmap(share, annot=True, fmt=".2f", cmap="Blues")
plt.title("Q9 BHK mix by city")
save("q09_bhk_by_city")

# Q10. Price trend for the 5 most expensive localities (by median price per sq ft)
top5 = df.groupby("Locality")["Price_per_SqFt"].median().nlargest(5).index
print("Q10 top 5:", list(top5))
t = df[df["Locality"].isin(top5)].groupby(["Year_Built", "Locality"])["Price_per_SqFt"].mean().reset_index()
sns.lineplot(data=t, x="Year_Built", y="Price_per_SqFt", hue="Locality", marker="o")
plt.title("Q10 Price per sq ft by build year, top 5 localities")
save("q10_top5_locality_trend")

# Q11. Correlation between numeric features
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Parking_Space", "Amenity_Count", "Infra_Score", "Growth_Rate"]
plt.figure(figsize=(10, 8)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Q11 Correlation heatmap")
corr = df[num].corr().abs().where(lambda x: x < 0.999).stack().nlargest(3)
print("Q11 strongest pairs:\n", corr)
save("q11_correlation")

# Q12. Nearby schools vs price per sq ft
sns.boxplot(data=df, x="Nearby_Schools", y="Price_per_SqFt")
plt.title("Q12 Nearby schools vs price per sq ft")
print("Q12 correlation:", df["Nearby_Schools"].corr(df["Price_per_SqFt"]).round(3))
save("q12_schools_vs_pps")

# Q13. Nearby hospitals vs price per sq ft
sns.boxplot(data=df, x="Nearby_Hospitals", y="Price_per_SqFt")
plt.title("Q13 Nearby hospitals vs price per sq ft")
print("Q13 correlation:", df["Nearby_Hospitals"].corr(df["Price_per_SqFt"]).round(3))
save("q13_hospitals_vs_pps")

# Q14. Price by furnished status
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs")
plt.title("Q14 Price by furnished status")
print("Q14 median price:\n", df.groupby("Furnished_Status")["Price_in_Lakhs"].median().round(1))
save("q14_price_by_furnishing")

# Q15. Price per sq ft by facing direction
sns.barplot(data=df, x="Facing", y="Price_per_SqFt", estimator="mean")
plt.title("Q15 Avg price per sq ft by facing")
print("Q15 mean by facing:\n", df.groupby("Facing")["Price_per_SqFt"].mean().round(0))
save("q15_pps_by_facing")

# Q16. Properties per owner type
df["Owner_Type"].value_counts().plot.bar(); plt.title("Q16 Properties by owner type")
print("Q16:", df["Owner_Type"].value_counts().to_dict())
save("q16_owner_type")

# Q17. Properties per availability status
df["Availability_Status"].value_counts().plot.bar(); plt.title("Q17 Properties by availability status")
print("Q17:", df["Availability_Status"].value_counts().to_dict())
save("q17_availability")

# Q18. Does parking affect price?
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs"); plt.title("Q18 Avg price by parking (0 = No, 1 = Yes)")
print("Q18 avg price by parking:\n", df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1))
save("q18_parking_vs_price")

# Q19. Do amenities affect price per sq ft?
sns.barplot(data=df, x="Amenity_Count", y="Price_per_SqFt"); plt.title("Q19 Avg price per sq ft by number of amenities")
print("Q19 correlation:", df["Amenity_Count"].corr(df["Price_per_SqFt"]).round(3))
save("q19_amenities_vs_pps")

# Q20. Public transport vs price per sq ft and investment potential
order = ["Low", "Medium", "High"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", order=order, ax=ax[0])
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=order, ax=ax[1])
ax[0].set_title("Q20 Price per sq ft"); ax[1].set_title("Q20 Share that are good investments")
print("Q20 good-investment share:\n", df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().round(3))
save("q20_transport")

# Extra A. Area vs investment return (brief: "correlation between area and investment return")
bands = pd.qcut(df["Size_in_SqFt"], 5)
print("Extra A good share by size band:\n", df.groupby(bands, observed=True)["Good_Investment"].mean().round(3))
sns.barplot(x=bands.astype(str), y=df["Good_Investment"]); plt.xticks(rotation=20)
plt.title("Extra A: size band vs share of good investments")
save("x1_size_vs_good_investment")

# Extra B. Infrastructure vs resale value (brief: "infrastructure score vs resale value")
sns.regplot(data=df.sample(5000, random_state=1), x="Infra_Score", y="Price_per_SqFt",
            scatter_kws={"alpha": 0.2}, line_kws={"color": "red"})
plt.title("Extra B: infrastructure score vs price per sq ft")
print("Extra B correlation:", df["Infra_Score"].corr(df["Price_per_SqFt"]).round(3))
save("x2_infra_vs_price")