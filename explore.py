import pandas as pd

df = pd.read_csv("data/india_housing_prices.csv")

print(df.shape)                      # rows, columns
print(df.head())                     # first 5 rows
print(df.dtypes)                     # type of each column
print(df.isna().sum())               # missing values per column
print(df.duplicated().sum())         # duplicate rows
for c in ["Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
          "Security", "Facing", "Owner_Type", "Availability_Status"]:
    print(c, df[c].unique()[:10])    # what values do these columns hold
print(df["Amenities"].head())        # what format are amenities in?

print(df["Parking_Space"].value_counts())                 
print(df["Amenities"].str.split(",").str.len().value_counts())  # how many amenities per row
print(df[["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
          "Nearby_Schools", "Nearby_Hospitals", "Age_of_Property"]].describe().round(2))
print(df[["City", "Locality"]].nunique())                     # how many cities / localities