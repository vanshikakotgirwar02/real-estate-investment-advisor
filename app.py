import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import config as C
from features import engineer

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_assets():
    clf = joblib.load("models/classifier.joblib")
    reg = joblib.load("models/regressor.joblib")
    stats = json.load(open("models/stats.json"))
    opts = json.load(open("models/options.json"))
    metrics = json.load(open("models/metrics.json"))
    return clf, reg, stats, opts, metrics


@st.cache_data
def load_data():
    return pd.read_csv("data/housing_processed.csv")


clf, reg, stats, opts, metrics = load_assets()
data = load_data()

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is this property a good investment, and what could it be worth in 5 years?")

# ------------------------------ sidebar form ------------------------------
with st.sidebar:
    st.header("Property details")
    city = st.selectbox("City", opts["City"])
    ptype = st.selectbox("Property type", opts["Property_Type"])
    bhk = st.slider("BHK", 1, 5, 3)
    size = st.number_input("Size (sq ft)", 200, 10000, 1500, step=50)
    price = st.number_input("Asking price (Lakhs)", 5.0, 5000.0, 250.0, step=5.0)
    year = st.number_input("Year built", 1950, C.CURRENT_YEAR, 2015)
    availability = st.selectbox("Availability", opts["Availability_Status"])
    furnished = st.selectbox("Furnishing", opts["Furnished_Status"])
    total_floors = st.number_input("Total floors in building", 1, 80, 10)
    floor_no = st.number_input("Floor number", 0, 80, 3)
    schools = st.slider("Nearby schools", 1, int(stats["schools_max"]), 5)
    hospitals = st.slider("Nearby hospitals", 1, int(stats["hospitals_max"]), 5)
    transport = st.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)
    parking = st.selectbox("Parking space", ["Yes", "No"])
    security = st.selectbox("Security", opts["Security"])
    amenities = st.multiselect("Amenities (pick at least one)", opts["amenities"], default=opts["amenities"][:2])
    facing = st.selectbox("Facing", opts["Facing"])
    owner = st.selectbox("Owner type", opts["Owner_Type"])
    go = st.button("Analyze property", type="primary", width="stretch")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Market insights", "🔎 Explore listings", "🧪 Model performance"])

# ------------------------------ tab 1: prediction ------------------------------
with tab1:
    if not go:
        st.info("Fill in the property details on the left and press **Analyze property**.")
    else:
        row = pd.DataFrame([{
            "State": opts["City_to_State"].get(city, ""), "City": city, "Property_Type": ptype, "BHK": bhk,
            "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year, "Furnished_Status": furnished,
            "Floor_No": min(floor_no, total_floors), "Total_Floors": total_floors,
            "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport,
            "Parking_Space": 1 if parking == "Yes" else 0,          # same 1/0 coding as training
            "Security": security, "Amenities": ", ".join(amenities) or "Gym", "Facing": facing,
            "Owner_Type": owner, "Availability_Status": availability,
        }])
        feats = engineer(row, stats)
        p_good = float(clf.predict_proba(feats)[0, 1])
        future = float(reg.predict(feats)[0])
        growth = (future / price) ** (1 / C.HORIZON_YEARS) - 1
        city_median = stats["city_median_pps"].get(city, stats["global_median_pps"])
        pps = float(feats["Price_per_SqFt"].iloc[0])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Investment verdict")
            if p_good >= 0.5:
                st.success("✅ Good Investment")
            else:
                st.error("⚠️ Not a strong investment")
            st.metric("Model confidence", f"{max(p_good, 1 - p_good):.0%}")
            st.progress(p_good, text=f"Probability of 'good': {p_good:.0%}")
        with c2:
            st.subheader(f"Estimated price in {C.HORIZON_YEARS} years")
            st.metric("Future price", f"₹{future:,.1f} L", f"+₹{future - price:,.1f} L vs today")
            st.metric("Implied yearly growth", f"{growth:.1%}")
        with c3:
            st.subheader("Price check")
            st.metric("Your price per sq ft", f"₹{pps:,.0f}", f"{pps / city_median - 1:+.0%} vs {city} median",
                      delta_color="inverse")
            st.caption(f"{city} median is ₹{city_median:,.0f} per sq ft.")

        path = [price * (future / price) ** (t / C.HORIZON_YEARS) for t in range(C.HORIZON_YEARS + 1)]
        st.plotly_chart(px.line(x=list(range(C.HORIZON_YEARS + 1)), y=path, markers=True,
                                labels={"x": "Years from now", "y": "Estimated value (Lakhs)"},
                                title="Projected value path"), width="stretch")

        st.subheader("What drives the models?")
        f1, f2 = st.columns(2)
        for col, file, title in [(f1, "importance_clf.csv", "Classifier: top features"),
                                 (f2, "importance_reg.csv", "Price model: top features")]:
            imp = pd.read_csv(f"models/{file}").head(10).iloc[::-1]
            imp["feature"] = imp["feature"].str.replace("num__", "").str.replace("cat__", "")
            col.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h", title=title), width="stretch")
        st.caption("Estimates come from models trained on this dataset plus stated growth assumptions. Not financial advice.")

# ------------------------------ tab 2: market insights ------------------------------
with tab2:
    st.subheader("Average price per sq ft: city × property type (₹)")
    top_cities = data["City"].value_counts().head(15).index
    heat = (data[data["City"].isin(top_cities)]
            .pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean").round(0))
    st.plotly_chart(px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd"), width="stretch")
    st.caption("Showing the 15 cities with the most listings.")

    a, b = st.columns(2)
    chosen = a.multiselect("Cities for trend chart", sorted(data["City"].unique()), default=sorted(data["City"].unique())[:3])
    trend = data[data["City"].isin(chosen)].groupby(["Year_Built", "City"])["Price_per_SqFt"].mean().reset_index()
    a.plotly_chart(px.line(trend, x="Year_Built", y="Price_per_SqFt", color="City", title="Price per sq ft by build year"),
                   width="stretch")
    share = data.groupby("City")["Good_Investment"].mean().sort_values().tail(15).reset_index()
    b.plotly_chart(px.bar(share, x="Good_Investment", y="City", orientation="h",
                          title="Share of good-investment listings (top 15 cities)"), width="stretch")

# ------------------------------ tab 3: explore listings ------------------------------
with tab3:
    st.subheader("Filter existing listings")
    f1, f2, f3, f4 = st.columns(4)
    cities = f1.multiselect("City", sorted(data["City"].unique()))
    bhks = f2.multiselect("BHK", sorted(data["BHK"].unique()))
    lo, hi = float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())
    prange = f3.slider("Price (Lakhs)", lo, hi, (lo, hi))
    only_good = f4.checkbox("Only 'good investment'")
    view = data[data["Price_in_Lakhs"].between(*prange)]
    if cities:
        view = view[view["City"].isin(cities)]
    if bhks:
        view = view[view["BHK"].isin(bhks)]
    if only_good:
        view = view[view["Good_Investment"] == 1]
    st.write(f"**{len(view):,}** properties match")
    st.dataframe(view[["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                       "Availability_Status", "Good_Investment", "Future_Price_Lakhs"]].head(500), width="stretch")

# ------------------------------ tab 4: model performance ------------------------------
with tab4:
    st.subheader("Classification (Good Investment)")
    st.dataframe(pd.DataFrame(metrics["classification"]).T.round(4), width="stretch")
    st.subheader("Regression (Price after 5 years)")
    st.dataframe(pd.DataFrame(metrics["regression"]).T.round(4), width="stretch")
    st.caption(f"Deployed models: {metrics['best_classifier']} (classifier), {metrics['best_regressor']} (regressor).")