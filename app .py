import streamlit as st
st.set_page_config(page_title="India Air Quality Explorer",
                   layout="wide",
                   initial_sidebar_state="expanded")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# POLLUTANTS
# =========================================================
POLLUTANTS = [
    "PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2",
    "O3","Benzene","Toluene","Xylene","AQI"
]

CITY_COORDS = {
    "Thiruvananthapuram":[8.5241,76.9366], "Shillong":[25.5788,91.8933],
    "Jaipur":[26.9124,75.7873], "Mumbai":[19.0760,72.8777],
    "Ernakulam":[9.9816,76.2999], "Guwahati":[26.1445,91.7362],
    "Aizawl":[23.7271,92.7176], "Delhi":[28.7041,77.1025],
    "Bengaluru":[12.9716,77.5946], "Visakhapatnam":[17.6868,83.2185],
    "Lucknow":[26.8467,80.9462], "Patna":[25.5941,85.1376],
    "Kochi":[9.9312,76.2673], "Gurugram":[28.4595,77.0266],
    "Coimbatore":[11.0168,76.9558], "Amaravati":[16.5414,80.5150],
    "Chandigarh":[30.7333,76.7794], "Amritsar":[31.6340,74.8723],
    "Jorapokhar":[23.8,86.4], "Talcher":[20.9497,85.2332],
    "Kolkata":[22.5726,88.3639], "Hyderabad":[17.3850,78.4867],
    "Ahmedabad":[23.0225,72.5714], "Chennai":[13.0827,80.2707],
    "Bhopal":[23.2599,77.4126], "Brajrajnagar":[21.8160,83.9008]
}

# =========================================================
# LOAD ALL CSVs FROM /dataset FOLDER
# =========================================================
@st.cache_data
def load_dataset():
    folder = "./dataset"
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]

    dfs = []
    for f in files:
        try:
            df_temp = pd.read_csv(os.path.join(folder, f))
            df_temp["__source"] = f
            dfs.append(df_temp)
        except:
            continue

    return pd.concat(dfs, ignore_index=True), files


# =========================================================
# PREPROCESS
# =========================================================
@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["City", "Date"]).sort_values(["City","Date"])

    present_cols = [c for c in POLLUTANTS if c in df.columns]
    df[present_cols] = df.groupby("City")[present_cols].transform(lambda g: g.ffill().bfill())

    for c in present_cols:
        df[c] = df[c].fillna(df[c].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)


# =========================================================
# PAGE 1 — HOME
# =========================================================
def page_home(df, files):
    st.title("India Air Quality — Streamlit App")

    st.header("Dataset Information")
    st.success("Dataset loaded automatically from GitHub repository `/dataset/` folder.")

    st.markdown("""
### 📘 About the Dataset
This dataset contains **daily air quality measurements** for multiple Indian cities.

Each file represents a city and includes pollutant concentrations for:
- **PM2.5, PM10**  
- **NO, NO2, NOx**  
- **NH3, CO, SO2**  
- **Ozone (O3)**  
- **Volatile Organic Compounds:** Benzene, Toluene, Xylene  
- **Air Quality Index (AQI)**  
- AQI Category (Good, Moderate, Poor, etc.)

The dataset is used for:
- Exploratory Data Analysis  
- Pollution pattern visualization  
- Geo-mapping of AQI  
- Machine Learning model to predict AQI  
""")

    st.subheader("Files Loaded")
    st.write(files[:50])

    st.subheader("Preview (First 200 rows)")
    st.dataframe(df.head(200))


# =========================================================
# PAGE 2 — DATA OVERVIEW
# =========================================================
def page_data_overview(df):
    st.header("Data Overview")

    st.subheader("Column Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").T)

    st.subheader("Sample Data")
    st.dataframe(df.head(200))


# =========================================================
# PAGE 3 — EDA
# =========================================================
def page_eda(df):
    st.header("Exploratory Data Analysis")

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("City", cities)
    pollutant = st.sidebar.selectbox("Pollutant", POLLUTANTS)

    date_min, date_max = df["Date"].min(), df["Date"].max()
    dr = st.sidebar.date_input("Date Range", [date_min, date_max])
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

    df_f = df[(df["Date"]>=start) & (df["Date"]<=end)]
    if city_sel != "All":
        df_f = df_f[df_f["City"]==city_sel]

    st.subheader(f"{pollutant} Distribution")
    st.write(df_f[pollutant].describe())

    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(monthly["Date"], monthly[pollutant], marker="o")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10,3))
    sns.boxplot(data=df_f, x="Month", y=pollutant, ax=ax2)
    st.pyplot(fig2)


# =========================================================
# PAGE 4 — MAPS
# =========================================================
def page_maps(df):
    st.header("AQI Maps")

    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None,None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None,None])[1])

    df_geo = df.dropna(subset=["Latitude","Longitude"])
    if df_geo.empty:
        st.error("No coordinate data found.")
        return

    means = df_geo.groupby("City")[["AQI","PM2.5","PM10"]].mean().reset_index()
    coords = df_geo.groupby("City")[["Latitude","Longitude"]].first().reset_index()
    stats = means.merge(coords, on="City")

    # AQI Bubble Map
    m = folium.Map(location=[22.97,78.65], zoom_start=5)

    for _, r in stats.iterrows():
        aqi = r["AQI"]
        lat, lon = r["Latitude"], r["Longitude"]

        if aqi<=100: color="green"
        elif aqi<=200: color="orange"
        else: color="red"

        folium.CircleMarker(
            [lat,lon],
            radius=max(6, min(25, aqi/10)),
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{r['City']} — AQI {aqi:.1f}"
        ).add_to(m)

    st.subheader("AQI Bubble Map")
    st_folium(m, width=900, height=500)

    # PM2.5 HeatMap
    hm = folium.Map(location=[22.97,78.65], zoom_start=5)
    heat_df = df_geo[["Latitude","Longitude","PM2.5"]].dropna()
    HeatMap(heat_df.values.tolist()).add_to(hm)

    st.subheader("PM2.5 Heatmap")
    st_folium(hm, width=900, height=500)


# =========================================================
# PAGE 5 — MODEL
# =========================================================
def page_model(df):
    st.header("AQI Prediction Model")

    usable = [c for c in POLLUTANTS if c in df.columns and c!="AQI"]
    usable += ["Year","Month","Day","Weekday","City_Code"]

    X = df[usable]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    # Predict section
    st.subheader("Manual AQI Prediction")

    inp = {}
    for col in usable:
        if col not in ["Year","Month","Day","Weekday","City_Code"]:
            inp[col] = st.number_input(col, value=float(df[col].median()))

    date = st.date_input("Date")
    city = st.selectbox("City", sorted(df["City"].unique()))
    city_code = int(df[df["City"] == city]["City_Code"].mode()[0])

    if st.button("Predict AQI"):
        row = [
            inp[c] if c in inp else None
            for c in usable
        ]

        # Add date features
        row[usable.index("Year")] = date.year
        row[usable.index("Month")] = date.month
        row[usable.index("Day")] = date.day
        row[usable.index("Weekday")] = date.weekday()
        row[usable.index("City_Code")] = city_code

        pred = model.predict(np.array(row).reshape(1,-1))[0]
        st.success(f"Predicted AQI: {pred:.1f}")


# =========================================================
# MAIN ROUTER
# =========================================================
def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Data Overview","EDA","Maps","Model"])

    if page=="Home": page_home(df, files)
    elif page=="Data Overview": page_data_overview(df)
    elif page=="EDA": page_eda(df)
    elif page=="Maps": page_maps(df)
    else: page_model(df)


if __name__ == "__main__":
    main()
