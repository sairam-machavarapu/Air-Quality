# =========================================================
# app.py (Final Optimized Version)
# =========================================================

import streamlit as st
st.set_page_config(
    page_title="India Air Quality Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# =========================================================
# CONSTANTS & CITY COORDINATES
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
# DATA LOADING
# =========================================================

@st.cache_data(show_spinner=False)
def load_dataset(folder="./dataset"):
    """Load all CSVs from dataset folder."""
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df_temp = pd.read_csv(os.path.join(folder, f))
            df_temp["__source"] = f
            dfs.append(df_temp)
        except Exception as e:
            st.warning(f"Error reading {f}: {e}")
            continue

    if len(dfs) == 0:
        return pd.DataFrame(), files

    return pd.concat(dfs, ignore_index=True), files


# =========================================================
# PREPROCESSING
# =========================================================

@st.cache_data(show_spinner=False)
def preprocess(df):
    """Clean data, convert types, fill missing values, add date features."""
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert pollutants
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing city/date
    df = df.dropna(subset=["City", "Date"]).sort_values(["City", "Date"])

    # Fill missing per city
    present = [c for c in POLLUTANTS if c in df.columns]
    df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())

    for col in present:
        df[col] = df[col].fillna(df[col].median())

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    # City code
    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)


# =========================================================
# PAGE: HOME
# =========================================================

def page_home(df, files):
    st.title("India Air Quality Explorer")

    st.markdown("""
Welcome!  
This platform allows you to explore, visualize and model India's air quality data across multiple cities.

Upload your CSVs into `/dataset` folder.
""")

    if df.empty:
        st.error("No data found.")
    else:
        st.success(f"Loaded {len(files)} files • {len(df):,} total rows")
        st.write(files)


# =========================================================
# PAGE: ABOUT
# =========================================================

def page_about():
    st.title("About This Project")

    st.markdown("""
This dashboard provides complete analysis of multi-city air quality data.

### ✔ Features
- Pollutant-wise EDA  
- City trends & seasonal patterns  
- Geographical maps (AQI & PM2.5)  
- AQI Prediction using Machine Learning  
- Clean preprocessing pipeline  

### ✔ Pollutants Included
PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI

### ✔ Data Notes
- Missing values filled *per city*  
- Chronological ordering maintained  
- Numerical consistency ensured  
""")


# =========================================================
# PAGE: DATA OVERVIEW
# =========================================================

def page_data_overview(df):
    st.header("Data Overview")

    if df.empty:
        st.error("Dataset Empty")
        return

    st.subheader("Column Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe().T)


# =========================================================
# PAGE: EDA
# =========================================================

def page_eda(df):
    st.header("Exploratory Data Analysis")

    cities = ["All"] + sorted(df["City"].unique())
    pollutant = st.sidebar.selectbox("Pollutant", POLLUTANTS)
    city = st.sidebar.selectbox("City", cities)

    date_min, date_max = df["Date"].min(), df["Date"].max()
    dr = st.sidebar.date_input("Date Range", [date_min, date_max])
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

    df_f = df[(df["Date"] >= start) & (df["Date"] <= end)]
    if city != "All":
        df_f = df_f[df_f["City"] == city]

    # Summary
    st.subheader(f"{pollutant} Summary")
    st.write(df_f[pollutant].describe())

    # Monthly trend
    temp = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(temp.index, temp.values, marker="o")
    st.pyplot(fig)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.boxplot(data=df_f, x="Month", y=pollutant)
    st.pyplot(fig2)

    # Correlation
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_f[POLLUTANTS].corr(), cmap="coolwarm")
    st.pyplot(fig3)


# =========================================================
# PAGE: MAPS
# =========================================================

def page_maps(df):
    st.header("Geographical Maps")

    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None,None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None,None])[1])

    df_geo = df.dropna(subset=["Latitude","Longitude"])
    if df_geo.empty:
        st.error("No coordinate data available.")
        return

    map_type = st.sidebar.selectbox(
        "Map Type",
        ["Geographical AQI Map", "PM2.5 Heatmap", "City Marker Cluster"]
    )

    center = [22.97, 78.65]

    if map_type == "Geographical AQI Map":
        m = folium.Map(location=center, zoom_start=5)
        stats = df_geo.groupby("City")[["AQI","Latitude","Longitude"]].mean()

        for city, r in stats.iterrows():
            aqi, lat, lon = r["AQI"], r["Latitude"], r["Longitude"]
            if aqi <= 100: color = "green"
            elif aqi <= 200: color = "orange"
            else: color = "red"
            folium.CircleMarker(
                [lat, lon],
                radius=max(6, min(25, aqi/10)),
                color=color,
                fill=True,
                popup=f"{city} — AQI {aqi:.1f}"
            ).add_to(m)

        st_folium(m, width=900, height=500)

    elif map_type == "PM2.5 Heatmap":
        m = folium.Map(location=center, zoom_start=5)
        heat = df_geo[["Latitude", "Longitude", "PM2.5"]].dropna()
        HeatMap(heat.values.tolist()).add_to(m)
        st_folium(m, width=900, height=500)

    else:
        m = folium.Map(location=center, zoom_start=5)
        cluster = MarkerCluster().add_to(m)
        for _, r in df_geo.iterrows():
            folium.Marker(
                [r["Latitude"], r["Longitude"]],
                popup=f"{r['City']} — AQI {r['AQI']}"
            ).add_to(cluster)
        st_folium(m, width=900, height=500)


# =========================================================
# PAGE: MODEL (FASTEST + CLEAN)
# =========================================================

def page_model(df):
    st.header("AQI Prediction Model")

    pollutant_features = [c for c in POLLUTANTS if c != "AQI"]
    date_feats = ["Year", "Month", "Day", "Weekday"]
    FEATURES = pollutant_features + date_feats + ["City_Code"]

    # Prepare data
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(df.median(numeric_only=True))
    y = df["AQI"].fillna(df["AQI"].median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Downsampling if dataset is huge
    if len(X_train) > 20000:
        X_train = X_train.sample(20000, random_state=42)
        y_train = y_train.loc[X_train.index]

    # Train Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R²", f"{r2:.3f}")

    # Manual Prediction UI
    st.subheader("Manual AQI Prediction")

    with st.form("predict_form"):
        user_vals = {}
        for feat in pollutant_features:
            user_vals[feat] = st.number_input(feat, value=float(df[feat].median()))

        date = st.date_input("Date")
        city = st.selectbox("City", sorted(df["City"].unique()))
        city_code = int(df[df["City"] == city]["City_Code"].mode().iloc[0])

        submitted = st.form_submit_button("Predict")

        if submitted:
            row = [user_vals[f] for f in pollutant_features]
            row += [date.year, date.month, date.day, date.weekday(), city_code]

            pred_value = model.predict(np.array(row).reshape(1, -1))[0]
            st.success(f"Predicted AQI: {pred_value:.1f}")


# =========================================================
# MAIN ROUTER
# =========================================================

def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go To", ["Home", "Data Overview", "EDA", "Maps", "Model", "About"])

    if page == "Home":
        page_home(df, files)
    elif page == "Data Overview":
        page_data_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Maps":
        page_maps(df)
    elif page == "Model":
        page_model(df)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
