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
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False


# -----------------------------
# CONSTANTS
# -----------------------------
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


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_dataset(folder="./dataset"):
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df_tmp = pd.read_csv(os.path.join(folder, f))
            df_tmp["__source"] = f
            dfs.append(df_tmp)
        except Exception as e:
            st.warning(f"Failed reading {f}: {e}")

    if not dfs:
        return pd.DataFrame(), files

    return pd.concat(dfs, ignore_index=True), files


# -----------------------------
# PREPROCESS
# -----------------------------
@st.cache_data
def preprocess(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)

    df = df.dropna(subset=["City","Date"])
    df = df.sort_values(["City","Date"])

    # Impute per city
    df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
    for c in present:
        df[c] = df[c].fillna(df[c].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)


# -----------------------------
# HOME
# -----------------------------
def page_home(df, files):
    st.title("India Air Quality Explorer")
    st.success(f"Dataset loaded: {len(files)} files — {len(df):,} rows.")


# -----------------------------
# ABOUT
# -----------------------------
def page_about():
    st.title("About")
    st.markdown("""
    This app visualizes and predicts Indian air quality using open-source data.
    """)


# -----------------------------
# DATA OVERVIEW
# -----------------------------
def page_data_overview(df):
    st.header("Data Overview")
    st.write(df.describe().T)
    st.write(df.head(50))


# -----------------------------
# EDA
# -----------------------------
def page_eda(df):
    st.header("Exploratory Data Analysis")

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("City", cities)

    pollutants = [c for c in POLLUTANTS if c in df.columns]
    pollutant = st.sidebar.selectbox("Pollutant", pollutants)

    df_f = df if city_sel=="All" else df[df["City"]==city_sel]

    # Trend plot
    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly.index, monthly.values, marker="o")
    st.pyplot(fig)


# -----------------------------
# MAPS  (FIXED ERROR)
# -----------------------------
def page_maps(df):
    st.header("Geographical Maps")

    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c,[None,None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c,[None,None])[1])

    df_geo = df.dropna(subset=["Latitude","Longitude"])

    # Force numeric to avoid category-type aggregation error
    df_geo["Latitude"] = pd.to_numeric(df_geo["Latitude"], errors="coerce")
    df_geo["Longitude"] = pd.to_numeric(df_geo["Longitude"], errors="coerce")
    df_geo["AQI"] = pd.to_numeric(df_geo["AQI"], errors="coerce")

    # Correct aggregation
    stats = df_geo.groupby("City").agg({
        "AQI": "mean",
        "Latitude": "first",
        "Longitude": "first"
    }).reset_index()

    m = folium.Map(location=[22.97,78.65], zoom_start=5)

    for _, r in stats.iterrows():
        aqi = r["AQI"]
        lat, lon = r["Latitude"], r["Longitude"]

        color = "green" if aqi<=100 else "orange" if aqi<=200 else "red"

        folium.CircleMarker(
            [lat, lon],
            radius=max(6, min(25, aqi/10)),
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{r['City']} — AQI {aqi:.1f}"
        ).add_to(m)

    st_folium(m, width=900, height=500)


# -----------------------------
# MODEL (FIXED ERROR)
# -----------------------------
def page_model(df):
    st.header("AQI Prediction Model")

    FEATURES = [c for c in POLLUTANTS if c!="AQI" and c in df.columns]
    FEATURES += ["Year","Month","Day","Weekday","City_Code"]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["AQI"].fillna(df["AQI"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use XGBoost if available
    if xgb_available:
        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # *** FIXED RMSE (no squared flag) ***
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.3f}")


# -----------------------------
# MAIN ROUTER
# -----------------------------
def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Data Overview","EDA","Maps","Model","About"])

    if page=="Home": page_home(df, files)
    elif page=="Data Overview": page_data_overview(df)
    elif page=="EDA": page_eda(df)
    elif page=="Maps": page_maps(df)
    elif page=="Model": page_model(df)
    elif page=="About": page_about()

if __name__ == "__main__":
    main()
