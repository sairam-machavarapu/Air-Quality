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

# Try Importing XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

# -----------------------
# CONSTANTS
# -----------------------
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

# -----------------------
# DATA LOADING
# -----------------------
@st.cache_data
def load_dataset(folder="./dataset"):
    """Loads ALL CSV files inside ./dataset folder."""
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df_temp = pd.read_csv(os.path.join(folder, f))
            df_temp["__source"] = f
            dfs.append(df_temp)
        except:
            continue

    if len(dfs) == 0:
        return pd.DataFrame(), files

    return pd.concat(dfs, ignore_index=True), files


# -----------------------
# PREPROCESSING
# -----------------------
@st.cache_data
def preprocess(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert pollutants to numeric
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["City", "Date"]).sort_values(["City", "Date"])

    # Fill missing values inside each city
    present = [c for c in POLLUTANTS if c in df.columns]
    df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())

    for col in present:
        df[col] = df[col].fillna(df[col].median())

    # Time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)


# -----------------------
# TRAIN CACHED XGBOOST
# -----------------------
@st.cache_resource
def train_xgb_model(X_train, y_train):
    """Train XGBoost once and reuse cached model."""
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X_train, y_train)
    return model


# -----------------------
# HOME PAGE
# -----------------------
def page_home(df, files):
    st.title("India Air Quality Explorer")

    st.markdown("""
### 🌍 Welcome

This dashboard allows you to:

- Analyze pollutant trends  
- Compare multiple cities  
- Visualize geospatial pollution  
- Predict AQI using Machine Learning (XGBoost)

CSV files must be placed inside the `./dataset` folder.
""")

    if df.empty:
        st.error("No dataset found in ./dataset/")
    else:
        st.success(f"Loaded {len(files)} files, Total rows: {len(df):,}")


# -----------------------
# ABOUT PAGE
# -----------------------
def page_about():
    st.title("About This Project")

    st.markdown("""
## 🌍 India Air Quality Monitoring & Prediction

This dashboard provides end-to-end analysis of India’s air quality, including:

### ✔ Preprocessing & Cleaning  
- City-wise gap filling  
- Temporal feature generation  
- Numeric conversion of pollutant levels  

### ✔ EDA (Exploratory Data Analysis)  
- Seasonal trends  
- Monthly distribution  
- City-wise comparisons  
- Correlation heatmaps  

### ✔ Maps  
- Geographical AQI map  
- PM2.5 heatmap  
- Marker clusters  

### ✔ Machine Learning  
- XGBoost-based AQI prediction  
- Feature engineering  
- Manual prediction section  

### Dataset includes pollutants:
PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3  
Benzene, Toluene, Xylene + AQI

Built for academic and research use.
""")


# -----------------------
# DATA OVERVIEW
# -----------------------
def page_data_overview(df):
    st.header("Data Overview")
    if df.empty:
        st.error("No data to show.")
        return

    st.subheader("Column Types")
    st.write(df.dtypes)

    st.subheader("Numeric Summary")
    st.write(df.describe().T)


# -----------------------
# EDA PAGE
# -----------------------
def page_eda(df):
    st.header("Exploratory Data Analysis")

    if df.empty:
        st.error("Dataset Empty")
        return

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("City", cities)
    pollutant = st.sidebar.selectbox("Pollutant", POLLUTANTS)

    # Date Range
    date_min, date_max = df["Date"].min(), df["Date"].max()
    dr = st.sidebar.date_input("Date Range", [date_min, date_max])

    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    df_f = df[(df["Date"] >= start) & (df["Date"] <= end)]

    if city_sel != "All":
        df_f = df_f[df_f["City"] == city_sel]

    st.subheader(f"{pollutant} Summary")
    st.write(df_f[pollutant].describe())

    # Monthly Trend
    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(monthly["Date"], monthly[pollutant], marker="o")
    st.pyplot(fig)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.boxplot(data=df_f, x="Month", y=pollutant, ax=ax2)
    st.pyplot(fig2)


# -----------------------
# MAP PAGE
# -----------------------
def page_maps(df):
    st.header("Geographical Maps")

    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[1])

    df_geo = df.dropna(subset=["Latitude", "Longitude"])
    if df_geo.empty:
        st.error("No coordinate-enabled city data.")
        return

    map_type = st.sidebar.selectbox(
        "Map Type",
        ["Geographical AQI Map", "PM2.5 Heatmap", "City Marker Cluster"]
    )

    means = df_geo.groupby("City")[["AQI","PM2.5","PM10"]].mean().reset_index()
    coords = df_geo.groupby("City")[["Latitude","Longitude"]].first().reset_index()
    stats = means.merge(coords, on="City")

    center = [22.97, 78.65]

    if map_type == "Geographical AQI Map":
        m = folium.Map(location=center, zoom_start=5)
        for _, r in stats.iterrows():
            aqi = r["AQI"]
            lat, lon = r["Latitude"], r["Longitude"]
            if aqi <= 100: color = "green"
            elif aqi <= 200: color = "orange"
            else: color = "red"

            folium.CircleMarker(
                [lat, lon],
                radius=max(6, min(25, aqi/10)),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{r['City']}: AQI {aqi:.1f}"
            ).add_to(m)
        st_folium(m, width=900, height=500)

    elif map_type == "PM2.5 Heatmap":
        m = folium.Map(location=center, zoom_start=5)
        heat_df = df_geo[["Latitude","Longitude","PM2.5"]].dropna()
        HeatMap(heat_df.values.tolist()).add_to(m)
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


# -----------------------
# MODEL PAGE (XGBOOST)
# -----------------------
def page_model(df):
    st.header("AQI Prediction Model (XGBoost)")

    pollutant_features = [c for c in POLLUTANTS if c in df.columns and c != "AQI"]
    FEATURES = pollutant_features + ["Year","Month","Day","Weekday", "City_Code"]

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(df.median(numeric_only=True))
    y = pd.to_numeric(df["AQI"], errors="coerce").fillna(df["AQI"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if xgb_available:
        st.info("Training (cached) XGBoost Model…")
        model = train_xgb_model(X_train, y_train)
    else:
        st.warning("XGBoost unavailable. Using RandomForest fallback.")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.3f}")

    # Manual Prediction
    st.subheader("Predict AQI for Custom Inputs")

    with st.form("predict_form"):
        user_inputs = {}
        for p in pollutant_features:
            user_inputs[p] = st.number_input(p, value=float(df[p].median()))

        date = st.date_input("Sample Date")
        city = st.selectbox("City", sorted(df["City"].unique()))
        city_code = int(df[df["City"] == city]["City_Code"].mode()[0])

        submit = st.form_submit_button("Predict")
        if submit:
            row = [user_inputs[p] for p in pollutant_features]
            row += [date.year, date.month, date.day, date.weekday(), city_code]

            pred = model.predict(np.array(row).reshape(1, -1))[0]
            st.success(f"Predicted AQI: {pred:.1f}")


# -----------------------
# MAIN ROUTER
# -----------------------
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
