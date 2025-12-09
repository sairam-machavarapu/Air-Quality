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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

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

# ====================================================
# LOAD DATA
# ====================================================
@st.cache_data
def load_dataset(folder="dataset"):
    if not os.path.exists(folder):
        return pd.DataFrame(), [], ["Dataset folder not found"]

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    errors = []
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(folder, f))
            df["__source"] = f
            dfs.append(df)
        except Exception as e:
            errors.append(f"Failed load {f}: {e}")

    if not dfs:
        return pd.DataFrame(), files, errors

    try:
        return pd.concat(dfs, ignore_index=True), files, errors
    except Exception as e:
        return pd.DataFrame(), files, [f"Concat error: {e}"]

# ====================================================
# PREPROCESS
# ====================================================
@st.cache_data
def preprocess(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()

    if "City" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["City"] = df["City"].astype(str).str.strip()
    df = df.dropna(subset=["City", "Date"])
    if df.empty:
        return df

    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
            present.append(c)

    # Impute per city
    if present:
        df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
        for col in present:
            df[col] = df[col].fillna(df[col].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)

# ====================================================
# MODEL TRAINING (CACHED) — SUPER FAST
# ====================================================
@st.cache_resource
def train_cached_model(X_train, y_train, use_xgb):
    if use_xgb:
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    return model

# ====================================================
# HOME PAGE
# ====================================================
def page_home(df, files, errors):
    st.title("India Air Quality Explorer")

    if errors:
        st.warning("Some files had issues:")
        for e in errors:
            st.write(f"- {e}")

    if df.empty:
        st.error("Dataset is empty or invalid!")
        return

    st.success(f"Loaded {len(files)} files — {len(df):,} rows")
    st.dataframe(df.head(10))

# ====================================================
# DATA OVERVIEW
# ====================================================
def page_data_overview(df):
    st.header("Data Overview")

    if df.empty:
        st.warning("No data loaded.")
        return

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Sample Rows")
    st.dataframe(df.head(50))

# ====================================================
# EDA (Enhanced)
# ====================================================
def page_eda(df):
    st.header("Exploratory Data Analysis")

    if df.empty:
        st.warning("No data available.")
        return

    cities = ["All"] + list(df["City"].cat.categories)
    city_sel = st.sidebar.selectbox("City", cities)

    pollutants = [c for c in POLLUTANTS if c in df.columns]
    pollutant = st.sidebar.selectbox("Pollutant", pollutants)

    df_f = df if city_sel == "All" else df[df["City"] == city_sel]

    # ----------------- MONTHLY TREND ------------------
    st.subheader("📅 Monthly Trend")

    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.grid(True)
    st.pyplot(fig)

    # ----------------- YEARLY TREND ------------------
    st.subheader("📆 Yearly Trend")

    yearly = df_f.groupby("Year")[pollutant].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index, yearly.values)
    st.pyplot(fig)

    # ----------------- MONTH-WISE ------------------
    st.subheader("🌤️ Seasonal Pattern")

    monthwise = df_f.groupby("Month")[pollutant].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthwise.index, monthwise.values, marker="o")
    ax.set_xticks(range(1, 13))
    st.pyplot(fig)

    # ----------------- DISTRIBUTION ------------------
    st.subheader("📊 Distribution & Boxplot")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_f[pollutant], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df_f[pollutant], ax=ax)
        st.pyplot(fig)

    # ----------------- CORRELATION ------------------
    st.subheader("📈 Correlation Heatmap")

    num_df = df_f.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ====================================================
# MAPS
# ====================================================
def page_maps(df):
    st.header("Geographical AQI Map")

    if df.empty:
        st.warning("No data available.")
        return

    df["Lat"] = df["City"].map(lambda x: CITY_COORDS.get(x, [None, None])[0])
    df["Lon"] = df["City"].map(lambda x: CITY_COORDS.get(x, [None, None])[1])

    df_geo = df.dropna(subset=["Lat", "Lon"])

    m = folium.Map(location=[20.59, 78.96], zoom_start=5)

    for _, r in df_geo.groupby("City").agg({"AQI":"mean", "Lat":"first", "Lon":"first"}).iterrows():
        aqi = r["AQI"]
        color = "green" if aqi <= 100 else "orange" if aqi <= 200 else "red"

        folium.CircleMarker(
            location=[r["Lat"], r["Lon"]],
            radius=10,
            popup=f"{_}: {aqi:.1f}",
            color=color,
            fill=True,
        ).add_to(m)

    st_folium(m, width=900, height=500)

# ====================================================
# MODEL (FAST)
# ====================================================
def page_model(df):
    st.header("AQI Prediction Model")

    if df.empty:
        st.warning("No data to train model.")
        return

    if "AQI" not in df.columns:
        st.error("Dataset must contain AQI column.")
        return

    FEATURES = [c for c in POLLUTANTS if c != "AQI" and c in df.columns]
    FEATURES += ["Year","Month","Day","Weekday","City_Code"]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------ CACHED TRAINING ------------------
    model = train_cached_model(X_train, y_train, xgb_available)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.3f}")

    st.subheader("Sample Predictions")
    out = X_test.copy()
    out["Actual AQI"] = y_test.values
    out["Predicted AQI"] = np.round(preds, 2)
    st.dataframe(out.head(20))

# ====================================================
# MAIN ROUTER
# ====================================================
def main():
    st.sidebar.title("Navigation")

    df_raw, files, errors = load_dataset("dataset")
    df = preprocess(df_raw)

    page = st.sidebar.radio("Go to", ["Home","Data Overview","EDA","Maps","Model","About"])

    if page == "Home":
        page_home(df, files, errors)
    elif page == "Data Overview":
        page_data_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Maps":
        page_maps(df)
    elif page == "Model":
        page_model(df)
    else:
        st.header("About")
        st.write("Created by You — enhanced with ML & EDA features.")

if __name__ == "__main__":
    main()
