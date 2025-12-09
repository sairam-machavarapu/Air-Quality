# app.py
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
except Exception:
    xgb_available = False

# -----------------------------
# CONSTANTS
# -----------------------------
POLLUTANTS = [
    "PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2",
    "O3","Benzene","Toluene","Xylene","AQI"
]

# city coordinates (fallback list)
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
def load_dataset(folder="dataset"):
    """
    Load all CSV files from the given folder and return concatenated DataFrame and file list.
    Returns (df, files, errors)
    """
    errors = []
    if not os.path.exists(folder):
        return pd.DataFrame(), [], [f"Folder not found: {folder}"]

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not files:
        return pd.DataFrame(), files, [f"No CSV files found in {folder}"]

    dfs = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            df_tmp = pd.read_csv(path)
            if df_tmp.empty:
                errors.append(f"{f}: empty file")
                continue
            df_tmp["__source"] = f
            dfs.append(df_tmp)
        except Exception as e:
            errors.append(f"{f}: {str(e)}")

    if not dfs:
        return pd.DataFrame(), files, errors

    try:
        combined = pd.concat(dfs, ignore_index=True)
        return combined, files, errors
    except Exception as e:
        return pd.DataFrame(), files, [f"Failed concatenation: {e}"]

# -----------------------------
# PREPROCESS
# -----------------------------
@st.cache_data
def preprocess(df):
    """
    Clean dataset and return a cleaned DataFrame.
    Performs:
      - strip column names
      - require City and Date
      - parse Date
      - numeric conversion of known pollutant columns
      - per-city forward/backward fill and median fallback
      - extract Year/Month/Day/Weekday and City_Code
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Basic required columns
    if "City" not in df.columns or "Date" not in df.columns:
        # cannot proceed if City or Date are missing
        return pd.DataFrame()

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Trim spaces in City and drop rows lacking city or date
    df["City"] = df["City"].astype(str).str.strip()
    df = df.dropna(subset=["City", "Date"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    # Ensure pollutant columns exist and convert to numeric
    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)

    # Grouped imputation per city for the present pollutant columns
    if present:
        df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
        # fill any remaining NaNs with column median (global)
        for c in present:
            median = df[c].median(skipna=True)
            if pd.isna(median):
                median = 0.0
            df[c] = df[c].fillna(median)

    # Date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    # City codes
    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    # Reset index
    return df.reset_index(drop=True)


# -----------------------------
# PAGE: HOME
# -----------------------------
def page_home(df, files, errors):
    st.title("India Air Quality Explorer")
    if errors:
        st.warning("Some issues were found when loading files:")
        for e in errors:
            st.warning(f"- {e}")

    if df is None or df.empty:
        st.error("No valid dataset loaded. Please add CSV files to the `dataset/` folder with columns at least: City, Date, AQI (recommended).")
        st.markdown("**CSV tips:** column headers should include `City` and `Date` (YYYY-MM-DD or similar). Pollutant columns should match names like `PM2.5`, `PM10`, etc.")
        return

    st.success(f"Loaded {len(files)} file(s) — dataset has {len(df):,} rows and {len(df.columns)} columns.")
    st.write("Sample rows:")
    st.dataframe(df.head(10))

# -----------------------------
# PAGE: ABOUT
# -----------------------------
def page_about():
    st.title("About")
    st.markdown("""
    This app visualizes and predicts Indian air quality using open-source data.
    - Drop CSV files into the `dataset/` folder.
    - Required columns: `City`, `Date`.
    - Recommended: pollutant columns like `PM2.5`, `PM10`, `NO2`, `AQI`.
    """)
    st.markdown("Built with Streamlit — rewritten for robustness.")

# -----------------------------
# PAGE: DATA OVERVIEW
# -----------------------------
def page_data_overview(df):
    st.header("Data Overview")
    if df is None or df.empty:
        st.info("No data to show.")
        return

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        st.info("No numeric columns to describe.")
    else:
        st.dataframe(numeric.describe().T)

    st.subheader("First 100 rows")
    st.dataframe(df.head(100))

# -----------------------------
# PAGE: EDA
# -----------------------------
def page_eda(df):
    st.header("Exploratory Data Analysis")
    if df is None or df.empty:
        st.info("No data available for EDA.")
        return

    # ensure City column is categorical
    cities = ["All"] + sorted(df["City"].cat.categories.tolist())
    city_sel = st.sidebar.selectbox("City", cities)

    pollutants = [c for c in POLLUTANTS if c in df.columns]
    if not pollutants:
        st.info("No pollutant columns found in dataset.")
        return
    pollutant = st.sidebar.selectbox("Pollutant", pollutants, index=max(0, pollutants.index("AQI")) if "AQI" in pollutants else 0)

    df_f = df if city_sel == "All" else df[df["City"] == city_sel]

    if df_f.empty:
        st.warning("No rows for the selected city.")
        return

    # Trend: monthly mean
    try:
        monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().dropna()
    except Exception:
        monthly = df_f.groupby(["Year", "Month"])[pollutant].mean().reset_index()
        monthly["Date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(DAY=1))
        monthly = monthly.set_index("Date")[pollutant].sort_index()

    if monthly.empty:
        st.info("Not enough data to plot monthly trend for the selected pollutant.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_title(f"Monthly mean — {pollutant} ({city_sel})")
        ax.set_ylabel(pollutant)
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Show distribution
    st.subheader(f"Distribution of {pollutant}")
    vals = df_f[pollutant].dropna()
    if vals.empty:
        st.info("No values to show distribution.")
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(vals, kde=True, ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

# -----------------------------
# PAGE: MAPS
# -----------------------------
def page_maps(df):
    st.header("Geographical Maps")
    if df is None or df.empty:
        st.info("No data for maps.")
        return

    # Map requires Latitude/Longitude — pull from CITY_COORDS
    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [np.nan, np.nan])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [np.nan, np.nan])[1])

    df_geo = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if df_geo.empty:
        st.info("No geolocated rows (cities not found in CITY_COORDS). Update CITY_COORDS or add lat/lon to dataset.")
        return

    # ensure numeric
    df_geo["Latitude"] = pd.to_numeric(df_geo["Latitude"], errors="coerce")
    df_geo["Longitude"] = pd.to_numeric(df_geo["Longitude"], errors="coerce")
    if "AQI" in df_geo.columns:
        df_geo["AQI"] = pd.to_numeric(df_geo["AQI"], errors="coerce")
    else:
        df_geo["AQI"] = np.nan

    # Aggregate by city
    stats = df_geo.groupby("City").agg({
        "AQI": "mean",
        "Latitude": "first",
        "Longitude": "first"
    }).reset_index()

    # Map
    try:
        m = folium.Map(location=[22.97, 78.65], zoom_start=5)
        for _, r in stats.iterrows():
            try:
                aqi = r["AQI"]
                lat = float(r["Latitude"])
                lon = float(r["Longitude"])
                if pd.isna(aqi):
                    color = "blue"
                    popup = f"{r['City']}"
                    radius = 6
                else:
                    color = "green" if aqi <= 100 else "orange" if aqi <= 200 else "red"
                    popup = f"{r['City']} — AQI {aqi:.1f}"
                    radius = max(6, min(25, float(aqi) / 10))

                folium.CircleMarker(
                    [lat, lon],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=popup
                ).add_to(m)
            except Exception:
                # skip invalid row
                continue

        st_folium(m, width=900, height=500)
    except Exception as e:
        st.error(f"Failed to render folium map: {e}")

# -----------------------------
# PAGE: MODEL
# -----------------------------
def page_model(df):
    st.header("AQI Prediction Model")
    if df is None or df.empty:
        st.info("No data to train a model.")
        return

    if "AQI" not in df.columns:
        st.info("AQI column not found — model requires target column named 'AQI'.")
        return

    FEATURES = [c for c in POLLUTANTS if c != "AQI" and c in df.columns]
    # include basic date/city features
    for f in ["Year", "Month", "Day", "Weekday", "City_Code"]:
        if f in df.columns and f not in FEATURES:
            FEATURES.append(f)

    if not FEATURES:
        st.info("No features found to train on. Add pollutant columns like PM2.5, PM10, etc.")
        return

    X = df[FEATURES].copy()
    y = df["AQI"].copy()

    # Fill missing features with median (per column)
    X = X.fillna(X.median(axis=0))
    y = y.fillna(y.median())

    # Check for finite numeric arrays
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(axis=0))

    # Enough samples?
    if len(X) < 50:
        st.warning("Not enough rows to train a reliable model (need at least 50). Showing diagnostics only.")
        st.write("Available rows:", len(X))
        st.dataframe(df.head(20))
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate model
    if xgb_available:
        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model_name = "RandomForest"

    # Fit
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return

    # Predict
    try:
        preds = model.predict(X_test)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    col1, col2 = st.columns(2)
    col1.metric("Model", model_name)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R²", f"{r2:.3f}")

    # Show sample predictions
    sample = X_test.copy()
    sample["Actual_AQI"] = y_test.values
    sample["Predicted_AQI"] = np.round(preds, 2)
    st.subheader("Sample predictions (test set)")
    st.dataframe(sample.head(20))

# -----------------------------
# MAIN ROUTER
# -----------------------------
def main():
    st.sidebar.title("Navigation")

    data_folder = st.sidebar.text_input("Dataset folder", value="dataset")
    if not data_folder:
        data_folder = "dataset"

    df_raw, files, errors = load_dataset(data_folder)
    df = preprocess(df_raw)

    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Maps", "Model", "About"])

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
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
