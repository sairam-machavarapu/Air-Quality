# app.py (optimized)
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

# -------------------------
# Constants
# -------------------------
POLLUTANTS = [
    "PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2",
    "O3","Benzene","Toluene","Xylene","AQI"
]

# City coordinates used for maps (extend if you have more cities)
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

# -------------------------
# Data load / preprocess
# -------------------------
@st.cache_data(show_spinner=False)
def load_dataset(folder="./dataset"):
    """Read all CSV files in folder and return (df, filenames)."""
    if not os.path.exists(folder):
        return pd.DataFrame(), []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    dfs = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            df_temp = pd.read_csv(path)
            df_temp["__source"] = f
            dfs.append(df_temp)
        except Exception as e:
            # skip problematic files but log in the app
            st.warning(f"Failed to read {f}: {e}")
            continue
    if len(dfs) == 0:
        return pd.DataFrame(), files
    df = pd.concat(dfs, ignore_index=True)
    return df, files

@st.cache_data(show_spinner=False)
def preprocess(df):
    """Clean columns, convert types, fill missing pollutant values per city and add date features."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # strip column names
    df.columns = df.columns.str.strip()
    # unify Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # convert pollutant columns to numeric if present
    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)
    # drop rows without City or Date (essential)
    if "City" in df.columns and "Date" in df.columns:
        df = df.dropna(subset=["City", "Date"])
    # sort for proper forward/backfill
    sort_cols = [c for c in ["City", "Date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    # forward/backfill per city for present pollutant columns
    if present and "City" in df.columns:
        df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
        # any remaining NA -> fill with median of column
        for c in present:
            df[c] = df[c].fillna(df[c].median())
    # date features
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday
    # city code
    if "City" in df.columns:
        df["City"] = df["City"].astype("category")
        df["City_Code"] = df["City"].cat.codes
    return df.reset_index(drop=True)

# -------------------------
# Pages
# -------------------------
def page_home(df, files):
    """Home / Data Upload page - minimal and informative."""
    st.title("India Air Quality — Explorer")
    st.markdown(
        """
        **Welcome.** Use the sidebar to navigate the app.

        This dashboard analyses daily air-quality measurements across Indian cities.
        Upload CSV files into the `./dataset` folder (each file per city) or place them
        in your repo / deployment folder before running the app.
        """
    )
    st.subheader("Dataset status")
    if df is None or df.empty:
        st.error("No data found. Please add CSV files to the `./dataset` folder.")
    else:
        st.success(f"Loaded {len(files)} file(s) — total rows: {len(df):,}")
        st.markdown("**Files:**")
        st.write(files[:200])  # small list only

def page_about():
    """About page with project description and dataset notes."""
    st.title("About This Project")
    st.markdown("""
    ## India Air Quality Monitoring & Prediction

    This project provides tools to explore, visualise and model air-quality
    measurements across Indian cities.

    **Features**
    - Exploratory Data Analysis (per pollutant and per city)
    - Interactive Geographical Maps (AQI & PM2.5)
    - Machine Learning models for AQI prediction (Random Forest baseline)
    - Clear preprocessing with city-wise imputation and temporal features

    **Dataset**
    - Daily pollutant readings across many cities
    - Pollutants included: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI

    **Notes**
    - Missing pollutant values are imputed **within each city** using forward/backward fill,
      then column medians for any remaining gaps.
    - The app focuses on analysis and reproducibility; cite data sources when using results.
    """)

def page_data_overview(df):
    st.header("Data Overview")
    if df is None or df.empty:
        st.error("No data available.")
        return
    st.subheader("Columns & Types")
    st.write(df.dtypes)
    st.subheader("Summary statistics (numeric columns)")
    st.write(df.describe().T)

def page_eda(df):
    st.header("Exploratory Data Analysis")
    if df is None or df.empty:
        st.error("No data to analyze.")
        return

    # choose city & pollutant (only columns that exist)
    cities = ["All"] + sorted(df["City"].unique().tolist())
    city_sel = st.sidebar.selectbox("City", cities, index=0)
    available_pollutants = [c for c in POLLUTANTS if c in df.columns]
    if not available_pollutants:
        st.error("No pollutant columns found in the dataset.")
        return
    pollutant = st.sidebar.selectbox("Pollutant", available_pollutants, index=0)

    # date range
    date_min, date_max = df["Date"].min(), df["Date"].max()
    dr = st.sidebar.date_input("Date range", [date_min.date(), date_max.date()])
    if len(dr) != 2:
        st.warning("Please select a start and end date.")
        return
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

    # filter
    df_f = df[(df["Date"] >= start) & (df["Date"] <= end)]
    if city_sel != "All":
        df_f = df_f[df_f["City"] == city_sel]

    st.subheader(f"{pollutant} — Summary")
    st.write(df_f[pollutant].describe())

    # monthly trend
    with st.expander("Monthly trend"):
        monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(monthly["Date"], monthly[pollutant], marker="o", linewidth=2)
        ax.set_title(f"Monthly average — {pollutant}")
        ax.set_xlabel("Month")
        ax.set_ylabel(pollutant)
        st.pyplot(fig)

    # monthly boxplot
    with st.expander("Monthly distribution (boxplot)"):
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        sns.boxplot(data=df_f, x="Month", y=pollutant, ax=ax2)
        ax2.set_title(f"Monthly distribution — {pollutant}")
        st.pyplot(fig2)

    # correlation small view
    with st.expander("Correlation (selected pollutants)"):
        cols_for_corr = [c for c in available_pollutants if c in df.columns]
        corr = df[cols_for_corr].corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax3)
        ax3.set_title("Pollutant correlation")
        st.pyplot(fig3)

def page_maps(df):
    st.header("Geographical Maps")

    if df is None or df.empty:
        st.error("No data available for maps.")
        return

    # attach coordinates where available
    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[1])

    df_geo = df.dropna(subset=["Latitude", "Longitude"])
    if df_geo.empty:
        st.error("No coordinate data found for any city. Please add coordinates to CITY_COORDS.")
        return

    # map type
    map_type = st.sidebar.selectbox(
        "Map type",
        ["Geographical AQI Map", "PM2.5 Heatmap", "City Marker Cluster"]
    )

    # compute city aggregates
    agg_cols = []
    for c in ["AQI", "PM2.5", "PM10"]:
        if c in df_geo.columns:
            agg_cols.append(c)
    means = df_geo.groupby("City")[agg_cols].mean().reset_index()
    coords = df_geo.groupby("City")[["Latitude", "Longitude"]].first().reset_index()
    stats = means.merge(coords, on="City", how="inner")

    # base map centered on India
    m_center = [22.97, 78.65]
    m_zoom = 5

    if map_type == "Geographical AQI Map":
        m = folium.Map(location=m_center, zoom_start=m_zoom)
        for _, r in stats.iterrows():
            aqi = r.get("AQI", np.nan)
            lat, lon = float(r["Latitude"]), float(r["Longitude"])
            # color scale
            if np.isnan(aqi):
                color = "gray"
            elif aqi <= 100:
                color = "green"
            elif aqi <= 200:
                color = "orange"
            else:
                color = "red"
            radius = max(6, min(25, (aqi if np.isfinite(aqi) else 50) / 10))
            folium.CircleMarker(
                [lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{r['City']}: AQI {aqi:.1f}" if np.isfinite(aqi) else f"{r['City']}: AQI N/A"
            ).add_to(m)
        st.subheader("Geographical AQI Map")
        st_folium(m, width=900, height=500)

    elif map_type == "PM2.5 Heatmap":
        if "PM2.5" not in df_geo.columns:
            st.error("PM2.5 column not present in the dataset.")
            return
        hm = folium.Map(location=m_center, zoom_start=m_zoom)
        heat_df = df_geo[["Latitude", "Longitude", "PM2.5"]].dropna()
        heat_data = [[float(r[0]), float(r[1]), float(r[2])] for r in heat_df.values]
        if len(heat_data):
            HeatMap(heat_data, radius=12, blur=15).add_to(hm)
        st.subheader("PM2.5 Heatmap")
        st_folium(hm, width=900, height=500)

    else:  # Marker Cluster
        m = folium.Map(location=m_center, zoom_start=m_zoom)
        cluster = MarkerCluster().add_to(m)
        for _, r in df_geo.iterrows():
            lat, lon = float(r["Latitude"]), float(r["Longitude"])
            aqi = r.get("AQI", None)
            popup = f"{r.get('City','')}"
            if aqi is not None and not pd.isna(aqi):
                popup += f" — AQI: {aqi:.1f}"
            folium.Marker(location=[lat, lon], popup=popup).add_to(cluster)
        st.subheader("City Marker Cluster")
        st_folium(m, width=900, height=500)

def page_model(df):
    st.header("AQI Prediction Model")
    if df is None or df.empty:
        st.error("No data for modeling.")
        return

    # select features that exist in the dataset
    pollutant_features = [c for c in POLLUTANTS if c in df.columns and c != "AQI"]
    date_feats = ["Year", "Month", "Day", "Weekday"]
    optional_feats = ["City_Code"] if "City_Code" in df.columns else []
    FEATURES = pollutant_features + date_feats + optional_feats

    if len(pollutant_features) == 0:
        st.error("No pollutant features available for training.")
        return

    # prepare X, y and handle missing values
    X = df[FEATURES].copy()
    y = df["AQI"].copy()

    # numeric ensure
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # fill remaining NaNs with median (safe for tree models and linear)
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model train & metrics with try/except for robustness
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("R²", f"{r2:.3f}")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return

    # single-sample prediction UI
    st.subheader("Manual AQI Prediction")
    with st.form("predict_form"):
        inputs = {}
        for feat in pollutant_features:
            median_value = float(df[feat].median()) if feat in df.columns else 0.0
            inputs[feat] = st.number_input(feat, value=median_value)
        pred_date = st.date_input("Date (for Year/Month/Day)", value=pd.to_datetime("2020-01-01"))
        if "City" in df.columns:
            city_choice = st.selectbox("City (used to set City_Code)", sorted(df["City"].unique()))
            if "City_Code" in df.columns:
                city_code_val = int(df[df["City"] == city_choice]["City_Code"].mode().iloc[0])
            else:
                city_code_val = 0
        else:
            city_choice = None
            city_code_val = 0

        submitted = st.form_submit_button("Predict AQI")
        if submitted:
            # assemble feature vector
            row = []
            for feat in pollutant_features:
                row.append(inputs.get(feat, 0.0))
            # date features
            row.append(pred_date.year)
            row.append(pred_date.month)
            row.append(pred_date.day)
            row.append(pred_date.weekday())
            if "City_Code" in FEATURES:
                row.append(city_code_val)
            row_arr = np.array(row).reshape(1, -1)
            try:
                pred_val = model.predict(row_arr)[0]
                st.success(f"Predicted AQI: {pred_val:.1f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# Main router
# -------------------------
def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Maps", "Model", "About"])

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
    else:
        st.info("Select a page from the sidebar.")

if __name__ == "__main__":
    main()
