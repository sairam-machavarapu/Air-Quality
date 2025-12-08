# app.py (Optimized - Clean & Professional layout)
import streamlit as st
st.set_page_config(page_title="India Air Quality Explorer", layout="wide", initial_sidebar_state="expanded")

import io, zipfile, math
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
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Helpers & Constants
# -------------------------
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]
CITY_COORDS = {
    "Thiruvananthapuram":[8.5241,76.9366], "Shillong":[25.5788,91.8933], "Jaipur":[26.9124,75.7873],
    "Mumbai":[19.0760,72.8777], "Ernakulam":[9.9816,76.2999], "Guwahati":[26.1445,91.7362],
    "Aizawl":[23.7271,92.7176], "Delhi":[28.7041,77.1025], "Bengaluru":[12.9716,77.5946],
    "Visakhapatnam":[17.6868,83.2185], "Lucknow":[26.8467,80.9462], "Patna":[25.5941,85.1376],
    "Kochi":[9.9312,76.2673], "Gurugram":[28.4595,77.0266], "Coimbatore":[11.0168,76.9558],
    "Amaravati":[16.5414,80.5150], "Chandigarh":[30.7333,76.7794], "Amritsar":[31.6340,74.8723],
    "Jorapokhar":[23.8,86.4], "Talcher":[20.9497,85.2332], "Kolkata":[22.5726,88.3639],
    "Hyderabad":[17.3850,78.4867], "Ahmedabad":[23.0225,72.5714], "Chennai":[13.0827,80.2707],
    "Bhopal":[23.2599,77.4126], "Brajrajnagar":[21.8160,83.9008]
}

# -------------------------
# Caching: load & preprocess
# -------------------------
@st.cache_data
def load_zip_to_df(uploaded_zip_bytes):
    """
    Accepts an uploaded zip file (streamlit UploadedFile) and returns merged DataFrame.
    """
    if uploaded_zip_bytes is None:
        return None, []
    z = zipfile.ZipFile(io.BytesIO(uploaded_zip_bytes.read()))
    dfs = []
    files = []
    for name in z.namelist():
        if name.lower().endswith(".csv"):
            try:
                df_tmp = pd.read_csv(z.open(name))
                df_tmp["__source_file"] = name
                dfs.append(df_tmp)
                files.append(name)
            except Exception as e:
                # skip unreadable file
                continue
    if not dfs:
        return None, []
    df = pd.concat(dfs, ignore_index=True)
    return df, files

@st.cache_data
def preprocess(df):
    """Safe, idempotent preprocessing for EDA & modeling"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Date parse
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # numeric conversion for known pollutant columns
    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)
    # drop rows without City/Date
    df = df.dropna(subset=["City","Date"])
    df = df.sort_values(["City","Date"]).reset_index(drop=True)
    # fill missing pollutant values per city using transform (index-aligned)
    if present:
        df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
        for c in present:
            df[c] = df[c].fillna(df[c].median())
    # time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    # category encoding for city
    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes
    return df

# -------------------------
# Model caching (resource)
# -------------------------
@st.cache_resource
def train_default_model(X, y):
    """Train a simple RandomForest and cache it."""
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

# -------------------------
# UI - Header & Sidebar
# -------------------------
def render_header():
    st.markdown("<h1 style='margin-bottom:0.2rem;'>India Air Quality Explorer</h1>", unsafe_allow_html=True)
    st.markdown("Clean & professional EDA, maps and baseline AQI model")

def sidebar_controls():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Home / Upload", "Data Overview", "EDA", "Maps", "Model & Prediction", "About"])

# -------------------------
# Page Implementations
# -------------------------
def page_home():
    st.subheader("Upload dataset.zip (contains city CSV files)")
    uploaded = st.file_uploader("Upload dataset.zip", type=["zip"])
    if uploaded:
        df, files = load_zip_to_df(uploaded)
        if df is None:
            st.error("No CSV files detected in the ZIP. Ensure files end with .csv")
            return
        st.success(f"Loaded {len(files)} CSV files")
        st.session_state["raw_df"] = df
        st.session_state["file_list"] = files
        st.dataframe(df.head(200))
    else:
        st.info("Upload a ZIP with your city CSV files, or push dataset to repo and redeploy.")

def page_data_overview():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip on Home first.")
        return
    df = st.session_state["raw_df"]
    st.subheader("Merged dataset sample")
    st.dataframe(df.head(200))
    st.subheader("Columns & dtypes")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))
    st.subheader("Basic statistics (numeric)")
    st.dataframe(df.describe().T)

def page_eda():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip on Home first.")
        return
    df = preprocess(st.session_state["raw_df"])
    st.sidebar.header("EDA Controls")
    cities = ["All"] + sorted(df["City"].unique().tolist())
    city = st.sidebar.selectbox("City", cities, index=0)
    pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5","PM10","AQI"])
    # date filter
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    dr = st.sidebar.date_input("Date range", [min_date, max_date])
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    # filter
    df_f = df[(df["Date"] >= start) & (df["Date"] <= end)]
    if city != "All":
        df_f = df_f[df_f["City"] == city]
    # summary
    st.subheader(f"{pollutant} — Summary")
    st.write(df_f[pollutant].describe())
    # time series (monthly)
    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(monthly["Date"], monthly[pollutant], marker="o")
    ax.set_title(f"Monthly average {pollutant}")
    ax.set_ylabel(pollutant)
    st.pyplot(fig)
    # boxplot by month
    st.subheader("Monthly spread")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    try:
        sns.boxplot(x="Month", y=pollutant, data=df_f, ax=ax2)
        st.pyplot(fig2)
    except Exception:
        st.info("Not enough data for monthly boxplot.")
    # correlation (small sample)
    st.subheader("Correlation (sample)")
    sample = df.sample(min(2000, len(df)), random_state=42)
    cols = [c for c in ["PM2.5","PM10","NO2","CO","O3","AQI"] if c in sample.columns]
    if len(cols) >= 2:
        fig3, ax3 = plt.subplots(figsize=(6,5))
        sns.heatmap(sample[cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Not enough pollutant columns to show correlation.")

def _make_base_map():
    return folium.Map(location=[22.9734,78.6569], zoom_start=5, control_scale=True)

def page_maps():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip on Home first.")
        return
    df = preprocess(st.session_state["raw_df"])
    # attach coords (vectorized)
    coords = CITY_COORDS
    df["Latitude"] = df["City"].map(lambda c: coords.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: coords.get(c, [None, None])[1])
    df_geo = df.dropna(subset=["Latitude","Longitude"])
    if df_geo.empty:
        st.error("No geolocated cities found. Check city names or add coords.")
        return
    st.subheader("AQI Bubble Map (aggregated by city)")
    city_stats = df_geo.groupby("City").mean(numeric_only=True).reset_index()
    m = _make_base_map()
    for _, r in city_stats.iterrows():
        lat, lon = r["Latitude"], r["Longitude"]
        aqi = r.get("AQI", float("nan"))
        radius = max(6, min(25, (aqi/10) if math.isfinite(aqi) else 6))
        color = "green" if (not math.isnan(aqi) and aqi<=100) else "orange" if (not math.isnan(aqi) and aqi<=200) else "red"
        folium.CircleMarker(location=[lat, lon], radius=radius, color=color, fill=True, fill_opacity=0.7,
                            popup=f"{r['City']} — AQI: {aqi:.1f}").add_to(m)
    st_folium(m, width=900, height=500)
    st.markdown("---")
    st.subheader("PM2.5 Heatmap")
    hm = _make_base_map()
    heat_df = df_geo[["Latitude","Longitude","PM2.5"]].dropna()
    HeatMap(heat_df.values.tolist(), radius=12, blur=18, max_val=heat_df["PM2.5"].max()).add_to(hm)
    st_folium(hm, width=900, height=500)

def page_model():
    st.subheader("Baseline AQI Model — Random Forest")
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip on Home first.")
        return
    df = preprocess(st.session_state["raw_df"])
    # select pollutant features available
    feat_poll = [c for c in ["PM2.5","PM10","NO2","NO","NOx","CO","SO2","O3"] if c in df.columns]
    if not feat_poll or "AQI" not in df.columns:
        st.error("Required columns (PM2.5, AQI, ...) are missing.")
        return
    features = feat_poll + ["Year","Month","Day","Weekday","City_Code"]
    X = df[features].fillna(0)
    y = df["AQI"].fillna(df["AQI"].median())
    # train-test split small convenience
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Training RandomForest (cached)...")
    model = train_default_model(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    st.metric("Test RMSE", f"{rmse:.2f}")
    st.metric("Test R2", f"{r2:.3f}")
    # feature importance (if available)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(12)
        st.subheader("Top feature importances")
        st.bar_chart(fi.set_index("feature"))
    # single sample prediction UI
    st.subheader("Single-sample prediction")
    cols = st.columns(3)
    sample = {}
    for idx, p in enumerate(feat_poll):
        sample[p] = cols[idx % 3].number_input(p, value=float(df[p].median()))
    date_sample = st.date_input("Date for Year/Month/Day", value=pd.to_datetime("2020-01-01"))
    city_choice = st.selectbox("City (for city_code)", sorted(df["City"].unique()))
    if st.button("Predict AQI"):
        row = [sample[p] for p in feat_poll] + [date_sample.year, date_sample.month, date_sample.day, date_sample.weekday(), int(df[df["City"]==city_choice]["City_Code"].mode()[0])]
        pred = model.predict(np.array(row).reshape(1,-1))[0]
        st.success(f"Predicted AQI: {pred:.1f}")

def page_about():
    st.markdown("### About")
    st.write("Clean & optimized Streamlit app for India air quality EDA and a baseline AQI model.")
    st.write("Instructions: Upload dataset.zip on Home. ZIP must contain CSV files for cities (e.g. Delhi_data.csv).")

# -------------------------
# Main app flow
# -------------------------
def main():
    render_header()
    choice = sidebar_controls()
    st.markdown("---")
    # route pages
    if choice == "Home / Upload":
        page_home()
    elif choice == "Data Overview":
        page_data_overview()
    elif choice == "EDA":
        page_eda()
    elif choice == "Maps":
        page_maps()
    elif choice == "Model & Prediction":
        page_model()
    else:
        page_about()
    # footer
    st.markdown("""---""")
    st.caption("Prepared for CMP7005 — India Air Quality Explorer")

if __name__ == "__main__":
    main()
