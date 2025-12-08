import streamlit as st
st.set_page_config(page_title="India Air Quality Explorer",
                   layout="wide",
                   initial_sidebar_state="expanded")

import io, zipfile, math
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
# CONSTANTS
# =========================================================
POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"
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
# LOAD ZIP → MERGED DATAFRAME
# =========================================================
@st.cache_data
def load_zip_to_df(uploaded_zip):
    if uploaded_zip is None:
        return None, []

    z = zipfile.ZipFile(io.BytesIO(uploaded_zip.read()))
    dfs, files = [], []

    for name in z.namelist():
        if name.lower().endswith(".csv"):
            try:
                df_temp = pd.read_csv(z.open(name))
                df_temp["__SOURCE"] = name
                dfs.append(df_temp)
                files.append(name)
            except:
                continue

    if not dfs:
        return None, []
    
    return pd.concat(dfs, ignore_index=True), files


# =========================================================
# CLEANING / PREPROCESSING
# =========================================================
@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)

    df = df.dropna(subset=["City", "Date"]).sort_values(["City", "Date"])

    if present:
        df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
        for c in present:
            df[c] = df[c].fillna(df[c].median())

    # Time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)


# =========================================================
# MODEL TRAINING (cached)
# =========================================================
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


# =========================================================
# PAGE: HOME
# =========================================================
def page_home():
    st.header("Upload Dataset (ZIP)")

    uploaded = st.file_uploader("Upload dataset.zip", type=["zip"])

    if uploaded:
        df, files = load_zip_to_df(uploaded)

        if df is None:
            st.error("No CSV files found inside ZIP.")
            return

        st.session_state["raw_df"] = df
        st.session_state["files"] = files

        st.success(f"Loaded {len(files)} CSV files")
        st.dataframe(df.head(200))
    else:
        st.info("Upload your dataset.zip to continue.")


# =========================================================
# PAGE: DATA OVERVIEW
# =========================================================
def page_data_overview():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip first.")
        return

    df = st.session_state["raw_df"]

    st.header("Data Overview")
    st.dataframe(df.head(200))

    st.subheader("Column Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T)


# =========================================================
# PAGE: EDA
# =========================================================
def page_eda():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip first.")
        return

    df = preprocess(st.session_state["raw_df"])
    st.header("Exploratory Data Analysis")

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("City", cities)
    pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5", "PM10", "AQI"])

    min_date, max_date = df["Date"].min(), df["Date"].max()
    dr = st.sidebar.date_input("Date Range", [min_date, max_date])
    start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])

    df_f = df[(df["Date"] >= start) & (df["Date"] <= end)]
    if city_sel != "All":
        df_f = df_f[df_f["City"] == city_sel]

    st.subheader(f"{pollutant} Summary")
    st.write(df_f[pollutant].describe())

    # Monthly trend
    st.subheader("Monthly Trend")
    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(monthly["Date"], monthly[pollutant], marker="o")
    ax.set_ylabel(pollutant)
    st.pyplot(fig)

    # Boxplot
    st.subheader("Monthly Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.boxplot(x="Month", y=pollutant, data=df_f, ax=ax2)
    st.pyplot(fig2)


# =========================================================
# PAGE: MAPS  (FULLY FIXED VERSION)
# =========================================================
def page_maps():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip first.")
        return

    df = preprocess(st.session_state["raw_df"])
    st.header("AQI Bubble Map")

    # add coordinates
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[1])
    df_geo = df.dropna(subset=["Latitude", "Longitude"])

    if df_geo.empty:
        st.error("No coordinates available for your dataset.")
        return

    # Proper aggregation
    pollutant_means = df_geo.groupby("City")[["AQI", "PM2.5", "PM10"]].mean().reset_index()
    city_coords = df_geo.groupby("City")[["Latitude", "Longitude"]].first().reset_index()
    city_stats = pollutant_means.merge(city_coords, on="City", how="left")

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

    for _, r in city_stats.iterrows():
        lat, lon = r["Latitude"], r["Longitude"]
        if pd.isna(lat) or pd.isna(lon): 
            continue
        aqi = r["AQI"]

        # color scale
        if aqi <= 100:
            color = "green"
        elif aqi <= 200:
            color = "orange"
        else:
            color = "red"

        radius = min(25, max(6, aqi / 10))

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{r['City']} — AQI: {aqi:.1f}"
        ).add_to(m)

    st_folium(m, width=900, height=500)

    # Heatmap
    st.subheader("PM2.5 Heatmap")
    hm = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
    heat_df = df_geo[["Latitude", "Longitude", "PM2.5"]].dropna()

    if not heat_df.empty:
        HeatMap(heat_df.values.tolist()).add_to(hm)

    st_folium(hm, width=900, height=500)


# =========================================================
# PAGE: MODEL
# =========================================================
def page_model():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip first.")
        return

    df = preprocess(st.session_state["raw_df"])

    st.header("AQI Prediction Model")

    usable = [c for c in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"] if c in df.columns]
    if not usable:
        st.error("Insufficient pollutant columns for modeling.")
        return

    features = usable + ["Year", "Month", "Day", "Weekday", "City_Code"]

    X = df[features]
    y = df["AQI"].fillna(df["AQI"].median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    # FIXED RMSE (NO squared=...)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.3f}")

    # Feature Importances
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        fi = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        st.bar_chart(fi.set_index("feature"))

    # User Prediction
    st.subheader("Predict AQI for Custom Input")

    inputs = {}
    for col in usable:
        inputs[col] = st.number_input(col, value=float(df[col].median()))

    date = st.date_input("Date")
    city_choice = st.selectbox("City", sorted(df["City"].unique()))

    if st.button("Predict AQI"):
        city_code = int(df[df["City"] == city_choice]["City_Code"].mode()[0])

        row = [
            inputs[c] for c in usable
        ] + [
            date.year, date.month, date.day, date.weekday(), city_code
        ]

        pred = model.predict(np.array(row).reshape(1, -1))[0]
        st.success(f"Predicted AQI: {pred:.1f}")


# =========================================================
# PAGE: ABOUT
# =========================================================
def page_about():
    st.header("About This App")
    st.write("""
This India Air Quality Explorer app provides:
- ZIP-based dataset upload  
- Automated city-level merging  
- Preprocessing & cleaning  
- EDA with trends, boxplots, correlations  
- Interactive pollution maps  
- Machine learning AQI predictor  
""")

# =========================================================
# ROUTER
# =========================================================
def main():
    st.title("India Air Quality Explorer")

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Data Overview", "EDA", "Maps", "Model", "About"]
    )

    if page == "Home":
        page_home()
    elif page == "Data Overview":
        page_data_overview()
    elif page == "EDA":
        page_eda()
    elif page == "Maps":
        page_maps()
    elif page == "Model":
        page_model()
    else:
        page_about()

    st.markdown("---")
    st.caption("CMP7005 — Air Quality App (Optimized for Streamlit Cloud)")


if __name__ == "__main__":
    main()
