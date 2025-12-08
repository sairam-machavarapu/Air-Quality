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
# GLOBAL CONSTANTS
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
# FILE LOADING (ZIP)
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
                df_temp["__source_file"] = name
                dfs.append(df_temp)
                files.append(name)
            except:
                continue

    if not dfs:
        return None, []

    df = pd.concat(dfs, ignore_index=True)
    return df, files


# =========================================================
# PREPROCESSING
# =========================================================
@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    present_pollutants = []
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            present_pollutants.append(col)

    df = df.dropna(subset=["City", "Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    # FIXED: transform keeps index aligned (no KeyErrors)
    if present_pollutants:
        df[present_pollutants] = df.groupby("City")[present_pollutants].transform(
            lambda g: g.ffill().bfill()
        )
        for col in present_pollutants:
            df[col] = df[col].fillna(df[col].median())

    # Time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df


# =========================================================
# MODEL TRAINING (CACHED)
# =========================================================
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


# =========================================================
# PAGE: HOME / UPLOAD
# =========================================================
def page_home():
    st.header("Upload Dataset (ZIP)")

    uploaded = st.file_uploader("Upload dataset.zip", type=["zip"])

    if uploaded:
        df, files = load_zip_to_df(uploaded)

        if df is None:
            st.error("No CSV files found in the ZIP.")
            return

        st.session_state["raw_df"] = df
        st.session_state["files"] = files

        st.success(f"Successfully loaded {len(files)} CSV files.")
        st.dataframe(df.head(200))

    else:
        st.info("Upload a ZIP containing your city CSV files.")


# =========================================================
# PAGE: DATA OVERVIEW
# =========================================================
def page_data_overview():
    if "raw_df" not in st.session_state:
        st.warning("Please upload dataset.zip first.")
        return

    df = st.session_state["raw_df"]

    st.header("Data Overview")
    st.subheader("Sample Rows")
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

    cities = ["All"] + sorted(df["City"].unique().tolist())
    city = st.sidebar.selectbox("City", cities)

    pollutant = st.sidebar.selectbox("Pollutant",
                                     ["PM2.5", "PM10", "AQI"])

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    dr = st.sidebar.date_input("Date Range", [min_date, max_date])

    df_f = df[(df["Date"] >= pd.to_datetime(dr[0])) &
              (df["Date"] <= pd.to_datetime(dr[1]))]

    if city != "All":
        df_f = df_f[df_f["City"] == city]

    st.subheader(f"{pollutant} Summary")
    st.write(df_f[pollutant].describe())

    # Time series
    st.subheader("Monthly Trend")
    monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(monthly["Date"], monthly[pollutant], marker="o")
    st.pyplot(fig)

    # Boxplot
    st.subheader("Monthly Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.boxplot(x="Month", y=pollutant, data=df_f, ax=ax2)
    st.pyplot(fig2)


# =========================================================
# PAGE: MAPS (FIXED VERSION)
# =========================================================
def page_maps():
    if "raw_df" not in st.session_state:
        st.warning("Upload dataset.zip first.")
        return

    df = preprocess(st.session_state["raw_df"])

    # Add coordinates
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[1])

    df_geo = df.dropna(subset=["Latitude", "Longitude"])

    if df_geo.empty:
        st.error("No coordinates found for the cities in your dataset.")
        return

    st.header("AQI Bubble Map")

    # FIXED aggregation: preserve coordinates
    pollutant_means = df_geo.groupby("City")[["AQI", "PM2.5", "PM10"]].mean().reset_index()
    city_coords = df_geo.groupby("City")[["Latitude", "Longitude"]].first().reset_index()

    city_stats = pollutant_means.merge(city_coords, on="City", how="left")

    # Bubble map
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, control_scale=True)

    for _, r in city_stats.iterrows():
        lat = r["Latitude"]
        lon = r["Longitude"]
        aqi = r["AQI"]

        color = "green" if aqi <= 100 else "orange" if aqi <= 200 else "red"
        radius = max(6, min(25, aqi / 10))

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{r['City']} — AQI: {aqi:.1f}"
        ).add_to(m)

    st_folium(m, width=900, height=500)

    st.markdown("---")
    st.header("PM2.5 Heatmap")

    hm = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
    heat_df = df_geo[["Latitude", "Longitude", "PM2.5"]].dropna()
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

    st.header("AQI Predictive Model")

    # Select features
    valid = [c for c in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
             if c in df.columns]

    if not valid:
        st.error("Required pollutant columns missing.")
        return

    features = valid + ["Year", "Month", "Day", "Weekday", "City_Code"]

    X = df[features]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train cached model
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.3f}")

    # Feature importances
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        fi = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        st.bar_chart(fi.set_index("feature"))

    # Prediction widget
    st.subheader("Predict AQI from Manual Inputs")

    inputs = {}
    for col in valid:
        inputs[col] = st.number_input(col, value=float(df[col].median()))

    date = st.date_input("Select Date")

    city_choice = st.selectbox("City", sorted(df["City"].unique()))

    if st.button("Predict AQI"):
        feat = [
            inputs[c] for c in valid
        ] + [
            date.year, date.month, date.day, date.weekday(),
            int(df[df["City"] == city_choice]["City_Code"].mode()[0])
        ]

        pred = model.predict(np.array(feat).reshape(1, -1))[0]
        st.success(f"Predicted AQI: {pred:.2f}")


# =========================================================
# PAGE: ABOUT
# =========================================================
def page_about():
    st.header("About This Application")
    st.write("""
This is a clean and optimized Streamlit app designed for CMP7005.  
It supports:
- ZIP-based dataset uploads  
- Automatic merging & preprocessing  
- EDA with plots  
- India-wide pollution maps  
- Random Forest AQI prediction  
    """)


# =========================================================
# MAIN APP ROUTER
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
    st.caption("CMP7005 • Air Quality App • Streamlit Cloud Version")


if __name__ == "__main__":
    main()
