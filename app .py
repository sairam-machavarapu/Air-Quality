import streamlit as st
st.set_page_config(layout="wide", page_title="India Air Quality Explorer", initial_sidebar_state="expanded")

import os, glob, zipfile, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CLOUD-READY ZIP UPLOAD LOADER
# =====================================================

@st.cache_data
def load_uploaded_zip(uploaded_zip):
    """
    Extracts CSVs from an uploaded ZIP and returns merged DataFrame.
    Works on Streamlit Cloud & local environments.
    """
    if uploaded_zip is None:
        return None, []

    z = zipfile.ZipFile(io.BytesIO(uploaded_zip.read()))

    dfs = []
    csv_files = []

    for name in z.namelist():
        if name.endswith(".csv"):
            csv_files.append(name)
            df_temp = pd.read_csv(z.open(name))
            df_temp["__source_file"] = name
            dfs.append(df_temp)

    if not dfs:
        return None, []

    df = pd.concat(dfs, ignore_index=True)
    return df, csv_files


# =====================================================
# DATA CLEANING & PREPROCESSING
# =====================================================

@st.cache_data
def preprocess(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    pollutant_cols = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3",
                      "Benzene","Toluene","Xylene","AQI"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in pollutant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["City","Date"])

    df = df.sort_values(["City","Date"])

    present_cols = [c for c in pollutant_cols if c in df.columns]
    if present_cols:
        df[present_cols] = df.groupby("City")[present_cols].apply(lambda g: g.ffill().bfill())
        for col in present_cols:
            df[col] = df[col].fillna(df[col].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df


# =====================================================
# CITY COORDS FOR MAPS
# =====================================================

@st.cache_data
def get_city_coordinates():
    return {
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


# =====================================================
# STREAMLIT NAVIGATION
# =====================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Upload", "Data Overview", "EDA", "Maps", "Model & Prediction", "About"])


# =====================================================
# PAGE 1 — HOME / UPLOAD
# =====================================================

if page == "Home / Upload":
    st.title("🇮🇳 India Air Quality — Streamlit App")
    st.write("Upload your **dataset.zip** containing all city CSVs.")

    uploaded_zip = st.file_uploader("Upload dataset.zip", type=["zip"])

    if uploaded_zip:
        df, files = load_uploaded_zip(uploaded_zip)
        st.session_state["df"] = df
        st.session_state["files"] = files

        if df is not None:
            st.success(f"Loaded {len(files)} CSV files!")
            st.dataframe(df.head())
        else:
            st.error("No CSV files found inside the ZIP.")


# =====================================================
# PAGE 2 — DATA OVERVIEW
# =====================================================

elif page == "Data Overview":
    st.title("Data Overview")

    if "df" not in st.session_state:
        st.warning("Please upload dataset.zip first.")
    else:
        df = st.session_state["df"]
        st.dataframe(df.head(200))

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Statistics")
        st.write(df.describe().T)


# =====================================================
# PAGE 3 — EDA
# =====================================================

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    if "df" not in st.session_state:
        st.warning("Please upload dataset.zip first.")
    else:
        df = preprocess(st.session_state["df"])

        st.sidebar.subheader("Filters")
        cities = ["All"] + sorted(df["City"].unique())
        city_sel = st.sidebar.selectbox("City", cities)
        pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5","PM10","NO2","CO","O3","AQI"])

        if city_sel != "All":
            df = df[df["City"] == city_sel]

        monthly = df.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(monthly["Date"], monthly[pollutant], marker="o")
        ax.set_title(f"Monthly {pollutant} Trend")
        st.pyplot(fig)


# =====================================================
# PAGE 4 — MAPS
# =====================================================

elif page == "Maps":
    st.title("Air Quality Maps")

    if "df" not in st.session_state:
        st.warning("Please upload dataset.zip first.")
    else:
        coords = get_city_coordinates()
        df = preprocess(st.session_state["df"])
        df["Latitude"] = df["City"].apply(lambda c: coords.get(c,[None,None])[0])
        df["Longitude"] = df["City"].apply(lambda c: coords.get(c,[None,None])[1])
        df_geo = df.dropna(subset=["Latitude","Longitude"])

        st.subheader("AQI Bubble Map")

        m = folium.Map(location=[22.9734,78.6569], zoom_start=5)

        grouped = df_geo.groupby("City").mean(numeric_only=True)

        for city, row in grouped.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=min(20, max(5, row["AQI"]/10)),
                color="red" if row["AQI"]>200 else "orange" if row["AQI"]>100 else "green",
                fill=True, fill_opacity=0.7,
                popup=f"{city}: AQI {row['AQI']:.1f}"
            ).add_to(m)

        st_folium(m, width=900, height=600)


# =====================================================
# PAGE 5 — MODEL & PREDICTION
# =====================================================

elif page == "Model & Prediction":
    st.title("Model Training & Prediction")

    if "df" not in st.session_state:
        st.warning("Please upload dataset.zip first.")
    else:
        df = preprocess(st.session_state["df"])

        pollutant_cols = ["PM2.5","PM10","NO2","NO","NOx","NH3","CO","SO2","O3"]
        pollutant_cols = [c for c in pollutant_cols if c in df.columns]

        features = pollutant_cols + ["Year","Month","Day","Weekday","City_Code"]

        X = df[features]
        y = df["AQI"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor(n_estimators=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        st.write("R2:", r2_score(y_test, preds))


# =====================================================
# PAGE 6 — ABOUT
# =====================================================

elif page == "About":
    st.title("About This App")
    st.write("""
    This Streamlit app was developed for exploring and analyzing India's air quality
    across multiple cities using uploaded CSV datasets.
    """)

