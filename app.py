# Disable Streamlit file watching to avoid inotify limit errors
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
st.set_page_config(
    page_title="India Air Quality Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# CONSTANTS
# -----------------------------
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

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_dataset(folder="dataset"):
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df_tmp = pd.read_csv(os.path.join(folder, f))
            df_tmp["__source"] = f
            dfs.append(df_tmp)
        except:
            pass

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

    df = df.dropna(subset=["City", "Date"])

    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)

    df = df.sort_values(["City", "Date"])
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
# HOME PAGE
# -----------------------------
def page_home(df, files):
    st.title("India Air Quality Explorer")

    st.success(f"Loaded {len(files)} file(s). Total rows: {len(df):,}")

    st.markdown("""
    ### 📘 About This Dataset

    This dataset contains **India's air quality measurements** collected from multiple
    Continuous Ambient Air Quality Monitoring Stations (CAAQMS).  
    Each file represents pollutant readings recorded at different time intervals across major Indian cities.

    #### **Dataset Includes:**
    - 🌆 **Cities:** 26 major urban locations across India  
    - 📅 **Time Span:** Multiple years of historical air quality data  
    - 🧪 **Pollutants Tracked:**  
      - PM2.5, PM10  
      - NO, NO2, NOx  
      - CO, SO2, NH3  
      - O3, Benzene, Toluene, Xylene  
      - **AQI (Air Quality Index)**  
    - 🗂 **Total Rows:** Provides a large enough sample for visualisation and machine learning prediction  

    #### **Purpose of This App**
    This tool helps you:
    - Visualize trends in air pollution  
    - Compare cities on pollution levels  
    - Explore spatial pollution patterns on maps  
    - Predict **AQI** using machine learning models  

    ---
    """)

    st.markdown("#### 🔍 *Navigate using the sidebar to explore different sections of the app.*")


# -----------------------------
# DATA OVERVIEW
# -----------------------------
def page_data_overview(df):
    st.header("📊 Data Overview")

    if df.empty:
        st.warning("Dataset is empty!")
        return

    # ================================
    # Sidebar Filters
    # ================================
    st.sidebar.subheader("Filters")

    # City filter
    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("Select City", cities)

    # Date filter
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Pollutant highlight
    highlight_col = st.sidebar.selectbox(
        "Highlight Pollutant Column",
        ["None"] + [c for c in POLLUTANTS if c in df.columns]
    )

    # ================================
    # Apply Filters
    # ================================
    df_filtered = df.copy()

    if city_sel != "All":
        df_filtered = df_filtered[df_filtered["City"] == city_sel]

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[(df_filtered["Date"] >= start_date) & (df_filtered["Date"] <= end_date)]

    # ================================
    # Summary Stats Cards
    # ================================
    st.subheader("📌 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", f"{len(df_filtered):,}")
    col2.metric("Date Range", f"{start_date.date()} → {end_date.date()}")
    col3.metric("Total Cities", df_filtered["City"].nunique())
    col4.metric("Avg AQI", f"{df_filtered['AQI'].mean():.2f}")

    # ================================
    # Data Preview Table
    # ================================
    st.subheader("📄 Filtered Dataset Preview")

    if highlight_col != "None":
        # Highlight selected pollutant column
        def highlight_column(x):
            return ['background-color: #2a2a9e' if col == highlight_col else '' for col in x.index]
        st.dataframe(df_filtered.head(50).style.apply(highlight_column, axis=1))
    else:
        st.dataframe(df_filtered.head(50))

    # ================================
    # Summary Table
    # ================================
    st.subheader("📈 Statistical Summary")
    st.dataframe(df_filtered.describe().T)

    # ================================
    # Quick Insights
    # ================================
    st.subheader("🔍 Insights")

    top_city = df_filtered.groupby("City")["AQI"].mean().sort_values(ascending=False).head(1)
    worst_city = top_city.index[0]
    worst_value = top_city.values[0]

    st.markdown(f"""
    - 🌆 **Most polluted city in selection:** `{worst_city}` (Avg AQI = {worst_value:.2f})  
    - 🕒 Data spans from **{start_date.date()}** to **{end_date.date()}**  
    - 📌 Showing **{len(df_filtered):,} rows** from **{df_filtered['City'].nunique()} cities**
    """)

    # ================================
    # Download Button
    # ================================
    st.subheader("⬇️ Download Filtered Data")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="filtered_air_quality.csv",
        mime="text/csv"
    )

# -----------------------------
# EDA
# -----------------------------
def page_eda(df):
    st.header("Exploratory Data Analysis")

    if df.empty:
        st.warning("No data available.")
        return

    # Sidebar Controls
    st.sidebar.subheader("Filters")

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("City", cities)

    pollutants = [c for c in POLLUTANTS if c in df.columns]
    pollutant = st.sidebar.selectbox("Pollutant", pollutants)

    eda_option = st.sidebar.selectbox(
        "Select EDA Visualisation",
        [
            "Monthly Trend",
            "Yearly Trend",
            "Seasonal Pattern (Month-wise)",
            "Weekday Pattern",
            "Distribution (Histogram + KDE)",
            "Boxplot",
            "Correlation Heatmap",
            "City-wise Comparison"
        ]
    )

    df_f = df if city_sel == "All" else df[df["City"] == city_sel]

    if df_f.empty:
        st.warning("No data for the selected filters.")
        return

    # -------------------------
    # 1. Monthly Trend
    # -------------------------
    if eda_option == "Monthly Trend":
        st.subheader(f"Monthly Trend: {pollutant}")

        monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_ylabel(pollutant)
        ax.grid(True)
        st.pyplot(fig)

    # -------------------------
    # 2. Yearly Trend
    # -------------------------
    elif eda_option == "Yearly Trend":
        st.subheader(f"Yearly Trend: {pollutant}")

        yearly = df_f.groupby("Year")[pollutant].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(yearly.index, yearly.values)
        ax.set_xlabel("Year")
        st.pyplot(fig)

    # -------------------------
    # 3. Seasonal Pattern
    # -------------------------
    elif eda_option == "Seasonal Pattern (Month-wise)":
        st.subheader(f"Seasonal Pattern (Month-wise): {pollutant}")

        monthwise = df_f.groupby("Month")[pollutant].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(monthwise.index, monthwise.values, marker="o")
        ax.set_xticks(range(1, 13))
        st.pyplot(fig)

    # -------------------------
    # 4. Weekday Pattern
    # -------------------------
    elif eda_option == "Weekday Pattern":
        st.subheader(f"Weekday Pattern: {pollutant}")

        weekday = df_f.groupby("Weekday")[pollutant].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            weekday.values
        )
        st.pyplot(fig)

    # -------------------------
    # 5. Distribution
    # -------------------------
    elif eda_option == "Distribution (Histogram + KDE)":
        st.subheader(f"Distribution of {pollutant}")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_f[pollutant], kde=True, ax=ax)
        st.pyplot(fig)

    # -------------------------
    # 6. Boxplot
    # -------------------------
    elif eda_option == "Boxplot":
        st.subheader(f"Boxplot of {pollutant}")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df_f[pollutant], ax=ax)
        st.pyplot(fig)

    # -------------------------
    # 7. Correlation Heatmap
    # -------------------------
    elif eda_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        num_df = df_f.select_dtypes(include=["float", "int"])
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------------------------
    # 8. City-wise Comparison
    # -------------------------
    elif eda_option == "City-wise Comparison":
        if city_sel != "All":
            st.info("City-wise comparison is available only when 'City = All'")
            return

        st.subheader(f"City-wise Average {pollutant}")

        city_avg = df.groupby("City")[pollutant].mean().sort_values()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(city_avg.index, city_avg.values)
        st.pyplot(fig)

#   maps

def page_maps(df):
    st.header("Geographical Maps")

    if df.empty:
        st.warning("No data available for maps.")
        return

    # Sidebar Options
    st.sidebar.subheader("Map Options")

    map_type = st.sidebar.selectbox(
        "Select Map Visualization",
        ["AQI Markers", "AQI Heatmap", "City Marker Clusters"]
    )

    map_theme = st.sidebar.selectbox(
        "Base Map Theme",
        ["OpenStreetMap", "CartoDB Dark Matter", "Stamen Terrain", "Stamen Toner"]
    )

    pollutant_choice = st.sidebar.selectbox(
        "Pollutant for Heatmap",
        [c for c in POLLUTANTS if c in df.columns]
    )

    # Prepare dataset with coordinates
    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [None, None])[1])

    df_geo = df.dropna(subset=["Latitude", "Longitude"])

    if df_geo.empty:
        st.error("No geolocation data available for these cities.")
        return

    # Convert pollutant to numeric
    if pollutant_choice in df_geo.columns:
        df_geo[pollutant_choice] = pd.to_numeric(df_geo[pollutant_choice], errors="coerce")

    # Base Map
    m = folium.Map(
        location=[22.97, 78.65],
        zoom_start=5,
        tiles=map_theme
    )

    # -----------------------------------------------------
    # 1. AQI Circle Markers
    # -----------------------------------------------------
    if map_type == "AQI Markers":
        stats = df_geo.groupby("City").agg({
            "AQI": "mean",
            "Latitude": "first",
            "Longitude": "first"
        }).reset_index()

        for _, r in stats.iterrows():
            if pd.isna(r["AQI"]):
                continue

            aqi = r["AQI"]
            color = "green" if aqi <= 100 else "orange" if aqi <= 200 else "red"

            folium.CircleMarker(
                [r["Latitude"], r["Longitude"]],
                radius=max(6, min(25, aqi / 10)),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{r['City']} — AQI {aqi:.1f}"
            ).add_to(m)

    # -----------------------------------------------------
    # 2. AQI Heatmap
    # -----------------------------------------------------
    elif map_type == "AQI Heatmap":
        from folium.plugins import HeatMap

        heat_df = df_geo.dropna(subset=[pollutant_choice])
        heat_df = heat_df[heat_df[pollutant_choice] > 0]  # heatmap cannot use zero/negative

        if heat_df.empty:
            st.error("No valid data available for this pollutant heatmap.")
            return

        heat_data = heat_df[["Latitude", "Longitude", pollutant_choice]].values.tolist()

        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            min_opacity=0.5
        ).add_to(m)

    # -----------------------------------------------------
    # 3. City Marker Clusters
    # -----------------------------------------------------
    elif map_type == "City Marker Clusters":
        from folium.plugins import MarkerCluster

        cluster = MarkerCluster().add_to(m)

        for _, r in df_geo.iterrows():
            if pd.isna(r["AQI"]):
                continue

            folium.Marker(
                [r["Latitude"], r["Longitude"]],
                popup=f"{r['City']}<br>AQI: {r['AQI']}"
            ).add_to(cluster)

    # -----------------------------------------------------
    # Display Map
    # -----------------------------------------------------
    st_folium(m, width=900, height=550)


# -----------------------------
# MODEL — FAST + CACHED
# -----------------------------
def page_model(df):
    st.header("AQI Prediction Model")

    if "AQI" not in df.columns:
        st.error("Dataset does not contain AQI column!")
        return

    st.sidebar.subheader("Model Options")

    model_choice = st.sidebar.selectbox(
        "Select Model (Fast Mode Available)",
        [
            "Random Forest (Fast)",
            "Extra Trees (Fast)",
            "Decision Tree",
            "Linear Regression",
            "Ridge Regression (Fast)",
            "Lasso Regression (Fast)",
            "SGD Regressor (Ultra Fast)",
            "KNN Regressor (Fast)"
        ]
    )

    test_size_percent = st.sidebar.slider("Test Size (%)", 10, 40, 20, step=5)
    test_size = test_size_percent / 100

    FEATURES = [c for c in POLLUTANTS if c in df.columns and c != "AQI"]
    FEATURES += ["Year", "Month", "Day", "Weekday", "City_Code"]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["AQI"].fillna(df["AQI"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ----------------------------------------------------
    # Model Selection (FAST MODE ENABLED)
    # ----------------------------------------------------
    if model_choice == "Random Forest (Fast)":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42)

    elif model_choice == "Extra Trees (Fast)":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=50, random_state=42)

    elif model_choice == "Decision Tree":
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)

    elif model_choice == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif model_choice == "Ridge Regression (Fast)":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

    elif model_choice == "Lasso Regression (Fast)":
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.001, max_iter=5000)

    elif model_choice == "SGD Regressor (Ultra Fast)":
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor(max_iter=1000, tol=1e-3)

    elif model_choice == "KNN Regressor (Fast)":
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)

    # ----------------------------------------------------
    # Cache Training
    # ----------------------------------------------------
    @st.cache_resource
    def train_model_cached(model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    with st.spinner("Training model..."):
        model = train_model_cached(model, X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.subheader("📌 Model Performance")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    # Avoid feature importance for non-tree models
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="barh", ax=ax)
        st.pyplot(fig)

#  About Page

def page_about():
    st.title("📘 About This Project")

    st.markdown("""
    ## 🌍 India Air Quality Explorer

    This interactive dashboard visualizes **India’s air pollution trends**, geographical
    patterns, and predicts AQI using machine learning models.  
    It is built as an end-to-end data science and geospatial analytics project.

    ---

    ## 📊 Dataset Details

    - **Source:** Continuous Ambient Air Quality Monitoring Stations (CAAQMS)  
    - **Cities Covered:** 26 major Indian cities  
    - **Data Type:** Hourly/continuous pollutant readings  
    - **Total Records:** ~30,000 rows  
    - **Main Pollutants Included:**  
      - PM2.5, PM10  
      - NO, NO2, NOx  
      - CO, SO2, NH3  
      - O3  
      - Benzene, Toluene, Xylene  
      - **AQI (Air Quality Index)**  

    ---

    ## 🧠 Features of This Application

    ### **1. Data Overview**
    - Summary statistics  
    - First 50 rows preview  

    ### **2. EDA (Exploratory Data Analysis)**
    - Monthly and yearly trends  
    - Seasonal and weekday patterns  
    - Distribution plots, boxplots  
    - City-wise pollutant comparison  
    - Correlation heatmaps  

    ### **3. Geographical Maps**
    - AQI marker maps  
    - Pollutant heatmaps  
    - Fast marker clusters  
    - Custom base map themes  

    ### **4. Machine Learning Model**
    - Multiple model selection:  
      **Random Forest, Extra Trees, Decision Tree, Linear Regression**  
    - Train/test split selection  
    - Feature importance  
    - Predicted vs Actual comparison  
    - User input prediction form  

    ---

    ## 🎯 Purpose of the Project

    This project aims to:

    - Understand India’s air pollution patterns  
    - Visualize city-level pollution hotpots  
    - Build ML models to predict AQI  
    - Provide interactive tools for environmental analysis  
    - Raise awareness on air quality and its impact  

    ---

    ## 👨‍💻 Author

    **Vamsi**  
    Developing intelligent data apps using Python, Streamlit, and machine learning.

    If you want to extend this app further (forecasting, alerts, ML explainability), just ask!
    """)


# -----------------------------
# MAIN ROUTER
# -----------------------------
def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Maps", "Model", "About"])

    if page == "Home": page_home(df, files)
    elif page == "Data Overview": page_data_overview(df)
    elif page == "EDA": page_eda(df)
    elif page == "Maps": page_maps(df)
    elif page == "Model": page_model(df)
    elif page == "About":page_about()

if __name__ == "__main__":
    main()
