# app.py
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

# ---------- Utility / cached functions ----------
@st.cache_data
def load_merged_csvs_from_folder(folder_path="/content/dataset/dataset"):
    """
    Scans folder_path for CSVs and returns merged DataFrame
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if len(csv_files) == 0:
        return None, []
    dfs = []
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f)
            df_temp["__source_file"] = os.path.basename(f)
            dfs.append(df_temp)
        except Exception as e:
            st.warning(f"Failed to read {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    return df, csv_files

@st.cache_data
def ensure_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    # columns assumed by the notebook; if missing, we'll warn later
    expected = ["City","Date","PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene","AQI","AQI_Bucket"]
    return df, expected

@st.cache_data
def preprocess(df):
    df = df.copy()
    # convert types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    pollutant_cols = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene","AQI"]
    for c in pollutant_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["City","Date"])
    df = df.sort_values(["City","Date"])
    # forward/backfill per city
    present_pollutants = [c for c in pollutant_cols if c in df.columns]
    if len(present_pollutants) > 0:
        df[present_pollutants] = df.groupby("City")[present_pollutants].apply(lambda g: g.ffill().bfill())
        for c in present_pollutants:
            df[c] = df[c].fillna(df[c].median())
    # time features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes
    return df

@st.cache_data
def get_city_coordinates():
    # full list from your dataset (complete; add if you have more)
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

# ---------- App layout ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Data Upload", "Data Overview", "EDA", "Maps", "Model & Prediction", "About"])

# ---------- Home / Data Upload ----------
if page == "Home / Data Upload":
    st.title("India Air Quality — Streamlit App")
    st.markdown("""
    **How to use**
    - Upload your `dataset.zip` containing all CSVs or mount Google Drive and place the folder at `/content/dataset/dataset/`
    - Navigate to Data Overview to inspect the merged data.
    """)
    st.header("Upload dataset.zip (optional)")
    uploaded = st.file_uploader("Upload dataset.zip (it will be extracted)", type=["zip"])
    if uploaded:
        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as z:
            z.extractall("dataset")
        st.success("Uploaded and extracted to ./dataset")
        st.info("Make sure extracted CSVs are under dataset/dataset/ or update path in code.")
    st.write("---")
    df, csv_files = load_merged_csvs_from_folder("/content/dataset/dataset")
    if df is None:
        st.warning("No CSV files found under /content/dataset/dataset/. Upload zip or mount drive and put CSVs there.")
    else:
        st.success(f"Found {len(csv_files)} CSVs. Merged shape: {df.shape}")
        if st.button("Inspect first 5 rows of merged DF"):
            st.dataframe(df.head())

# ---------- Data Overview ----------
elif page == "Data Overview":
    st.title("Data Overview")
    df, files = load_merged_csvs_from_folder("/content/dataset/dataset")
    if df is None:
        st.warning("No data found. Go to Home to upload.")
    else:
        df, expected = ensure_columns(df)
        st.subheader("Files loaded")
        st.write(f"{len(files)} files (sample):")
        st.write([os.path.basename(f) for f in files[:30]])
        st.subheader("Merged data sample & info")
        st.dataframe(df.head(200))
        st.write("Data types:")
        st.write(df.dtypes)
        st.write("Basic statistics:")
        st.write(df.describe(include='all').T)
        st.write("Columns expected by the app:")
        st.write(expected)

# ---------- EDA ----------
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("Interactive EDA: choose city, date range and pollutant to view trends.")
    df, files = load_merged_csvs_from_folder("/content/dataset/dataset")
    if df is None:
        st.warning("No data. Upload at Home.")
    else:
        df = preprocess(df)
        st.sidebar.subheader("Filters (EDA)")
        cities = ["All"] + sorted(df["City"].unique().tolist())
        city_sel = st.sidebar.selectbox("City", cities, index=0)
        date_min = df["Date"].min()
        date_max = df["Date"].max()
        dr = st.sidebar.date_input("Date range", [date_min.date(), date_max.date()])
        start_date, end_date = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5","PM10","NO2","CO","O3","AQI"])
        # filter
        df_f = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if city_sel != "All":
            df_f = df_f[df_f["City"]==city_sel]
        st.subheader(f"Summary statistics for {pollutant}")
        st.write(df_f[pollutant].describe())
        # Time series
        st.subheader(f"Time series ({pollutant})")
        monthly = df_f.set_index("Date").groupby(pd.Grouper(freq="M"))[pollutant].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(monthly["Date"], monthly[pollutant], marker="o")
        ax.set_title(f"Monthly average {pollutant}")
        ax.set_xlabel("Date")
        ax.set_ylabel(pollutant)
        st.pyplot(fig)
        # Boxplot by month
        st.subheader("Monthly distribution")
        fig2, ax2 = plt.subplots(figsize=(12,4))
        sns.boxplot(x="Month", y=pollutant, data=df_f, ax=ax2)
        st.pyplot(fig2)
        # Correlation heatmap (sampled)
        st.subheader("Correlation heatmap (pollutants)")
        sample = df.sample(min(3000, len(df)), random_state=42)
        cols_for_corr = [c for c in ["PM2.5","PM10","NO","NO2","NOx","CO","SO2","O3","AQI"] if c in sample.columns]
        fig3, ax3 = plt.subplots(figsize=(10,8))
        sns.heatmap(sample[cols_for_corr].corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax3)
        st.pyplot(fig3)

# ---------- Maps ----------
elif page == "Maps":
    st.title("Geographical Visualisations")
    df, files = load_merged_csvs_from_folder("/content/dataset/dataset")
    if df is None:
        st.warning("No data. Upload first.")
    else:
        df = preprocess(df)
        coords = get_city_coordinates()
        df["Latitude"] = df["City"].apply(lambda c: coords.get(c,[None,None])[0])
        df["Longitude"] = df["City"].apply(lambda c: coords.get(c,[None,None])[1])
        df_geo = df.dropna(subset=["Latitude","Longitude"]).copy()

        st.sidebar.subheader("Map controls")
        map_type = st.sidebar.selectbox("Map type", ["AQI bubble map","PM2.5 heatmap","City markers cluster"])
        agg_by = st.sidebar.selectbox("Aggregate by", ["City","Month","Year"])
        pollutant_map = st.sidebar.selectbox("Map pollutant (for heatmap/circles)", ["AQI","PM2.5","PM10"])

        st.subheader("Interactive map")
        # compute city averages
        if agg_by == "City":
            city_df = df_geo.groupby("City").mean(numeric_only=True).reset_index()
        elif agg_by == "Month":
            city_df = df_geo.groupby(["City","Month"]).mean(numeric_only=True).reset_index()
        else:
            city_df = df_geo.groupby(["City","Year"]).mean(numeric_only=True).reset_index()

        # Base map
        m = folium.Map(location=[22.9734,78.6569], zoom_start=5)
        if map_type == "AQI bubble map":
            for _, row in city_df.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                val = row.get("AQI", np.nan)
                color = "green" if val<=100 else "orange" if val<=200 else "red"
                folium.CircleMarker(location=[lat, lon],
                                    radius=max(4, min(val/10 if np.isfinite(val) else 5, 25)),
                                    color=color, fill=True, fill_opacity=0.7,
                                    popup=f"{row.get('City','')}: AQI {val:.1f}").add_to(m)
            st_folium(m, width=900, height=600)
        elif map_type == "PM2.5 heatmap":
            heat_df = df_geo[["Latitude","Longitude",pollutant_map]].dropna()
            HeatMap(data=heat_df.values.tolist(), radius=12, blur=18).add_to(m)
            st_folium(m, width=900, height=600)
        else:
            mc = MarkerCluster().add_to(m)
            for _, row in city_df.iterrows():
                folium.Marker(location=[row["Latitude"], row["Longitude"]],
                              popup=(f"{row.get('City','')}<br>{pollutant_map}: {row.get(pollutant_map, np.nan):.1f}")).add_to(mc)
            st_folium(m, width=900, height=600)

# ---------- Model & Prediction ----------
elif page == "Model & Prediction":
    st.title("Model training, comparison & prediction")
    df, files = load_merged_csvs_from_folder("/content/dataset/dataset")
    if df is None:
        st.warning("No data. Upload dataset first.")
    else:
        df = preprocess(df)
        st.sidebar.subheader("Model controls")
        do_train = st.sidebar.checkbox("(Re)train models now", value=False)
        show_compare = st.sidebar.checkbox("Show CV comparison", value=True)
        # features
        pollutant_cols = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"] if c in df.columns]
        features = pollutant_cols + ["Year","Month","Day","Weekday","City_Code"]
        X = df[features]
        y = df["AQI"]
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # define models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }
        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        except:
            pass

        # cross-validate if asked
        if show_compare:
            st.subheader("Cross-validated RMSE (5-fold)")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            results = {}
            for name, m in models.items():
                X_use = X_train_scaled if name == "Linear Regression" else X_train
                scores = cross_val_score(m, X_use, y_train, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
                rmse = np.sqrt(-scores)
                results[name] = (rmse.mean(), rmse.std())
            res_df = pd.DataFrame([(k, v[0], v[1]) for k, v in results.items()], columns=["Model","RMSE_Mean","RMSE_STD"])
            res_df = res_df.sort_values("RMSE_Mean")
            st.dataframe(res_df)

        # training (if asked) or load pre-saved
        model_path = "saved_model.joblib"
        pipeline_path = "saved_pipeline.joblib"

        if do_train:
            st.info("Training selected models. This may take a few minutes.")
            best_name = None
            best_rmse = 1e9
            for name, m in models.items():
                X_use = X_train_scaled if name == "Linear Regression" else X_train
                m.fit(X_use, y_train)
                # evaluate on validation (X_test)
                X_eval = X_test_scaled if name == "Linear Regression" else X_test
                preds = m.predict(X_eval)
                rmse = mean_squared_error(y_test, preds, squared=False)
                st.write(f"{name} RMSE (test): {rmse:.3f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_name = name
                    best_model_obj = m
            st.success(f"Best model: {best_name} (RMSE {best_rmse:.3f})")
            # save best model + scaler
            joblib.dump({"model_name":best_name, "model":best_model_obj}, model_path)
            joblib.dump({"scaler":scaler}, pipeline_path)
            st.info(f"Saved model to {model_path}")
        else:
            # try load
            if os.path.exists(model_path):
                mpack = joblib.load(model_path)
                best_name = mpack["model_name"]
                best_model_obj = mpack["model"]
                st.success(f"Loaded saved model: {best_name}")
            else:
                st.info("No saved model found. Check '(Re)train models now' to train and save.")

        # Evaluate loaded/trained model on test set
        if 'best_model_obj' in locals():
            name = best_name
            X_eval = X_test_scaled if name == "Linear Regression" else X_test
            preds = best_model_obj.predict(X_eval)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)
            st.write(f"Evaluation on held-out test set — {name}")
            st.write(f"RMSE: {rmse:.3f} | R2: {r2:.3f}")
            # feature importance if tree-based
            if hasattr(best_model_obj, "feature_importances_"):
                st.subheader("Feature importance")
                fi = pd.DataFrame({"feature": features, "importance": best_model_obj.feature_importances_})
                fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
                st.bar_chart(fi.set_index("feature").head(15))
        # Single-sample prediction UI
        st.subheader("Single-sample prediction")
        st.write("Enter pollutant values (or pick a city + date to autofill mean values).")
        col1, col2 = st.columns(2)
        with col1:
            city_pick = st.selectbox("City for autofill", ["None"] + sorted(df["City"].unique().tolist()))
            date_pick = st.date_input("Date (used for Year/Month/Day)", value=pd.to_datetime("2020-01-01"))
        with col2:
            # input fields for pollutant values
            user_vals = {}
            for p in pollutant_cols:
                user_vals[p] = st.number_input(p, value=float(df[p].median()))
        # autofill if city selected
        if city_pick != "None":
            city_mean = df[df["City"]==city_pick][pollutant_cols].mean()
            for p in pollutant_cols:
                user_vals[p] = float(city_mean[p])
            st.info(f"Autofilled pollutant values using {city_pick} historical mean.")

        if st.button("Predict AQI"):
            if 'best_model_obj' not in locals():
                st.error("No trained model available. Train a model first or enable '(Re)train models now'.")
            else:
                # assemble feature vector
                yy = pd.to_datetime(date_pick)
                feat = []
                for p in pollutant_cols:
                    feat.append(user_vals[p])
                feat += [yy.year, yy.month, yy.day, yy.weekday(), 0]  # city_code=0 placeholder
                X_sample = np.array(feat).reshape(1,-1)
                if best_name == "Linear Regression":
                    X_sample = scaler.transform(X_sample)
                pred = best_model_obj.predict(X_sample)[0]
                st.success(f"Predicted AQI: {pred:.1f}")
                st.write("Use saved model for batch predictions or to integrate into the Streamlit app.")

# ---------- About ----------
elif page == "About":
    st.title("About this app")
    st.markdown("""
    This app:
    - Loads & merges the India city CSVs (place them under `/content/dataset/dataset/` in Colab or upload a zip).
    - Provides interactive EDA, maps, and a model training + prediction interface.
    - Designed to match CMP7005 assessment requirements (EDA, model selection, visualization).
    """)
    st.markdown("Created by: Your Name — use responsibly and cite data sources in your submission.")
