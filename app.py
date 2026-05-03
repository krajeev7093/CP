import re
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

st.set_page_config(
    page_title="Smart Product Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "wishlist" not in st.session_state:
    st.session_state.wishlist = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def apply_theme(dark_mode):
    if not dark_mode:
        return

    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .main,
        .block-container {
            background-color: #0f172a;
            color: #e5e7eb;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #111827;
            color: #e5e7eb;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"] {
            color: #e5e7eb !important;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #cbd5e1 !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            background-color: #1f2937 !important;
            border-color: #374151 !important;
            color: #f8fafc !important;
        }

        div[data-testid="stTabs"] button {
            color: #cbd5e1 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #f87171 !important;
        }

        div[data-testid="stExpander"] {
            background-color: #111827;
            border-color: #374151;
        }

        div[data-testid="stDataFrame"],
        iframe {
            color-scheme: dark;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def first_number(value, default=np.nan):
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else default


def parse_storage_gb(value):
    text = str(value).lower()
    number = first_number(text, 0)
    if "tb" in text:
        return number * 1024
    return number


def parse_bool(value):
    return 1 if str(value).strip().lower() in {"true", "yes", "1"} else 0


def normalize_01(series, higher_is_better=True):
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(1.0, index=series.index)
    scaled = MinMaxScaler().fit_transform(numeric.to_frame()).ravel()
    if not higher_is_better:
        scaled = 1 - scaled
    return pd.Series(scaled, index=series.index)


def extract_price_from_query(query):
    text = query.lower().replace(",", "")
    matches = re.findall(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac)?", text)
    prices = []
    for number, suffix in matches:
        price = float(number)
        if suffix == "k":
            price *= 1000
        elif suffix in {"lakh", "lac"}:
            price *= 100000
        if price >= 1000:
            prices.append(price)
    return max(prices) if prices else None


VIDEO_QUALITY_SCORES = {
    "1080p": 1,
    "2.7K": 2,
    "4K": 3,
    "5.1K": 4,
    "6K": 5,
    "8K": 6,
}

OBSTACLE_SCORES = {
    "No Avoidance": 0,
    "None": 0,
    "Downward": 1,
    "Front/Rear": 2,
    "Tri-Directional": 3,
    "Omnidirectional": 4,
}

GIMBAL_SCORES = {
    "No Gimbal": 0,
    "None": 0,
    "2-axis": 2,
    "3-axis": 3,
}


@st.cache_data
def load_mobile_data():
    df = pd.read_excel("mobile_dataset.xlsx")
    brands_to_remove = ["hmd", "acer", "mtr", "ikall", "itel"]
    df = df[~df["brand_name"].astype(str).str.lower().isin(brands_to_remove)].copy()

    df = df.rename(
        columns={
            "brand_name": "Brand",
            "model": "Model",
            "ram_support": "RAM",
            "rom_GB": "Storage",
            "battery_capacity": "Battery",
            "rating": "Average Rating",
            "price": "Price",
            "has_5g": "5G",
            "display_size": "Screen Size",
            "rear_camera": "Rear Camera",
            "front_camera": "Front Camera",
            "refresh_rates": "Refresh Rate",
        }
    )

    df["Product Type"] = "Phone"
    df["Brand"] = df["Brand"].astype(str).str.title()
    df["Rear Camera"] = df["Rear Camera"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Front Camera"] = df["Front Camera"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Processor Score"] = pd.to_numeric(df.get("processor_speed_Hz"), errors="coerce")
    df["Fast Charging"] = df.get("fast_charging", False).apply(parse_bool)
    df["5G"] = df["5G"].apply(parse_bool)

    numeric_cols = [
        "RAM",
        "Storage",
        "Battery",
        "Average Rating",
        "Price",
        "Screen Size",
        "Rear Camera",
        "Front Camera",
        "Refresh Rate",
        "Processor Score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Storage"] = df["Storage"].replace(1, 1024)
    df["Average Rating"] = df["Average Rating"] / 20
    df["Average Rating"] = df["Average Rating"].clip(0, 5)
    df["Refresh Rate"] = df["Refresh Rate"].fillna(60)
    df["Processor Score"] = df["Processor Score"].fillna(df["Processor Score"].median())

    fill_cols = [
        "RAM",
        "Storage",
        "Battery",
        "Average Rating",
        "Price",
        "Screen Size",
        "Rear Camera",
        "Front Camera",
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(df[col].median())

    df["Battery Label"] = df["Battery"].map(lambda x: f"{x:.0f} mAh")
    df["Storage Label"] = df["Storage"].map(lambda x: f"{x:.0f} GB")
    df["Extra 1"] = df["Rear Camera"].map(lambda x: f"{x:.0f} MP rear camera")
    df["Extra 2"] = df["5G"].map(lambda x: "5G" if x == 1 else "4G")
    return df


@st.cache_data
def load_laptop_data():
    df = pd.read_excel("laptop_dataset.xlsx")
    df = df.rename(
        columns={
            "Brand Name": "Brand",
            "Updated Price (₹)": "Price",
            "Rating (out of 5)": "Average Rating",
            "Battery Life (hours)": "Battery",
            "Refresh Rate": "Refresh Rate",
            "GPU (Integrated/RTX/GTX)": "GPU",
        }
    )

    df["Product Type"] = "Laptop"
    df["Brand"] = df["Brand"].astype(str).str.title()
    df["RAM"] = df["RAM"].apply(first_number)
    df["Storage"] = df["Storage"].apply(parse_storage_gb)
    df["Screen Size"] = df["Screen Size"].apply(first_number)
    df["Refresh Rate"] = df["Refresh Rate"].apply(first_number)
    df["Weight Value"] = pd.to_numeric(df["Weight (kg)"], errors="coerce")
    df["Touchscreen"] = df["Touchscreen"].apply(parse_bool)
    df["Fingerprint Sensor"] = df["Fingerprint Sensor"].apply(parse_bool)

    gpu_text = df["GPU"].astype(str).str.lower()
    df["GPU Score"] = np.select(
        [
            gpu_text.str.contains("4090|4080|3080|4070"),
            gpu_text.str.contains("4060|3070|3060|1660|1650"),
            gpu_text.str.contains("rtx|gtx"),
        ],
        [10, 8, 6],
        default=3,
    )

    processor_text = df["Processor"].astype(str).str.lower()
    df["Processor Score"] = np.select(
        [
            processor_text.str.contains("i9|ryzen 9|ultra 9"),
            processor_text.str.contains("i7|ryzen 7|ultra 7"),
            processor_text.str.contains("i5|ryzen 5|ultra 5"),
        ],
        [10, 8, 6],
        default=4,
    )

    numeric_cols = [
        "RAM",
        "Storage",
        "Battery",
        "Average Rating",
        "Price",
        "Screen Size",
        "Refresh Rate",
        "Weight Value",
        "GPU Score",
        "Processor Score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["Battery Label"] = df["Battery"].map(lambda x: f"{x:.0f} hours")
    df["Storage Label"] = df["Storage"].map(lambda x: f"{x:.0f} GB")
    df["Extra 1"] = df["GPU"].astype(str)
    df["Extra 2"] = df["Processor"].astype(str)
    return df


@st.cache_data
def load_drone_data():
    df = pd.read_excel("drone_dataset.xlsx", keep_default_na=False)
    df = df.rename(
        columns={
            "Brand Name": "Brand",
            "Price (Rs)": "Price",
            "Rating (out of 5)": "Average Rating",
            "Camera Resolution (MP)": "Camera MP",
            "Max Flight Time (min)": "Battery",
            "Max Range (km)": "Range",
            "Max Speed (km/h)": "Max Speed",
            "Weight (g)": "Weight Value",
        }
    )

    df["Product Type"] = "Drone"
    df["Brand"] = df["Brand"].astype(str).str.title()
    df["GPS"] = df["GPS"].apply(parse_bool)
    df["Foldable"] = df["Foldable"].apply(parse_bool)
    df["Follow Me Mode"] = df["Follow Me Mode"].apply(parse_bool)
    df["Return To Home"] = df["Return To Home"].apply(parse_bool)
    df["Video Resolution"] = df["Video Resolution"].astype(str).str.strip()
    df["Obstacle Avoidance"] = df["Obstacle Avoidance"].replace("", "No Avoidance").astype(str)
    df["Gimbal Stabilization"] = df["Gimbal Stabilization"].replace("", "No Gimbal").astype(str)
    df["Video Quality Score"] = df["Video Resolution"].map(VIDEO_QUALITY_SCORES).fillna(1)
    df["Obstacle Score"] = df["Obstacle Avoidance"].map(OBSTACLE_SCORES).fillna(0)
    df["Gimbal Score"] = df["Gimbal Stabilization"].map(GIMBAL_SCORES).fillna(0)

    numeric_cols = [
        "Camera MP",
        "Battery",
        "Range",
        "Max Speed",
        "Weight Value",
        "Price",
        "Average Rating",
        "Wind Resistance Level",
        "Battery Capacity (mAh)",
        "Video Quality Score",
        "Obstacle Score",
        "Gimbal Score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["Battery Label"] = df["Battery"].map(lambda x: f"{x:.0f} min flight")
    df["Extra 1"] = df.apply(
        lambda row: f"{row['Camera MP']:.0f} MP camera / {row['Video Resolution']} video",
        axis=1,
    )
    df["Extra 2"] = df.apply(
        lambda row: f"{row['Range']:.1f} km range / {row['Gimbal Stabilization']} gimbal",
        axis=1,
    )
    return df


def score_products(data, product_type, use_case):
    scored = data.copy()

    scored["Rating Score"] = normalize_01(scored["Average Rating"])
    scored["Battery Score"] = normalize_01(scored["Battery"])
    scored["Price Score"] = normalize_01(scored["Price"], higher_is_better=False)

    if product_type == "Phone":
        scored["RAM Score"] = normalize_01(scored["RAM"])
        scored["Storage Score"] = normalize_01(scored["Storage"])
        scored["Refresh Score"] = normalize_01(scored["Refresh Rate"])
        scored["Screen Score"] = normalize_01(scored["Screen Size"])
        scored["Processor Norm"] = normalize_01(scored["Processor Score"])
        scored["Camera Score"] = normalize_01(scored["Rear Camera"])
        weights = {
            "Balanced": {
                "Rating Score": 0.25,
                "RAM Score": 0.15,
                "Storage Score": 0.15,
                "Battery Score": 0.15,
                "Camera Score": 0.20,
                "Price Score": 0.10,
            },
            "Gaming": {
                "RAM Score": 0.25,
                "Processor Norm": 0.25,
                "Refresh Score": 0.20,
                "Battery Score": 0.20,
                "Rating Score": 0.10,
            },
            "Photography": {
                "Camera Score": 0.45,
                "Rating Score": 0.20,
                "Storage Score": 0.15,
                "RAM Score": 0.10,
                "Battery Score": 0.10,
            },
            "Productivity": {
                "RAM Score": 0.25,
                "Storage Score": 0.25,
                "Battery Score": 0.20,
                "Rating Score": 0.20,
                "Screen Score": 0.10,
            },
            "Budget": {
                "Price Score": 0.45,
                "Battery Score": 0.20,
                "RAM Score": 0.15,
                "Storage Score": 0.10,
                "Rating Score": 0.10,
            },
        }[use_case]
        feature_cols = [
            "RAM",
            "Storage",
            "Battery",
            "Average Rating",
            "Price",
            "Refresh Rate",
            "Rear Camera",
            "Processor Score",
        ]
    elif product_type == "Laptop":
        scored["RAM Score"] = normalize_01(scored["RAM"])
        scored["Storage Score"] = normalize_01(scored["Storage"])
        scored["Refresh Score"] = normalize_01(scored["Refresh Rate"])
        scored["Screen Score"] = normalize_01(scored["Screen Size"])
        scored["Processor Norm"] = normalize_01(scored["Processor Score"])
        scored["GPU Norm"] = normalize_01(scored["GPU Score"])
        scored["Weight Score"] = normalize_01(scored["Weight Value"], higher_is_better=False)
        weights = {
            "Balanced": {
                "Rating Score": 0.20,
                "Processor Norm": 0.20,
                "RAM Score": 0.15,
                "Storage Score": 0.15,
                "Battery Score": 0.15,
                "Price Score": 0.15,
            },
            "Gaming": {
                "GPU Norm": 0.35,
                "Processor Norm": 0.25,
                "RAM Score": 0.15,
                "Refresh Score": 0.15,
                "Battery Score": 0.10,
            },
            "Coding": {
                "Processor Norm": 0.25,
                "RAM Score": 0.25,
                "Storage Score": 0.20,
                "Battery Score": 0.15,
                "Rating Score": 0.15,
            },
            "Design": {
                "GPU Norm": 0.25,
                "Processor Norm": 0.20,
                "RAM Score": 0.20,
                "Screen Score": 0.15,
                "Storage Score": 0.10,
                "Rating Score": 0.10,
            },
            "Student/Budget": {
                "Price Score": 0.35,
                "Battery Score": 0.20,
                "Weight Score": 0.20,
                "Rating Score": 0.15,
                "RAM Score": 0.10,
            },
        }[use_case]
        feature_cols = [
            "RAM",
            "Storage",
            "Battery",
            "Average Rating",
            "Price",
            "Refresh Rate",
            "GPU Score",
            "Processor Score",
            "Weight Value",
        ]
    else:
        scored["Camera Score"] = normalize_01(scored["Camera MP"])
        scored["Video Score"] = normalize_01(scored["Video Quality Score"])
        scored["Range Score"] = normalize_01(scored["Range"])
        scored["Speed Score"] = normalize_01(scored["Max Speed"])
        scored["Weight Score"] = normalize_01(scored["Weight Value"], higher_is_better=False)
        scored["Obstacle Norm"] = normalize_01(scored["Obstacle Score"])
        scored["Gimbal Norm"] = normalize_01(scored["Gimbal Score"])
        scored["Wind Score"] = normalize_01(scored["Wind Resistance Level"])
        scored["GPS Score"] = pd.to_numeric(scored["GPS"], errors="coerce").fillna(0)
        scored["Foldable Score"] = pd.to_numeric(scored["Foldable"], errors="coerce").fillna(0)
        weights = {
            "Balanced": {
                "Rating Score": 0.18,
                "Camera Score": 0.16,
                "Video Score": 0.12,
                "Battery Score": 0.14,
                "Range Score": 0.14,
                "Obstacle Norm": 0.12,
                "Price Score": 0.14,
            },
            "Aerial Photography": {
                "Camera Score": 0.28,
                "Video Score": 0.22,
                "Gimbal Norm": 0.18,
                "Obstacle Norm": 0.10,
                "Battery Score": 0.12,
                "Rating Score": 0.10,
            },
            "Travel": {
                "Weight Score": 0.25,
                "Foldable Score": 0.15,
                "Battery Score": 0.18,
                "Range Score": 0.15,
                "Price Score": 0.15,
                "Rating Score": 0.12,
            },
            "Racing": {
                "Speed Score": 0.35,
                "Weight Score": 0.20,
                "Video Score": 0.10,
                "Range Score": 0.10,
                "Rating Score": 0.10,
                "Price Score": 0.15,
            },
            "Professional": {
                "Camera Score": 0.20,
                "Video Score": 0.18,
                "Range Score": 0.16,
                "Obstacle Norm": 0.16,
                "Gimbal Norm": 0.14,
                "Wind Score": 0.08,
                "Rating Score": 0.08,
            },
            "Budget": {
                "Price Score": 0.40,
                "Battery Score": 0.18,
                "Camera Score": 0.14,
                "Range Score": 0.12,
                "GPS Score": 0.08,
                "Rating Score": 0.08,
            },
        }[use_case]
        feature_cols = [
            "Camera MP",
            "Video Quality Score",
            "Battery",
            "Range",
            "Max Speed",
            "Weight Value",
            "Price",
            "Average Rating",
            "Obstacle Score",
            "Gimbal Score",
            "Wind Resistance Level",
            "GPS",
            "Foldable",
        ]

    scored["Use_Case_Score"] = sum(scored[col] * weight for col, weight in weights.items())

    feature_frame = scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(scored) > 1:
        features_scaled = MinMaxScaler().fit_transform(feature_frame)
        similarity = cosine_similarity(features_scaled).mean(axis=1)
    else:
        similarity = np.ones(len(scored))

    scored["Final_Score"] = scored["Use_Case_Score"] * 70 + similarity * 30
    return scored.sort_values("Final_Score", ascending=False)


def apply_query_filters(data, product_type, query):
    filtered = data.copy()
    text = query.lower()
    detected = []

    extracted_price = extract_price_from_query(text)
    if extracted_price:
        filtered = filtered[filtered["Price"] <= extracted_price]
        detected.append(f"under Rs {extracted_price:.0f}")

    keyword_rules = [("budget", lambda d: d[d["Price"] <= d["Price"].quantile(0.35)])]

    if product_type == "Phone":
        keyword_rules += [
            ("high ram", lambda d: d[d["RAM"] >= 8]),
            ("large screen", lambda d: d[d["Screen Size"] >= d["Screen Size"].median()]),
            ("battery", lambda d: d[d["Battery"] >= d["Battery"].quantile(0.65)]),
            ("gaming", lambda d: d[(d["RAM"] >= 8) & (d["Refresh Rate"] >= 120)]),
            ("photography", lambda d: d[d["Rear Camera"] >= 50]),
            ("camera", lambda d: d[d["Rear Camera"] >= 50]),
            ("5g", lambda d: d[d["5G"] == 1]),
            ("fast charging", lambda d: d[d["Fast Charging"] == 1]),
        ]
    elif product_type == "Laptop":
        keyword_rules += [
            ("high ram", lambda d: d[d["RAM"] >= 16]),
            ("large screen", lambda d: d[d["Screen Size"] >= d["Screen Size"].median()]),
            ("battery", lambda d: d[d["Battery"] >= d["Battery"].quantile(0.65)]),
            ("gaming", lambda d: d[(d["RAM"] >= 16) & (d["Refresh Rate"] >= 120)]),
            ("rtx", lambda d: d[d["GPU"].astype(str).str.contains("rtx", case=False, na=False)]),
            ("lightweight", lambda d: d[d["Weight Value"] <= d["Weight Value"].quantile(0.35)]),
            ("touchscreen", lambda d: d[d["Touchscreen"] == 1]),
            ("coding", lambda d: d[(d["RAM"] >= 16) & (d["Processor Score"] >= 6)]),
            ("design", lambda d: d[(d["RAM"] >= 16) & (d["GPU Score"] >= 6)]),
        ]
    else:
        keyword_rules += [
            ("aerial photography", lambda d: d[(d["Camera MP"] >= 20) & (d["Video Quality Score"] >= 3) & (d["Gimbal Score"] >= 2)]),
            ("photography", lambda d: d[(d["Camera MP"] >= 20) & (d["Video Quality Score"] >= 3)]),
            ("camera", lambda d: d[d["Camera MP"] >= 20]),
            ("8k", lambda d: d[d["Video Quality Score"] >= 6]),
            ("6k", lambda d: d[d["Video Quality Score"] >= 5]),
            ("4k", lambda d: d[d["Video Quality Score"] >= 3]),
            ("long range", lambda d: d[d["Range"] >= d["Range"].quantile(0.70)]),
            ("long flight", lambda d: d[d["Battery"] >= d["Battery"].quantile(0.70)]),
            ("battery", lambda d: d[d["Battery"] >= d["Battery"].quantile(0.65)]),
            ("travel", lambda d: d[(d["Weight Value"] <= d["Weight Value"].quantile(0.40)) & (d["Foldable"] == 1)]),
            ("portable", lambda d: d[(d["Weight Value"] <= d["Weight Value"].quantile(0.40)) & (d["Foldable"] == 1)]),
            ("lightweight", lambda d: d[d["Weight Value"] <= d["Weight Value"].quantile(0.35)]),
            ("racing", lambda d: d[d["Max Speed"] >= d["Max Speed"].quantile(0.75)]),
            ("fast", lambda d: d[d["Max Speed"] >= d["Max Speed"].quantile(0.70)]),
            ("gps", lambda d: d[d["GPS"] == 1]),
            ("obstacle", lambda d: d[d["Obstacle Score"] > 0]),
            ("gimbal", lambda d: d[d["Gimbal Score"] >= 2]),
            ("professional", lambda d: d[(d["Camera MP"] >= 48) & (d["Video Quality Score"] >= 4) & (d["Gimbal Score"] >= 3)]),
        ]

    for label, rule in keyword_rules:
        if label in text:
            filtered = rule(filtered)
            detected.append(label)

    return filtered, detected


def score_average(scored, weighted_cols):
    total_weight = sum(weight for _, weight in weighted_cols)
    if total_weight == 0:
        return pd.Series(0.0, index=scored.index)
    total = pd.Series(0.0, index=scored.index)
    for col, weight in weighted_cols:
        if col in scored.columns:
            total += pd.to_numeric(scored[col], errors="coerce").fillna(0) * weight
    return total / total_weight


def apply_recommendation_style(scored, product_type, style, catalog, feedback):
    enhanced = scored.copy()
    enhanced["Base_Score"] = enhanced["Final_Score"]

    if product_type == "Phone":
        enhanced["Portability Score"] = normalize_01(enhanced["Screen Size"], higher_is_better=False)
        performance_cols = [
            ("RAM Score", 0.22),
            ("Processor Norm", 0.22),
            ("Refresh Score", 0.18),
            ("Camera Score", 0.18),
            ("Battery Score", 0.20),
        ]
        premium_cols = [
            ("RAM Score", 0.20),
            ("Storage Score", 0.18),
            ("Camera Score", 0.24),
            ("Processor Norm", 0.20),
            ("Rating Score", 0.18),
        ]
        portable_cols = [("Portability Score", 0.45), ("Battery Score", 0.20), ("Rating Score", 0.20), ("Price Score", 0.15)]
    elif product_type == "Laptop":
        performance_cols = [
            ("Processor Norm", 0.28),
            ("GPU Norm", 0.28),
            ("RAM Score", 0.20),
            ("Refresh Score", 0.12),
            ("Storage Score", 0.12),
        ]
        premium_cols = [
            ("Processor Norm", 0.24),
            ("GPU Norm", 0.24),
            ("RAM Score", 0.18),
            ("Storage Score", 0.14),
            ("Rating Score", 0.20),
        ]
        portable_cols = [("Weight Score", 0.42), ("Battery Score", 0.24), ("Rating Score", 0.18), ("Price Score", 0.16)]
    else:
        performance_cols = [
            ("Camera Score", 0.18),
            ("Video Score", 0.18),
            ("Range Score", 0.18),
            ("Speed Score", 0.16),
            ("Obstacle Norm", 0.15),
            ("Gimbal Norm", 0.15),
        ]
        premium_cols = [
            ("Camera Score", 0.22),
            ("Video Score", 0.20),
            ("Range Score", 0.18),
            ("Obstacle Norm", 0.16),
            ("Gimbal Norm", 0.16),
            ("Rating Score", 0.08),
        ]
        portable_cols = [("Weight Score", 0.36), ("Foldable Score", 0.24), ("Battery Score", 0.18), ("Range Score", 0.12), ("Price Score", 0.10)]

    style_map = {
        "Balanced AI": [("Use_Case_Score", 1.0)],
        "Best Value": [("Price Score", 0.45), ("Rating Score", 0.25), ("Use_Case_Score", 0.30)],
        "Performance First": performance_cols,
        "Budget Pick": [("Price Score", 0.65), ("Rating Score", 0.20), ("Battery Score", 0.15)],
        "Premium Choice": premium_cols,
        "Portable Choice": portable_cols,
    }

    enhanced["Style_Score"] = score_average(enhanced, style_map[style]) * 100

    liked_models = []
    for item in feedback:
        product_key = item.get("product", "")
        rating = item.get("rating", 0)
        if product_key.startswith(f"{product_type}:") and rating >= 4:
            liked_models.append(product_key.split(":", 1)[1])

    liked_brands = set(catalog[catalog["Model"].isin(liked_models)]["Brand"].dropna().astype(str))
    enhanced["Preference_Boost"] = 0.0
    if liked_models:
        enhanced.loc[enhanced["Model"].isin(liked_models), "Preference_Boost"] = 100
    if liked_brands:
        enhanced.loc[enhanced["Brand"].astype(str).isin(liked_brands), "Preference_Boost"] = enhanced[
            "Preference_Boost"
        ].clip(lower=55)

    enhanced["Final_Score"] = (
        enhanced["Base_Score"] * 0.72
        + enhanced["Style_Score"] * 0.23
        + enhanced["Preference_Boost"] * 0.05
    )
    return enhanced.sort_values("Final_Score", ascending=False)


def product_summary(product, product_type, use_case, style):
    if product_type == "Phone":
        return (
            f"Best suited for {use_case.lower()} buyers who want {product['RAM']:.0f} GB RAM, "
            f"{product['Storage']:.0f} GB storage, a {product['Rear Camera']:.0f} MP camera, "
            f"and a {product['Battery']:.0f} mAh battery at Rs {product['Price']:.0f}. "
            f"The {style.lower()} ranking gives extra credit to its strongest matching specs."
        )
    if product_type == "Laptop":
        return (
            f"Best suited for {use_case.lower()} work with {product['RAM']:.0f} GB RAM, "
            f"{product['Storage']:.0f} GB storage, {product['Processor']}, and {product['GPU']}. "
            f"It balances performance, battery life, and price under the {style.lower()} profile."
        )
    return (
        f"Best suited for {use_case.lower()} flying with a {product['Camera MP']:.0f} MP camera, "
        f"{product['Video Resolution']} video, {product['Battery']:.0f} minutes of flight time, "
        f"{product['Range']:.1f} km range, and {product['Gimbal Stabilization']} stabilization."
    )


def score_breakdown(product, product_type):
    if product_type == "Phone":
        items = [
            ("Price value", product.get("Price Score", 0)),
            ("Camera", product.get("Camera Score", 0)),
            ("Battery", product.get("Battery Score", 0)),
            ("RAM", product.get("RAM Score", 0)),
            ("Storage", product.get("Storage Score", 0)),
            ("Rating", product.get("Rating Score", 0)),
        ]
    elif product_type == "Laptop":
        items = [
            ("Price value", product.get("Price Score", 0)),
            ("Processor", product.get("Processor Norm", 0)),
            ("GPU", product.get("GPU Norm", 0)),
            ("RAM", product.get("RAM Score", 0)),
            ("Battery", product.get("Battery Score", 0)),
            ("Portability", product.get("Weight Score", 0)),
        ]
    else:
        items = [
            ("Price value", product.get("Price Score", 0)),
            ("Camera", product.get("Camera Score", 0)),
            ("Video", product.get("Video Score", 0)),
            ("Flight time", product.get("Battery Score", 0)),
            ("Range", product.get("Range Score", 0)),
            ("Portability", product.get("Weight Score", 0)),
        ]
    return sorted(items, key=lambda item: item[1], reverse=True)


def top_reasons(product, product_type):
    strengths = [label for label, value in score_breakdown(product, product_type)[:3] if value >= 0.55]
    if not strengths:
        strengths = [label for label, _ in score_breakdown(product, product_type)[:3]]
    return "Strongest match: " + ", ".join(strengths) + "."


def comparison_specs(product_type):
    if product_type == "Phone":
        return {
            "Price": ("Price", False),
            "Rating": ("Average Rating", True),
            "RAM": ("RAM", True),
            "Storage": ("Storage", True),
            "Battery": ("Battery", True),
            "Rear Camera": ("Rear Camera", True),
            "Refresh Rate": ("Refresh Rate", True),
        }
    if product_type == "Laptop":
        return {
            "Price": ("Price", False),
            "Rating": ("Average Rating", True),
            "RAM": ("RAM", True),
            "Storage": ("Storage", True),
            "Battery": ("Battery", True),
            "GPU": ("GPU Score", True),
            "Processor": ("Processor Score", True),
            "Portability": ("Weight Value", False),
        }
    return {
        "Price": ("Price", False),
        "Rating": ("Average Rating", True),
        "Camera": ("Camera MP", True),
        "Video": ("Video Quality Score", True),
        "Flight Time": ("Battery", True),
        "Range": ("Range", True),
        "Speed": ("Max Speed", True),
        "Portability": ("Weight Value", False),
        "Obstacle Avoidance": ("Obstacle Score", True),
    }


def build_winner_table(compare_source, product_type):
    rows = []
    for label, (col, higher_is_better) in comparison_specs(product_type).items():
        if col not in compare_source.columns:
            continue
        values = pd.to_numeric(compare_source[col], errors="coerce")
        if values.isna().all():
            continue
        best_value = values.max() if higher_is_better else values.min()
        winners = compare_source.loc[values == best_value, "Model"].tolist()
        rows.append(
            {
                "Spec": label,
                "Winner": ", ".join(winners),
                "Winning Value": round(float(best_value), 2),
            }
        )
    return pd.DataFrame(rows)


def dashboard_metric_columns(product_type):
    if product_type == "Phone":
        return ["Price", "Average Rating", "RAM", "Storage", "Battery", "Rear Camera"]
    if product_type == "Laptop":
        return ["Price", "Average Rating", "RAM", "Storage", "Battery", "GPU Score", "Processor Score"]
    return ["Price", "Average Rating", "Camera MP", "Battery", "Range", "Max Speed"]


mobile_df = load_mobile_data()
laptop_df = load_laptop_data()
drone_df = load_drone_data()

st.sidebar.header("Filters & Preferences")
st.session_state.dark_mode = st.sidebar.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode,
)
apply_theme(st.session_state.dark_mode)
px.defaults.template = "plotly_dark" if st.session_state.dark_mode else "plotly"

st.title("Smart Product Recommendation System")

product_type = st.sidebar.radio("Product dataset", ["Phone", "Laptop", "Drone"], horizontal=True)
df = {"Phone": mobile_df, "Laptop": laptop_df, "Drone": drone_df}[product_type]

if product_type == "Phone":
    use_case_options = ["Balanced", "Gaming", "Photography", "Productivity", "Budget"]
    query_placeholder = "budget gaming phone under Rs 30000"
elif product_type == "Laptop":
    use_case_options = ["Balanced", "Gaming", "Coding", "Design", "Student/Budget"]
    query_placeholder = "lightweight coding laptop with RTX under Rs 120000"
else:
    use_case_options = ["Balanced", "Aerial Photography", "Travel", "Racing", "Professional", "Budget"]
    query_placeholder = "portable 4K drone with obstacle avoidance under Rs 90000"

use_case = st.sidebar.selectbox("What's your priority?", use_case_options)
recommendation_style = st.sidebar.selectbox(
    "Recommendation style",
    ["Balanced AI", "Best Value", "Performance First", "Budget Pick", "Premium Choice", "Portable Choice"],
)
learn_from_feedback = st.sidebar.checkbox("Learn from my likes", value=True)

st.sidebar.subheader("Advanced Filters")
price_floor = int(max(0, df["Price"].min() // 1000 * 1000))
price_ceiling = int(np.ceil(df["Price"].max() / 1000) * 1000)
default_max = int(min(price_ceiling, df["Price"].quantile(0.75)))

col1, col2 = st.sidebar.columns(2)
with col1:
    price_min = st.number_input("Min Price (Rs)", price_floor, price_ceiling, price_floor, step=1000)
with col2:
    price_max = st.number_input("Max Price (Rs)", price_floor, price_ceiling, default_max, step=1000)

brand = st.sidebar.selectbox("Brand", ["All"] + sorted(df["Brand"].dropna().unique()))

if product_type == "Phone":
    ram_min_value = int(max(1, df["RAM"].min()))
    ram_max_value = int(max(ram_min_value, df["RAM"].max()))
    ram = st.sidebar.slider("Minimum RAM (GB)", ram_min_value, ram_max_value, 4)

    storage_values = sorted(df["Storage"].dropna().astype(int).unique())
    storage = st.sidebar.slider("Minimum Storage (GB)", int(min(storage_values)), int(max(storage_values)), 128)

    refresh_rate = st.sidebar.slider(
        "Minimum Refresh Rate (Hz)",
        int(df["Refresh Rate"].min()),
        int(df["Refresh Rate"].max()),
        90,
    )
    screen_min, screen_max = st.sidebar.slider(
        "Screen Size (inches)",
        float(df["Screen Size"].min()),
        float(df["Screen Size"].max()),
        (5.5, 6.7),
    )
    battery = st.sidebar.slider("Minimum Battery (mAh)", int(df["Battery"].min()), int(df["Battery"].max()), 4000)
    fiveg = st.sidebar.checkbox("5G Support")
    camera = st.sidebar.slider("Minimum Rear Camera (MP)", int(df["Rear Camera"].min()), int(df["Rear Camera"].max()), 12)
elif product_type == "Laptop":
    ram_min_value = int(max(1, df["RAM"].min()))
    ram_max_value = int(max(ram_min_value, df["RAM"].max()))
    ram = st.sidebar.slider("Minimum RAM (GB)", ram_min_value, ram_max_value, 8)

    storage_values = sorted(df["Storage"].dropna().astype(int).unique())
    storage = st.sidebar.slider("Minimum Storage (GB)", int(min(storage_values)), int(max(storage_values)), 512)

    refresh_rate = st.sidebar.slider(
        "Minimum Refresh Rate (Hz)",
        int(df["Refresh Rate"].min()),
        int(df["Refresh Rate"].max()),
        60,
    )
    screen_min, screen_max = st.sidebar.slider(
        "Screen Size (inches)",
        float(df["Screen Size"].min()),
        float(df["Screen Size"].max()),
        (14.0, 16.0),
    )
    battery = st.sidebar.slider("Minimum Battery Life (hours)", int(df["Battery"].min()), int(df["Battery"].max()), 6)
    dedicated_gpu = st.sidebar.checkbox("Dedicated RTX/GTX GPU")
    touchscreen = st.sidebar.checkbox("Touchscreen")
    max_weight = st.sidebar.slider("Maximum Weight (kg)", float(df["Weight Value"].min()), float(df["Weight Value"].max()), float(df["Weight Value"].max()))
else:
    camera = st.sidebar.slider(
        "Minimum Camera Resolution (MP)",
        int(df["Camera MP"].min()),
        int(df["Camera MP"].max()),
        12,
    )
    min_video = st.sidebar.selectbox("Minimum Video Resolution", list(VIDEO_QUALITY_SCORES.keys()), index=2)
    flight_time = st.sidebar.slider(
        "Minimum Flight Time (min)",
        int(df["Battery"].min()),
        int(df["Battery"].max()),
        20,
    )
    min_range = st.sidebar.slider(
        "Minimum Range (km)",
        float(df["Range"].min()),
        float(df["Range"].max()),
        2.0,
    )
    max_speed = st.sidebar.slider(
        "Minimum Speed (km/h)",
        int(df["Max Speed"].min()),
        int(df["Max Speed"].max()),
        int(df["Max Speed"].min()),
    )
    max_weight = st.sidebar.slider(
        "Maximum Weight (g)",
        int(df["Weight Value"].min()),
        int(df["Weight Value"].max()),
        int(df["Weight Value"].max()),
    )
    obstacle_avoidance = st.sidebar.checkbox("Obstacle Avoidance")
    gps_required = st.sidebar.checkbox("GPS")
    gimbal_required = st.sidebar.checkbox("2-axis/3-axis Gimbal")
    foldable_required = st.sidebar.checkbox("Foldable")

st.sidebar.subheader("AI Query Assistant")
user_query = st.sidebar.text_input(
    f"Describe your ideal {product_type.lower()}",
    placeholder=query_placeholder,
)

filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["Price"] >= price_min) & (filtered_df["Price"] <= price_max)]

if brand != "All":
    filtered_df = filtered_df[filtered_df["Brand"] == brand]

if product_type == "Phone":
    filtered_df = filtered_df[filtered_df["RAM"] >= ram]
    filtered_df = filtered_df[filtered_df["Storage"] >= storage]
    filtered_df = filtered_df[filtered_df["Battery"] >= battery]
    filtered_df = filtered_df[filtered_df["Refresh Rate"] >= refresh_rate]
    filtered_df = filtered_df[(filtered_df["Screen Size"] >= screen_min) & (filtered_df["Screen Size"] <= screen_max)]
    filtered_df = filtered_df[filtered_df["Rear Camera"] >= camera]
    if fiveg:
        filtered_df = filtered_df[filtered_df["5G"] == 1]
elif product_type == "Laptop":
    filtered_df = filtered_df[filtered_df["RAM"] >= ram]
    filtered_df = filtered_df[filtered_df["Storage"] >= storage]
    filtered_df = filtered_df[filtered_df["Battery"] >= battery]
    filtered_df = filtered_df[filtered_df["Refresh Rate"] >= refresh_rate]
    filtered_df = filtered_df[(filtered_df["Screen Size"] >= screen_min) & (filtered_df["Screen Size"] <= screen_max)]
    filtered_df = filtered_df[filtered_df["Weight Value"] <= max_weight]
    if dedicated_gpu:
        filtered_df = filtered_df[filtered_df["GPU"].astype(str).str.contains("rtx|gtx", case=False, na=False)]
    if touchscreen:
        filtered_df = filtered_df[filtered_df["Touchscreen"] == 1]
else:
    filtered_df = filtered_df[filtered_df["Camera MP"] >= camera]
    filtered_df = filtered_df[filtered_df["Video Quality Score"] >= VIDEO_QUALITY_SCORES[min_video]]
    filtered_df = filtered_df[filtered_df["Battery"] >= flight_time]
    filtered_df = filtered_df[filtered_df["Range"] >= min_range]
    filtered_df = filtered_df[filtered_df["Max Speed"] >= max_speed]
    filtered_df = filtered_df[filtered_df["Weight Value"] <= max_weight]
    if obstacle_avoidance:
        filtered_df = filtered_df[filtered_df["Obstacle Score"] > 0]
    if gps_required:
        filtered_df = filtered_df[filtered_df["GPS"] == 1]
    if gimbal_required:
        filtered_df = filtered_df[filtered_df["Gimbal Score"] >= 2]
    if foldable_required:
        filtered_df = filtered_df[filtered_df["Foldable"] == 1]

if user_query:
    filtered_df, detected_intents = apply_query_filters(filtered_df, product_type, user_query)
    if detected_intents:
        st.sidebar.write("Detected: " + ", ".join(detected_intents))

filtered_df = filtered_df.reset_index(drop=True)


if filtered_df.empty:
    st.error(f"No {product_type.lower()}s match your filters. Try adjusting your preferences.")
else:
    ranked_df = score_products(filtered_df, product_type, use_case)
    ranked_df = apply_recommendation_style(
        ranked_df,
        product_type,
        recommendation_style,
        df,
        st.session_state.feedback if learn_from_feedback else [],
    )
    top_products = ranked_df.head(15)
    product_label = {"Phone": "Phones", "Laptop": "Laptops", "Drone": "Drones"}[product_type]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Top Picks", "Dashboard", "Visualizations", "Compare", "Wishlist", "Feedback"]
    )

    with tab1:
        st.subheader(f"Top {product_label} for {use_case}")
        st.caption(f"Ranking profile: {recommendation_style}. Feedback learning: {'on' if learn_from_feedback else 'off'}.")
        download_cols = [
            col
            for col in ["Brand", "Model", "Price", "Average Rating", "Final_Score", "Base_Score", "Style_Score"]
            if col in top_products.columns
        ]
        csv_buffer = StringIO()
        top_products[download_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Top Recommendations (CSV)",
            csv_buffer.getvalue(),
            file_name=f"{product_type.lower()}_top_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
        )

        for idx, (row_index, product) in enumerate(top_products.iterrows(), 1):
            item_key = f"{product_type}:{product['Model']}"
            title = (
                f"#{idx} {product['Brand']} {product['Model']} - "
                f"Rs {product['Price']:.0f} | {product['Average Rating']:.1f}/5"
            )
            with st.expander(title):
                col1, col2, col3 = st.columns(3)
                if product_type == "Drone":
                    with col1:
                        st.metric("Price", f"Rs {product['Price']:.0f}")
                        st.metric("Camera", f"{product['Camera MP']:.0f} MP")
                        st.metric("Video", product["Video Resolution"])
                    with col2:
                        st.metric("Flight Time", product["Battery Label"])
                        st.metric("Range", f"{product['Range']:.1f} km")
                        st.metric("Rating", f"{product['Average Rating']:.1f}/5")
                    with col3:
                        st.metric("Weight", f"{product['Weight Value']:.0f} g")
                        st.metric("Speed", f"{product['Max Speed']:.0f} km/h")
                        st.metric("Score", f"{product['Final_Score']:.1f}")
                else:
                    with col1:
                        st.metric("Price", f"Rs {product['Price']:.0f}")
                        st.metric("RAM", f"{product['RAM']:.0f} GB")
                        st.metric("Storage", product["Storage Label"])
                    with col2:
                        st.metric("Battery", product["Battery Label"])
                        st.metric("Rating", f"{product['Average Rating']:.1f}/5")
                        st.metric("Refresh Rate", f"{product['Refresh Rate']:.0f} Hz")
                    with col3:
                        st.metric("Screen", f"{product['Screen Size']:.1f}\"")
                        st.metric("Score", f"{product['Final_Score']:.1f}")
                        if product_type == "Phone":
                            st.metric("Rear Camera", f"{product['Rear Camera']:.0f} MP")
                        else:
                            st.metric("Weight", f"{product['Weight Value']:.2f} kg")

                st.caption(f"{product['Extra 1']} | {product['Extra 2']}")
                st.info(product_summary(product, product_type, use_case, recommendation_style))
                st.write(top_reasons(product, product_type))

                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Base AI Score", f"{product['Base_Score']:.1f}")
                with score_col2:
                    st.metric("Style Fit", f"{product['Style_Score']:.1f}")
                with score_col3:
                    st.metric("Learned Boost", f"{product['Preference_Boost']:.0f}")

                st.write("Why this recommendation")
                for label, value in score_breakdown(product, product_type):
                    st.progress(float(np.clip(value, 0, 1)), text=f"{label}: {value * 100:.0f}%")

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("Add to Wishlist", key=f"wishlist_{product_type}_{row_index}"):
                        if item_key not in st.session_state.wishlist:
                            st.session_state.wishlist.append(item_key)
                            st.success(f"Added {product['Model']} to wishlist.")
                with action_col2:
                    if st.button("Good Recommendation", key=f"feedback_{product_type}_{row_index}"):
                        st.session_state.feedback.append(
                            {
                                "product": item_key,
                                "rating": 5,
                                "product_type": product_type,
                                "use_case": use_case,
                                "timestamp": datetime.now(),
                            }
                        )
                        st.success("Thanks for the feedback.")

    with tab2:
        st.subheader(f"{product_type} Dashboard")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Matches", len(filtered_df))
        with metric_cols[1]:
            st.metric("Avg Price", f"Rs {filtered_df['Price'].mean():.0f}")
        with metric_cols[2]:
            st.metric("Avg Rating", f"{filtered_df['Average Rating'].mean():.1f}/5")
        with metric_cols[3]:
            st.metric("Brands", filtered_df["Brand"].nunique())

        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            brand_counts = filtered_df["Brand"].value_counts().head(10).reset_index()
            brand_counts.columns = ["Brand", "Count"]
            fig_brands = px.bar(
                brand_counts,
                x="Count",
                y="Brand",
                orientation="h",
                title=f"Top {product_type} Brands in Current Results",
            )
            st.plotly_chart(fig_brands, width="stretch")

        with insight_col2:
            fig_price = px.histogram(
                filtered_df,
                x="Price",
                nbins=20,
                title="Price Distribution",
            )
            st.plotly_chart(fig_price, width="stretch")

        metrics_for_brand = dashboard_metric_columns(product_type)
        brand_summary = (
            filtered_df.groupby("Brand")[metrics_for_brand]
            .mean(numeric_only=True)
            .round(2)
            .reset_index()
        )
        brand_summary["Value Index"] = (
            brand_summary["Average Rating"] * 100000 / brand_summary["Price"].replace(0, np.nan)
        ).round(2)
        brand_summary = brand_summary.sort_values("Value Index", ascending=False)
        st.write("Best value brands in the current filter set")
        st.dataframe(brand_summary.head(12), width="stretch")

        if not top_products.empty:
            best_value = top_products.sort_values(["Price Score", "Average Rating"], ascending=False).iloc[0]
            best_performance = top_products.sort_values("Style_Score", ascending=False).iloc[0]
            st.success(
                f"Quick picks: best value is {best_value['Brand']} {best_value['Model']}; "
                f"best style match is {best_performance['Brand']} {best_performance['Model']}."
            )

    with tab3:
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            if product_type == "Phone":
                color_col = "Rear Camera"
                size_col = "RAM"
            elif product_type == "Laptop":
                color_col = "GPU Score"
                size_col = "RAM"
            else:
                color_col = "Camera MP"
                size_col = "Battery"
            fig_scatter = px.scatter(
                top_products,
                x="Price",
                y="Average Rating",
                size=size_col,
                color=color_col,
                hover_name="Model",
                title=f"{product_type}: Price vs Rating",
            )
            st.plotly_chart(fig_scatter, width="stretch")

        with col2:
            fig_bar = px.bar(
                top_products.head(10),
                x="Final_Score",
                y="Model",
                title=f"Top 10 {product_label} by Final Score",
                orientation="h",
            )
            st.plotly_chart(fig_bar, width="stretch")

        selected_product = st.selectbox("Compare specs (Radar Chart)", top_products["Model"])
        product_data = top_products[top_products["Model"] == selected_product].iloc[0]

        if product_type == "Phone":
            categories = ["RAM", "Storage", "Battery", "Rating", "Camera", "Price Value"]
            values = [
                product_data["RAM"] / df["RAM"].max() * 100,
                product_data["Storage"] / df["Storage"].max() * 100,
                product_data["Battery"] / df["Battery"].max() * 100,
                product_data["Average Rating"] / 5 * 100,
                product_data["Rear Camera"] / df["Rear Camera"].max() * 100,
                (1 - product_data["Price"] / df["Price"].max()) * 100,
            ]
        elif product_type == "Laptop":
            categories = ["RAM", "Storage", "Battery", "Rating", "GPU", "Portability"]
            values = [
                product_data["RAM"] / df["RAM"].max() * 100,
                product_data["Storage"] / df["Storage"].max() * 100,
                product_data["Battery"] / df["Battery"].max() * 100,
                product_data["Average Rating"] / 5 * 100,
                product_data["GPU Score"] / df["GPU Score"].max() * 100,
                (1 - product_data["Weight Value"] / df["Weight Value"].max()) * 100,
            ]
        else:
            categories = ["Camera", "Video", "Flight", "Range", "Speed", "Portability"]
            values = [
                product_data["Camera MP"] / df["Camera MP"].max() * 100,
                product_data["Video Quality Score"] / df["Video Quality Score"].max() * 100,
                product_data["Battery"] / df["Battery"].max() * 100,
                product_data["Range"] / df["Range"].max() * 100,
                product_data["Max Speed"] / df["Max Speed"].max() * 100,
                (1 - product_data["Weight Value"] / df["Weight Value"].max()) * 100,
            ]

        fig_radar = go.Figure(
            data=go.Scatterpolar(r=values, theta=categories, fill="toself", name=selected_product)
        )
        fig_radar.update_layout(title=f"Specs: {selected_product}")
        st.plotly_chart(fig_radar, width="stretch")

        if product_type == "Phone":
            heatmap_cols = ["RAM", "Storage", "Battery", "Average Rating", "Rear Camera"]
        elif product_type == "Laptop":
            heatmap_cols = ["RAM", "Storage", "Battery", "Average Rating", "GPU Score", "Processor Score"]
        else:
            heatmap_cols = [
                "Camera MP",
                "Video Quality Score",
                "Battery",
                "Range",
                "Max Speed",
                "Average Rating",
            ]
        heatmap_data = top_products[["Brand", "Model"] + heatmap_cols].head(10).copy()
        heatmap_data = heatmap_data.reset_index(drop=True)
        heatmap_data["Product"] = (
            heatmap_data["Brand"] + " " + heatmap_data["Model"] + " (#" + (heatmap_data.index + 1).astype(str) + ")"
        )
        heatmap_data = heatmap_data[["Product"] + heatmap_cols].set_index("Product")
        heatmap_data = heatmap_data.div(heatmap_data.max()).fillna(0) * 100

        fig_heatmap = px.imshow(
            heatmap_data.T,
            labels=dict(x=product_type, y="Specs", color="Normalized Score"),
            title=f"Feature Heatmap (Top {product_label})",
            aspect="auto",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_heatmap, width="stretch")

    with tab4:
        st.subheader(f"Compare {product_label}")
        selected_compare = st.multiselect(
            f"Select {product_type.lower()}s to compare",
            top_products["Model"],
            default=top_products["Model"].head(2).tolist(),
        )
        if selected_compare:
            if product_type == "Phone":
                compare_cols = [
                    "Brand",
                    "Model",
                    "Price",
                    "RAM",
                    "Storage",
                    "Battery",
                    "Screen Size",
                    "Refresh Rate",
                    "Average Rating",
                    "Rear Camera",
                    "Front Camera",
                    "5G",
                ]
            elif product_type == "Laptop":
                compare_cols = [
                    "Brand",
                    "Model",
                    "Price",
                    "RAM",
                    "Storage",
                    "Battery",
                    "Screen Size",
                    "Refresh Rate",
                    "Average Rating",
                    "Processor",
                    "GPU",
                    "Weight Value",
                    "Touchscreen",
                    "Operating System",
                ]
            else:
                compare_cols = [
                    "Brand",
                    "Model",
                    "Price",
                    "Camera MP",
                    "Video Resolution",
                    "Battery",
                    "Range",
                    "Max Speed",
                    "Weight Value",
                    "Obstacle Avoidance",
                    "GPS",
                    "Gimbal Stabilization",
                    "Average Rating",
                ]

            compare_df = df[df["Model"].isin(selected_compare)][compare_cols].copy()
            st.dataframe(compare_df, width="stretch")

            compare_scored = ranked_df[ranked_df["Model"].isin(selected_compare)].copy()
            if not compare_scored.empty:
                overall_winner = compare_scored.iloc[0]
                st.success(
                    f"Overall winner: {overall_winner['Brand']} {overall_winner['Model']} "
                    f"with a {overall_winner['Final_Score']:.1f} final score."
                )
                scorecard_cols = [
                    "Brand",
                    "Model",
                    "Final_Score",
                    "Base_Score",
                    "Style_Score",
                    "Preference_Boost",
                ]
                st.write("Scorecard")
                st.dataframe(compare_scored[scorecard_cols].round(2), width="stretch")

            winner_table = build_winner_table(df[df["Model"].isin(selected_compare)], product_type)
            if not winner_table.empty:
                st.write("Spec-by-spec winners")
                st.dataframe(winner_table, width="stretch")

            csv_buffer = StringIO()
            compare_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Comparison (CSV)",
                csv_buffer.getvalue(),
                file_name=f"{product_type.lower()}_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            )

    with tab5:
        st.subheader("Your Wishlist")
        current_keys = [key for key in st.session_state.wishlist if key.startswith(f"{product_type}:")]
        current_models = [key.split(":", 1)[1] for key in current_keys]

        if current_models:
            if product_type == "Phone":
                wishlist_cols = ["Brand", "Model", "Price", "RAM", "Storage", "Battery", "Average Rating"]
                wishlist_cols += ["Rear Camera", "5G"]
            elif product_type == "Laptop":
                wishlist_cols = ["Brand", "Model", "Price", "RAM", "Storage", "Battery", "Average Rating"]
                wishlist_cols += ["Processor", "GPU", "Weight Value"]
            else:
                wishlist_cols = [
                    "Brand",
                    "Model",
                    "Price",
                    "Camera MP",
                    "Video Resolution",
                    "Battery",
                    "Range",
                    "Weight Value",
                    "Average Rating",
                ]

            wishlist_source = df[df["Model"].isin(current_models)].copy()
            wishlist_ranked = score_products(wishlist_source, product_type, use_case)
            wishlist_ranked = apply_recommendation_style(
                wishlist_ranked,
                product_type,
                recommendation_style,
                df,
                st.session_state.feedback if learn_from_feedback else [],
            )
            wishlist_display_cols = wishlist_cols + ["Final_Score", "Style_Score"]
            wishlist_df = wishlist_ranked[wishlist_display_cols].copy()

            best_wishlist = wishlist_ranked.iloc[0]
            st.success(
                f"Best choice in your wishlist: {best_wishlist['Brand']} {best_wishlist['Model']} "
                f"with a {best_wishlist['Final_Score']:.1f} score."
            )
            st.dataframe(wishlist_df.round(2), width="stretch")

            if st.button("Clear Wishlist"):
                st.session_state.wishlist = [
                    key for key in st.session_state.wishlist if not key.startswith(f"{product_type}:")
                ]
                st.rerun()

            csv_buffer = StringIO()
            wishlist_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Wishlist (CSV)",
                csv_buffer.getvalue(),
                file_name=f"{product_type.lower()}_wishlist_{datetime.now().strftime('%Y%m%d')}.csv",
            )
        else:
            st.info(f"Your {product_type.lower()} wishlist is empty. Add items from the Top Picks.")

    with tab6:
        st.subheader("Feedback & Ratings")
        rating = st.slider("Rate this recommendation", 1, 5, 3)
        comment = st.text_area("Your feedback (optional)")

        if st.button("Submit Feedback"):
            st.session_state.feedback.append(
                {
                    "rating": rating,
                    "comment": comment,
                    "product_type": product_type,
                    "use_case": use_case,
                    "timestamp": datetime.now(),
                }
            )
            st.success("Thank you for your feedback.")

        if st.session_state.feedback:
            if st.button("Clear Feedback Learning"):
                st.session_state.feedback = []
                st.rerun()
            st.write("Recent Feedback:")
            for feedback in st.session_state.feedback[-5:]:
                st.write(
                    f"{feedback.get('product_type', product_type)} | "
                    f"{feedback.get('rating', 'N/A')}/5 - "
                    f"{feedback.get('comment', 'No comment')} "
                    f"({feedback.get('timestamp', 'N/A')})"
                )


st.sidebar.markdown("---")
st.sidebar.markdown("### Trending This Week")
trending = df.nlargest(3, "Average Rating")[["Model", "Brand", "Average Rating"]]
for idx, (_, product) in enumerate(trending.iterrows(), 1):
    st.sidebar.write(f"{idx}. {product['Brand']} {product['Model']} | {product['Average Rating']:.1f}/5")
