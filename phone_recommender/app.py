import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress openpyxl style warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel(r"C:\Users\kraje\Desktop\CP\phone_recommender\mobile_dataset.xlsx")

# Remove unwanted brands
brands_to_remove = ["hmd", "acer", "mtr", "ikall", "itel"]
df = df[~df["brand_name"].str.lower().isin(brands_to_remove)]

# Rename columns for clarity
df = df.rename(columns={
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
    "front_camera": "Front Camera"
})

# -----------------------------
# Data Cleaning
# -----------------------------
# Extract numeric values from camera specs
df["Rear Camera"] = df["Rear Camera"].astype(str).str.extract(r"(\d+)").astype(float)
df["Front Camera"] = df["Front Camera"].astype(str).str.extract(r"(\d+)").astype(float)

# Convert numeric columns
numeric_cols = ["RAM", "Storage", "Battery", "Average Rating", "Price", "Screen Size", "Rear Camera", "Front Camera"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fix processor speed and fill missing values
df["processor_speed_Hz"] = pd.to_numeric(df.get("processor_speed_Hz"), errors="coerce")
df["processor_speed_Hz"] = df["processor_speed_Hz"].fillna(df["processor_speed_Hz"].mean())

# Fix 1TB phones
df["Storage"] = df["Storage"].replace(1, 1024)

# Refresh rates
df["refresh_rates"] = pd.to_numeric(df.get("refresh_rates"), errors="coerce")
df["refresh_rates"] = df["refresh_rates"].fillna(120)

# Charger support
df["charger_support"] = df["charger_support"].replace("Fast", 1)

# Fill other missing values with defaults
df["RAM"] = df["RAM"].fillna(df["RAM"].median())
df["Battery"] = df["Battery"].fillna(df["Battery"].median())
df["Rear Camera"] = df["Rear Camera"].fillna(df["Rear Camera"].median())
df["Front Camera"] = df["Front Camera"].fillna(df["Front Camera"].median())
df["Average Rating"] = df["Average Rating"].fillna(df["Average Rating"].median())

# Price categories
def price_category(p):
    if p < 10000:
        return "Under ₹10,000"
    elif p < 20000:
        return "₹10k–₹20k"
    elif p < 40000:
        return "₹20k–₹40k"
    else:
        return "₹40k+"

df["Price Category"] = df["Price"].apply(price_category)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📱 Smart Phone Recommendation System")

st.sidebar.header("Filter Options")

# Brand filter
df["Brand"] = df["Brand"].astype(str)
brand = st.sidebar.selectbox("Select Brand", ["All"] + sorted(df["Brand"].dropna().unique()))

# Price filter
price = st.sidebar.selectbox("Price Category", ["All", "Under ₹10,000", "₹10k–₹20k", "₹20k–₹40k", "₹40k+"])

# RAM filter
ram = st.sidebar.slider("Minimum RAM (GB)", 2, 16, 4)

# 5G filter
fiveg = st.sidebar.checkbox("Only show 5G phones")

# Screen size filter
screen_min, screen_max = st.sidebar.slider("Screen Size Range (inches)", 4.5, 7.5, (5.5, 6.5))

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

if brand != "All":
    filtered_df = filtered_df[filtered_df["Brand"] == brand]

if price != "All":
    filtered_df = filtered_df[filtered_df["Price Category"] == price]

filtered_df = filtered_df[filtered_df["RAM"] >= ram]
filtered_df = filtered_df[(filtered_df["Screen Size"] >= screen_min) & (filtered_df["Screen Size"] <= screen_max)]

if fiveg:
    filtered_df = filtered_df[filtered_df["5G"] == 1]

# -----------------------------
# Preference Weights
# -----------------------------
st.sidebar.subheader("Preference Weights")
w_rating = st.sidebar.slider("Weight: Rating", 0.0, 1.0, 0.4)
w_ram = st.sidebar.slider("Weight: RAM", 0.0, 1.0, 0.2)
w_storage = st.sidebar.slider("Weight: Storage", 0.0, 1.0, 0.2)
w_battery = st.sidebar.slider("Weight: Battery", 0.0, 1.0, 0.2)
w_camera = st.sidebar.slider("Weight: Rear Camera", 0.0, 1.0, 0.2)

# -----------------------------
# Recommendation System
# -----------------------------
if filtered_df.empty:
    st.warning("⚠️ No phones match your filters.")
else:
    filtered_df["Score"] = (
        filtered_df["Average Rating"] * w_rating +
        filtered_df["RAM"] * w_ram +
        filtered_df["Storage"] * w_storage +
        filtered_df["Battery"] * w_battery +
        filtered_df["Rear Camera"] * w_camera
    )

    # Hybrid score
    filtered_df["Hybrid Score"] = (
        filtered_df["Score"] * 0.7 + filtered_df["Average Rating"] * 0.3
    )

    # AI-Powered Recommendations
    features = filtered_df[["RAM","Storage","Battery","Average Rating","Price"]]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    similarity_matrix = cosine_similarity(features_scaled)

    def recommend_similar(model_name, top_n=5):
        idx = filtered_df[filtered_df["Model"] == model_name].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[1:top_n+1]]
        return filtered_df.iloc[top_indices][["Brand","Model","Price","RAM","Battery","Average Rating"]]

    st.subheader("🤖 AI-Powered Recommendations")
    phone_choice = st.selectbox("Pick a phone to find similar ones", filtered_df["Model"].unique())
    st.write(recommend_similar(phone_choice, top_n=5))

    top_phones = filtered_df.sort_values("Hybrid Score", ascending=False).head(10)

    st.subheader("🏆 Top Recommended Phones")
    st.dataframe(top_phones[["Brand","Model","Price","RAM","Storage","Battery","Rear Camera","Average Rating","Hybrid Score"]])

    # Visualization
    st.subheader("📊 Top Phones by Score")
    fig, ax = plt.subplots()
    top_phones.plot(x="Model", y="Hybrid Score", kind="barh", ax=ax, color="skyblue", legend=False)
    ax.set_xlabel("Hybrid Score")
    st.pyplot(fig)

    # Phone Details
    phone = st.selectbox("Select phone for details", top_phones["Model"])
    phone_data = df[df["Model"] == phone]
    st.subheader("📊 Phone Details")
    st.dataframe(phone_data)

    # Phone Comparison
    st.subheader("🔍 Compare Phones")
    compare = st.multiselect("Select phones to compare", top_phones["Model"], default=top_phones["Model"].head(2))
    if len(compare) >= 2:
        st.dataframe(df[df["Model"].isin(compare)][["Brand","Model","Price","RAM","Storage","Battery","Rear Camera","Front Camera","Average Rating"]])
