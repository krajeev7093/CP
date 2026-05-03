import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
from transformers import pipeline
import re
from datetime import datetime
import csv
from io import StringIO

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Page config
st.set_page_config(page_title="📱 Smart Phone Recommender", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "wishlist" not in st.session_state:
    st.session_state.wishlist = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Custom CSS for dark mode
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .main {color: white; background-color: #1e1e1e;}
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_resource
def load_data():
    df = pd.read_excel(r"C:\Users\kraje\Desktop\CP\phone_recommender\mobile_dataset.xlsx")
    brands_to_remove = ["hmd", "acer", "mtr", "ikall", "itel"]
    df = df[~df["brand_name"].str.lower().isin(brands_to_remove)]
    
    df = df.rename(columns={
        "brand_name": "Brand", "model": "Model", "ram_support": "RAM",
        "rom_GB": "Storage", "battery_capacity": "Battery",
        "rating": "Average Rating", "price": "Price", "has_5g": "5G",
        "display_size": "Screen Size", "rear_camera": "Rear Camera",
        "front_camera": "Front Camera"
    })
    
    # Data cleaning
    df["Rear Camera"] = df["Rear Camera"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Front Camera"] = df["Front Camera"].astype(str).str.extract(r"(\d+)").astype(float)
    
    numeric_cols = ["RAM", "Storage", "Battery", "Average Rating", "Price", "Screen Size", "Rear Camera", "Front Camera"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["processor_speed_Hz"] = pd.to_numeric(df.get("processor_speed_Hz"), errors="coerce")
    df["processor_speed_Hz"] = df["processor_speed_Hz"].fillna(df["processor_speed_Hz"].mean())
    df["Storage"] = df["Storage"].replace(1, 1024)
    df["refresh_rates"] = pd.to_numeric(df.get("refresh_rates"), errors="coerce")
    df["refresh_rates"] = df["refresh_rates"].fillna(120)
    df["charger_support"] = df["charger_support"].replace("Fast", 1)
    
    df["RAM"] = df["RAM"].fillna(df["RAM"].median())
    df["Battery"] = df["Battery"].fillna(df["Battery"].median())
    df["Rear Camera"] = df["Rear Camera"].fillna(df["Rear Camera"].median())
    df["Front Camera"] = df["Front Camera"].fillna(df["Front Camera"].median())
    df["Average Rating"] = df["Average Rating"].fillna(df["Average Rating"].median())
    
    return df

# Load NLP model with caching
@st.cache_resource
def load_nlp_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

df = load_data()
df["Brand"] = df["Brand"].astype(str)

# Header with dark mode toggle
col1, col2 = st.columns([10, 1])
with col1:
    st.title("📱 Smart Phone Recommendation System")
with col2:
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", st.session_state.dark_mode)

# Define price categories
def price_category(p):
    if p < 10000: return "Under ₹10,000"
    elif p < 20000: return "₹10k–₹20k"
    elif p < 40000: return "₹20k–₹40k"
    else: return "₹40k+"

df["Price Category"] = df["Price"].apply(price_category)

# Sidebar
st.sidebar.header("🔧 Filters & Preferences")

# Use case selection
use_case = st.sidebar.selectbox("📌 What's your priority?", 
    ["Balanced", "Gaming", "Photography", "Productivity", "Budget"])

# Advanced Filters
st.sidebar.subheader("📊 Advanced Filters")
col1, col2 = st.sidebar.columns(2)
with col1:
    price_min = st.number_input("Min Price (₹)", 5000, 150000, 5000, step=1000)
with col2:
    price_max = st.number_input("Max Price (₹)", 5000, 150000, 50000, step=1000)

brand = st.sidebar.selectbox("Brand", ["All"] + sorted(df["Brand"].dropna().unique()))
ram = st.sidebar.slider("Minimum RAM (GB)", 2, 16, 4)
storage = st.sidebar.slider("Minimum Storage (GB)", 32, 1024, 128)
battery = st.sidebar.slider("Minimum Battery (mAh)", 2500, 6000, 4000)
refresh_rate = st.sidebar.slider("Minimum Refresh Rate (Hz)", 60, 165, 90)
fiveg = st.sidebar.checkbox("5G Support")
screen_min, screen_max = st.sidebar.slider("Screen Size (inches)", 4.5, 7.5, (5.5, 6.5))

# AI Query Assistant
st.sidebar.subheader("🤖 AI Query Assistant")
user_query = st.sidebar.text_input("Describe your ideal phone (e.g., 'budget gaming phone under ₹30k')")

# Extract price from query
query_price_min, query_price_max = price_min, price_max
if user_query:
    price_pattern = re.findall(r'₹?(\d+k?)', user_query.lower())
    if price_pattern:
        try:
            extracted_price = int(price_pattern[0].replace('k', '000')) if 'k' in price_pattern[0] else int(price_pattern[0])
            query_price_max = min(query_price_max, extracted_price)
        except:
            pass

# Apply NLP
classifier = load_nlp_model()
labels = ["budget", "gaming", "photography", "battery", "5G", "large screen", "fast charging", "high RAM"]
intent_weights = {
    "Balanced": {"rating": 0.25, "ram": 0.15, "storage": 0.15, "battery": 0.15, "camera": 0.20, "price": 0.10},
    "Gaming": {"ram": 0.30, "processor": 0.25, "refresh_rate": 0.25, "battery": 0.15, "rating": 0.05},
    "Photography": {"camera": 0.40, "ram": 0.15, "rating": 0.20, "storage": 0.15, "battery": 0.10},
    "Productivity": {"ram": 0.25, "storage": 0.25, "battery": 0.20, "rating": 0.20, "screen": 0.10},
    "Budget": {"price": 0.40, "battery": 0.20, "ram": 0.15, "storage": 0.15, "rating": 0.10}
}

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[(filtered_df["Price"] >= price_min) & (filtered_df["Price"] <= price_max)]

if user_query:
    result = classifier(user_query, labels)
    intents = [label for label, score in zip(result['labels'], result['scores']) if score > 0.4]
    
    if "budget" in intents:
        filtered_df = filtered_df[filtered_df["Price"] <= 30000]
    if "gaming" in intents:
        ram = max(ram, 8)
        filtered_df = filtered_df[filtered_df["refresh_rates"] >= 120]
    if "photography" in intents:
        filtered_df = filtered_df[filtered_df["Rear Camera"] >= 50]
    if "battery" in intents:
        battery = max(battery, 5000)
    if "5G" in intents:
        fiveg = True
    if "large screen" in intents:
        screen_min = max(screen_min, 6.0)
    if "fast charging" in intents:
        filtered_df = filtered_df[filtered_df["charger_support"] > 0]
    if "high RAM" in intents:
        ram = max(ram, 12)
    
    st.sidebar.write(f"✓ Detected: {', '.join(intents)}")

if brand != "All":
    filtered_df = filtered_df[filtered_df["Brand"] == brand]

filtered_df = filtered_df[filtered_df["RAM"] >= ram]
filtered_df = filtered_df[filtered_df["Storage"] >= storage]
filtered_df = filtered_df[filtered_df["Battery"] >= battery]
filtered_df = filtered_df[filtered_df["refresh_rates"] >= refresh_rate]
filtered_df = filtered_df[(filtered_df["Screen Size"] >= screen_min) & (filtered_df["Screen Size"] <= screen_max)]

if fiveg:
    filtered_df = filtered_df[filtered_df["5G"] == 1]

filtered_df = filtered_df.reset_index(drop=True)

# Scoring based on use case
weights = intent_weights[use_case]

if not filtered_df.empty:
    # Normalize features for scoring
    scaler = MinMaxScaler()
    
    if use_case == "Gaming":
        filtered_df["Use_Case_Score"] = (
            (scaler.fit_transform(filtered_df[["RAM"]]) * weights["ram"] +
             scaler.fit_transform(filtered_df[["processor_speed_Hz"]]) * weights["processor"] +
             scaler.fit_transform(filtered_df[["refresh_rates"]]) * weights["refresh_rate"] +
             scaler.fit_transform(filtered_df[["Battery"]]) * weights["battery"] +
             scaler.fit_transform(filtered_df[["Average Rating"]]) * weights["rating"]).sum(axis=1))
    elif use_case == "Photography":
        filtered_df["Use_Case_Score"] = (
            (scaler.fit_transform(filtered_df[["Rear Camera"]]) * weights["camera"] +
             scaler.fit_transform(filtered_df[["RAM"]]) * weights["ram"] +
             scaler.fit_transform(filtered_df[["Average Rating"]]) * weights["rating"] +
             scaler.fit_transform(filtered_df[["Storage"]]) * weights["storage"] +
             scaler.fit_transform(filtered_df[["Battery"]]) * weights["battery"]).sum(axis=1))
    elif use_case == "Productivity":
        filtered_df["Use_Case_Score"] = (
            (scaler.fit_transform(filtered_df[["RAM"]]) * weights["ram"] +
             scaler.fit_transform(filtered_df[["Storage"]]) * weights["storage"] +
             scaler.fit_transform(filtered_df[["Battery"]]) * weights["battery"] +
             scaler.fit_transform(filtered_df[["Average Rating"]]) * weights["rating"] +
             scaler.fit_transform(filtered_df[["Screen Size"]]) * weights["screen"]).sum(axis=1))
    elif use_case == "Budget":
        filtered_df["Use_Case_Score"] = (
            (scaler.fit_transform(filtered_df[["Price"]].transform(lambda x: 1/x)) * weights["price"] +
             scaler.fit_transform(filtered_df[["Battery"]]) * weights["battery"] +
             scaler.fit_transform(filtered_df[["RAM"]]) * weights["ram"] +
             scaler.fit_transform(filtered_df[["Storage"]]) * weights["storage"] +
             scaler.fit_transform(filtered_df[["Average Rating"]]) * weights["rating"]).sum(axis=1))
    else:  # Balanced
        filtered_df["Use_Case_Score"] = (
            (scaler.fit_transform(filtered_df[["Average Rating"]]) * weights["rating"] +
             scaler.fit_transform(filtered_df[["RAM"]]) * weights["ram"] +
             scaler.fit_transform(filtered_df[["Storage"]]) * weights["storage"] +
             scaler.fit_transform(filtered_df[["Battery"]]) * weights["battery"] +
             scaler.fit_transform(filtered_df[["Rear Camera"]]) * weights["camera"] +
             scaler.fit_transform(filtered_df[["Price"]].transform(lambda x: 1/x)) * weights["price"]).sum(axis=1))
    
    # Content-based filtering with KNN
    features = filtered_df[["RAM","Storage","Battery","Average Rating","Price","refresh_rates"]]
    features_scaled = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features_scaled)
    
    filtered_df["AI_Score"] = filtered_df["Use_Case_Score"] * 0.6 + filtered_df["Average Rating"] * 0.4
    filtered_df["Final_Score"] = filtered_df["AI_Score"] * 0.7 + similarity_matrix.mean(axis=1) * 30 * 0.3
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Top Picks", "📊 Visualizations", "🔍 Compare", "⭐ Wishlist", "💬 Feedback"])
    
    with tab1:
        st.subheader(f"Top Phones for {use_case}")
        top_phones = filtered_df.sort_values("Final_Score", ascending=False).head(15)
        
        # Display with score breakdown
        for idx, (i, phone) in enumerate(top_phones.iterrows(), 1):
            with st.expander(f"#{idx} {phone['Brand']} {phone['Model']} - ₹{phone['Price']:.0f} ⭐{phone['Average Rating']:.1f}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"₹{phone['Price']:.0f}")
                    st.metric("RAM", f"{phone['RAM']:.0f}GB")
                    st.metric("Storage", f"{phone['Storage']:.0f}GB")
                with col2:
                    st.metric("Battery", f"{phone['Battery']:.0f}mAh")
                    st.metric("Rating", f"{phone['Average Rating']:.1f}/5")
                    st.metric("5G", "✓" if phone['5G'] == 1 else "✗")
                with col3:
                    st.metric("Rear Camera", f"{phone['Rear Camera']:.0f}MP")
                    st.metric("Front Camera", f"{phone['Front Camera']:.0f}MP")
                    st.metric("Screen", f"{phone['Screen Size']:.1f}\"")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"❤️ Add to Wishlist", key=f"wishlist_{i}"):
                        if phone['Model'] not in st.session_state.wishlist:
                            st.session_state.wishlist.append(phone['Model'])
                            st.success(f"Added {phone['Model']} to wishlist!")
                with col_b:
                    if st.button(f"👍 Good Recommendation", key=f"feedback_{i}"):
                        st.session_state.feedback.append({"phone": phone['Model'], "rating": 5, "timestamp": datetime.now()})
                        st.success("Thanks for the feedback!")
    
    with tab2:
        st.subheader("📊 Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter: Price vs Rating
            fig_scatter = px.scatter(top_phones, x="Price", y="Average Rating", 
                                     size="RAM", color="Rear Camera", hover_name="Model",
                                     title="Price vs Rating (Size=RAM, Color=Camera)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Bar chart: Top phones
            fig_bar = px.bar(top_phones.head(10), x="Final_Score", y="Model", 
                            title="Top 10 Phones by Final Score", orientation="h")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Radar chart for selected phone
        selected_phone = st.selectbox("Compare specs (Radar Chart)", top_phones["Model"])
        phone_data = top_phones[top_phones["Model"] == selected_phone].iloc[0]
        
        categories = ["RAM", "Storage", "Battery", "Rating", "Camera", "Price Value"]
        values = [
            phone_data["RAM"] / df["RAM"].max() * 100,
            phone_data["Storage"] / df["Storage"].max() * 100,
            phone_data["Battery"] / df["Battery"].max() * 100,
            phone_data["Average Rating"] / 5 * 100,
            phone_data["Rear Camera"] / df["Rear Camera"].max() * 100,
            (1 - phone_data["Price"] / df["Price"].max()) * 100
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself',
            name=selected_phone
        ))
        fig_radar.update_layout(title=f"Specs: {selected_phone}")
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Heatmap: Top features (using unique identifiers to avoid duplicates)
        heatmap_data = top_phones[["Brand", "Model", "RAM", "Storage", "Battery", "Average Rating", "Rear Camera"]].head(10).copy()
        heatmap_data = heatmap_data.reset_index(drop=True)
        heatmap_data["Phone"] = heatmap_data["Brand"] + " " + heatmap_data["Model"] + " (#" + (heatmap_data.index + 1).astype(str) + ")"
        heatmap_data = heatmap_data[["Phone", "RAM", "Storage", "Battery", "Average Rating", "Rear Camera"]]
        heatmap_data = heatmap_data.set_index("Phone")
        heatmap_data = heatmap_data.div(heatmap_data.max()) * 100
        
        fig_heatmap = px.imshow(heatmap_data.T, labels=dict(x="Phone", y="Specs", color="Normalized Score"),
                               title="Feature Heatmap (Top Phones)", aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Compare Phones")
        selected_compare = st.multiselect("Select phones to compare", top_phones["Model"], 
                                         default=top_phones["Model"].head(2).tolist())
        if selected_compare:
            compare_df = df[df["Model"].isin(selected_compare)][
                ["Brand", "Model", "Price", "RAM", "Storage", "Battery", "Rear Camera", "Front Camera", "Average Rating", "5G"]
            ].copy()
            st.dataframe(compare_df, use_container_width=True)
            
            # Export comparison
            csv_buffer = StringIO()
            compare_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Comparison (CSV)", csv_buffer.getvalue(), 
                             file_name=f"phone_comparison_{datetime.now().strftime('%Y%m%d')}.csv")
    
    with tab4:
        st.subheader("⭐ Your Wishlist")
        if st.session_state.wishlist:
            wishlist_df = df[df["Model"].isin(st.session_state.wishlist)][
                ["Brand", "Model", "Price", "RAM", "Storage", "Battery", "Rear Camera", "Average Rating"]
            ]
            st.dataframe(wishlist_df, use_container_width=True)
            
            if st.button("🗑️ Clear Wishlist"):
                st.session_state.wishlist = []
                st.rerun()
            
            # Export wishlist
            csv_buffer = StringIO()
            wishlist_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Wishlist (CSV)", csv_buffer.getvalue(),
                             file_name=f"wishlist_{datetime.now().strftime('%Y%m%d')}.csv")
        else:
            st.info("Your wishlist is empty. Add phones from the Top Picks!")
    
    with tab5:
        st.subheader("💬 Feedback & Ratings")
        rating = st.slider("Rate this recommendation", 1, 5, 3)
        comment = st.text_area("Your feedback (optional)")
        
        if st.button("📤 Submit Feedback"):
            st.session_state.feedback.append({
                "rating": rating,
                "comment": comment,
                "use_case": use_case,
                "timestamp": datetime.now()
            })
            st.success("Thank you for your feedback!")
        
        if st.session_state.feedback:
            st.write("Recent Feedback:")
            for fb in st.session_state.feedback[-5:]:
                st.write(f"⭐ {fb.get('rating', 'N/A')}/5 - {fb.get('comment', 'No comment')} ({fb.get('timestamp', 'N/A')})")
else:
    st.error("❌ No phones match your filters. Try adjusting your preferences!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Trending This Week")
trending = df.nlargest(3, "Average Rating")[["Model", "Brand", "Average Rating"]]
for idx, (i, phone) in enumerate(trending.iterrows(), 1):
    st.sidebar.write(f"{idx}. {phone['Brand']} {phone['Model']} ⭐{phone['Average Rating']:.1f}")
