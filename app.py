import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from rapidfuzz import fuzz
import math

# -----------------------------
# Inject Custom CSS for Better Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* Custom Sidebar width */
    [data-testid="stSidebar"] {
        width: 350px;
        min-width: 350px;
        max-width: 350px;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 350px;
    }

    /* Header styling */
    .main-header {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #333333;
        margin-bottom: 20px;
    }
    /* Metric card styling */
    .metric-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2em;
        color: #4B4B4B;
    }
    .metric-card p {
        margin: 0;
        color: #555555;
    }
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    table, th, td {
         border: 1px solid #ddd;
         padding: 8px;
    }
    tr:nth-child(even){background-color: #f9f9f9;}
    tr:hover {background-color: #f1f1f1;}
    th {
         background-color: #4B4B4B;
         color: white;
         text-align: left;
         padding: 12px;
    }
    /* Link styling */
    a {
         color: #1f77b4;
         text-decoration: none;
    }
    a:hover {
         text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helper Function: Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "0bc5ea69-fcc7-4998-ab6c-70c3a0df778b.csv",
        parse_dates=["START_DATE", "REPORT_DATE", "OFFENSE_DATE"],
        dtype={"CDTS": str},
        low_memory=False
    )
    df["START_DATE"] = pd.to_datetime(df["START_DATE"], errors="coerce")
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    df["OFFENSE_DATE"] = pd.to_datetime(df["OFFENSE_DATE"], errors="coerce")
    return df

# -----------------------------
# Function: Create a 3D Bar (Mesh3d) for a Cuboid at (x,y)
# -----------------------------
def create_3d_bar(x, y, height, width=0.005, depth=0.005, color='blue'):
    # Define the base and top coordinates
    x0, x1 = x - width/2, x + width/2
    y0, y1 = y - depth/2, y + depth/2
    z0, z1 = 0, height
    # 8 vertices of the cuboid
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)   # top
    ]
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    # Define the 12 triangles (each face split into 2 triangles)
    faces = [
        (0,1,2), (0,2,3),       # bottom
        (4,5,6), (4,6,7),       # top
        (0,1,5), (0,5,4),       # front
        (1,2,6), (1,6,5),       # right
        (2,3,7), (2,7,6),       # back
        (3,0,4), (3,4,7)        # left
    ]
    i_indices = [face[0] for face in faces]
    j_indices = [face[1] for face in faces]
    k_indices = [face[2] for face in faces]

    bar = go.Mesh3d(
        x = x_coords, y = y_coords, z = z_coords,
        i = i_indices, j = j_indices, k = k_indices,
        opacity = 0.8,
        color = color,
        flatshading=True,
        showscale=False
    )
    return bar

# -----------------------------
# Function: Display Combined Dashboard
# -----------------------------
def display_combined_dashboard():
    df = load_data()
    
    # -----------------------------
    # Sidebar: Filter Options (single date range on OFFENSE_DATE)
    # -----------------------------
    with st.sidebar.expander("Filter Options", expanded=True):
        st.markdown("### Filter Options")
        if "OFFENSE_DATE" in df.columns:
            st.markdown("**Offense Date**")
            min_date = df["OFFENSE_DATE"].min().date()
            max_date = df["OFFENSE_DATE"].max().date()
            date_range = st.date_input("Select Date Range", [min_date, max_date], key="offense_date")
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range
                df = df[df["OFFENSE_DATE"].dt.date.between(start_date, end_date)]
        
        col1, col2 = st.columns(2)
        with col1:
            if "CALL_TYPE" in df.columns:
                st.markdown("**Call Types**")
                call_types = sorted(df["CALL_TYPE"].dropna().unique())
                selected_call_types = st.multiselect("Select", call_types, default=call_types, key="call_type")
                df = df[df["CALL_TYPE"].isin(selected_call_types)]
        with col2:
            if "FINAL_DISPO" in df.columns:
                st.markdown("**Final Dispositions**")
                dispo_options = sorted(df["FINAL_DISPO"].dropna().unique())
                selected_dispo = st.multiselect("Select", dispo_options, default=dispo_options, key="final_disp")
                df = df[df["FINAL_DISPO"].isin(selected_dispo)]
        
        if "PRIORITY" in df.columns:
            st.markdown("**Priority Range**")
            min_priority = int(df["PRIORITY"].min())
            max_priority = int(df["PRIORITY"].max())
            priority_range = st.slider("Select Range", min_priority, max_priority, (min_priority, max_priority), key="priority_range")
            df = df[(df["PRIORITY"] >= priority_range[0]) & (df["PRIORITY"] <= priority_range[1])]
        
        # Enhanced Smart Search: auto-detect keywords to adjust filters
        search_text = st.text_input("Smart Search", value="", placeholder="Enter search terms (e.g., 'armed robbery in august')", key="search_text")
        if search_text:
            # Parse for month keywords
            month_mapping = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }
            for month_name, month_num in month_mapping.items():
                if month_name in search_text.lower():
                    df = df[df["OFFENSE_DATE"].dt.month == month_num]
                    break
            # Filter based on call types if keywords match
            if "CALL_TYPE" in df.columns:
                call_types = sorted(df["CALL_TYPE"].dropna().unique())
                matching_calls = [ct for ct in call_types if ct.lower() in search_text.lower()]
                if matching_calls:
                    df = df[df["CALL_TYPE"].isin(matching_calls)]
            # Filter based on final dispositions if keywords match
            if "FINAL_DISPO" in df.columns:
                dispo_options = sorted(df["FINAL_DISPO"].dropna().unique())
                matching_disp = [d for d in dispo_options if d.lower() in search_text.lower()]
                if matching_disp:
                    df = df[df["FINAL_DISPO"].isin(matching_disp)]
            # Apply fuzzy search filter across all text columns
            text_columns = list(df.select_dtypes(include=["object"]).columns)
            threshold = 60
            def row_fuzzy_score(row):
                max_score = 0
                for col in text_columns:
                    value = str(row[col]).lower() if pd.notnull(row[col]) else ""
                    score = fuzz.token_set_ratio(search_text.lower(), value)
                    max_score = max(max_score, score)
                return max_score
            mask = df.apply(lambda r: row_fuzzy_score(r) >= threshold, axis=1)
            df = df[mask]
        
        if st.button("Reset Filters"):
            st.experimental_rerun()
    
    # Create a copy of the filtered data.
    filtered_df = df.copy()
    
    # -----------------------------
    # Specific Call Lookup: Dynamic Search & Select
    # -----------------------------
    with st.sidebar.expander("Specific Call Lookup"):
        # As the user types lookup keywords (e.g., "theft"), update the lookup results dynamically.
        lookup_query = st.text_input("Enter lookup keywords", key="lookup_keyword")
        if lookup_query:
            full_data = load_data()  # Using full dataset for lookup
            text_cols = list(full_data.select_dtypes(include=["object"]).columns)
            # Use a lower threshold for short queries so suggestions appear
            threshold_lookup = 30 if len(lookup_query) < 3 else 60
            def row_fuzzy_score_lookup(row):
                max_score = 0
                for col in text_cols:
                    value = str(row[col]).lower() if pd.notnull(row[col]) else ""
                    score = fuzz.token_set_ratio(lookup_query.lower(), value)
                    max_score = max(max_score, score)
                return max_score
            # Compute fuzzy score for each row and add as a temporary column
            full_data["score"] = full_data.apply(lambda r: row_fuzzy_score_lookup(r), axis=1)
            lookup_results = full_data[full_data["score"] >= threshold_lookup]
            lookup_results = lookup_results.sort_values("score", ascending=False).head(20)
            if lookup_results.empty:
                st.write("No matching results found.")
            else:
                # Format each result with key information for display in the selectbox
                def format_row(row):
                    call_type = row["CALL_TYPE"] if ("CALL_TYPE" in row and pd.notnull(row["CALL_TYPE"])) else "Unknown Call"
                    offense_date = row["OFFENSE_DATE"].date() if ("OFFENSE_DATE" in row and pd.notnull(row["OFFENSE_DATE"])) else "Unknown Date"
                    return f"ID {row['_id']} – {call_type} on {offense_date}"
                options = lookup_results.apply(format_row, axis=1).tolist()
                selected_option = st.selectbox("Select a specific call", options=options, key="lookup_select")
                if selected_option:
                    # Parse the selected option to extract the call ID and display its details
                    call_id_str = selected_option.split("–")[0].replace("ID", "").strip()
                    try:
                        call_id_val = int(call_id_str)
                        specific_call = full_data[full_data["_id"] == call_id_val]
                        st.markdown("### Specific Call Details")
                        st.dataframe(specific_call.T.astype(str), use_container_width=True)
                    except Exception as e:
                        st.error("Error retrieving call details.")
    
    # -----------------------------
    # Optionally Show Call Details if a call_id is passed via query parameter
    # -----------------------------
    params = st.query_params
    call_id = params.get("call_id", [None])[0]
    if call_id:
        try:
            call_id_val = int(call_id)
        except ValueError:
            st.error("Invalid call id. Must be a number.")
        else:
            selected_call = filtered_df[filtered_df["_id"] == call_id_val]
            if selected_call.empty:
                st.error("Call details not found.")
            else:
                st.markdown("### Call Details")
                st.dataframe(selected_call.T.astype(str), use_container_width=True)
                st.markdown("---")
    
    # -----------------------------
    # KPI Metrics
    # -----------------------------
    st.markdown('<div class="main-header">Police Calls Dashboard</div>', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown(f'<div class="metric-card"><h2>{filtered_df.shape[0]}</h2><p>Total Calls</p></div>', unsafe_allow_html=True)
    with colB:
        unique_calls = filtered_df["CALL_TYPE"].nunique() if "CALL_TYPE" in filtered_df.columns else 0
        st.markdown(f'<div class="metric-card"><h2>{unique_calls}</h2><p>Unique Call Types</p></div>', unsafe_allow_html=True)
    with colC:
        unique_disp = filtered_df["FINAL_DISPO"].nunique() if "FINAL_DISPO" in filtered_df.columns else 0
        st.markdown(f'<div class="metric-card"><h2>{unique_disp}</h2><p>Unique Dispositions</p></div>', unsafe_allow_html=True)
    with colD:
        avg_priority = round(filtered_df["PRIORITY"].mean(), 2) if "PRIORITY" in filtered_df.columns else "N/A"
        st.markdown(f'<div class="metric-card"><h2>{avg_priority}</h2><p>Average Priority</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # -----------------------------
    # Visualizations Section: Daily Calls Trend & Call Type Distribution
    # -----------------------------
    st.subheader("Key Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        if "OFFENSE_DATE" in filtered_df.columns:
            filtered_df["OFFENSE_DAY"] = filtered_df["OFFENSE_DATE"].dt.date
            daily_counts = filtered_df.groupby("OFFENSE_DAY").size().reset_index(name="Count")
            fig_line = px.line(daily_counts, x="OFFENSE_DAY", y="Count", title="Daily Calls Trend", template="plotly_white")
            st.plotly_chart(fig_line, use_container_width=True)
    with viz_col2:
        if "CALL_TYPE" in filtered_df.columns:
            fig_hist = px.histogram(filtered_df, x="CALL_TYPE", title="Call Type Distribution", template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # -----------------------------
    # Priority Analysis
    # -----------------------------
    st.subheader("Priority Analysis")
    if "PRIORITY" in filtered_df.columns and "OFFENSE_DATE" in filtered_df.columns:
        df_temp = filtered_df.copy()
        df_temp["OFFENSE_MONTH"] = df_temp["OFFENSE_DATE"].dt.to_period("M").astype(str)
        priority_grouped = df_temp.groupby("OFFENSE_MONTH")["PRIORITY"].agg(["mean"]).reset_index()
        fig_priority = px.line(priority_grouped, x="OFFENSE_MONTH", y="mean", markers=True,
                               title="Average Priority Over Time", template="plotly_white")
        fig_priority.update_layout(xaxis_title="Month", yaxis_title="Average Priority", xaxis_tickangle=-45)
        st.plotly_chart(fig_priority, use_container_width=True)
    
    st.markdown("---")
    
    # -----------------------------
    # 3D Geographic Insights using Pydeck (Optimized)
    # -----------------------------
    st.subheader("3D Geographic Insights")
    base_lat = 37.3382082
    base_lon = -121.8863286
    optimized_sample = filtered_df.copy()
    # Ensure geocoded coordinates exist; if not, simulate them (centered on San Jose)
    if "lat" not in optimized_sample.columns or "lon" not in optimized_sample.columns:
        optimized_sample["lat"] = base_lat + np.random.normal(0, 0.01, size=len(optimized_sample))
        optimized_sample["lon"] = base_lon + np.random.normal(0, 0.01, size=len(optimized_sample))
    
    # Sample up to 50000 points for more detailed insights.
    if len(optimized_sample):
        optimized_sample = optimized_sample.sample(n=len(optimized_sample), random_state=42)
    
    # Bin the lat/lon; round to 3 decimals for increased granularity (more columns)
    optimized_sample["lat_bin"] = optimized_sample["lat"].round(3)
    optimized_sample["lon_bin"] = optimized_sample["lon"].round(3)
     
    agg_geo = optimized_sample.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="Count")
    agg_geo.rename(columns={"lat_bin": "lat", "lon_bin": "lon"}, inplace=True)

    # Compute dynamic exponent based on the number of records
    num_records = len(optimized_sample)
    max_records = 120000
    ratio = min(num_records / max_records, 1)
    dynamic_exponent = 2.0 - 0.55 * ratio  # Exponent ranges from 1.0 to 1.45 based on dataset size
    agg_geo["elevation"] = agg_geo["Count"] ** dynamic_exponent
    
    import pydeck as pdk
    view_state = pdk.ViewState(latitude=base_lat, longitude=base_lon, zoom=14, pitch=45)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=agg_geo,
        get_position="[lon, lat]",
        elevation_scale=1,
        get_elevation="elevation",
        radius=100,
        get_fill_color="[200, 20, 20, 120]",
        pickable=True,
        auto_highlight=True
    )
    deck = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state,
        tooltip={"text": "Calls: {Count}"}
    )
    st.pydeck_chart(deck)

# -----------------------------
# Run the Combined Dashboard
# -----------------------------
st.set_page_config(page_title="Police Calls Dashboard", layout="wide", initial_sidebar_state="expanded")
display_combined_dashboard()