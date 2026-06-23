import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Train Journey Prediction System",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")

    # ========================= Task 2.1 - Handle Missing Values =========================

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    # ========================= Task 2.1 - Remove Duplicates =========================

    df.drop_duplicates(inplace=True)

    # ========================= Task 2.2 - Standardize Time Format =========================

    df["Arrival_time"] = pd.to_datetime(
        df["Arrival_time"],
        format="%H:%M:%S",
        errors="coerce"
    )

    df["Departure_Time"] = pd.to_datetime(
        df["Departure_Time"],
        format="%H:%M:%S",
        errors="coerce"
    )

    # ========================= Task 2.3 - Journey Duration =========================

    journey_df = (
        df.groupby("Train_No")
        .agg(
            Start_Time=("Departure_Time", "first"),
            End_Time=("Arrival_time", "last")
        )
        .reset_index()
    )

    journey_df["Journey_Duration"] = (
            journey_df["End_Time"] - journey_df["Start_Time"]
    )

    mask = journey_df["Journey_Duration"] < pd.Timedelta(0)

    journey_df.loc[mask, "Journey_Duration"] += pd.Timedelta(days=1)

    journey_df["Journey_Duration_Hours"] = (
            journey_df["Journey_Duration"].dt.total_seconds() / 3600
    )

    journey_df["Duration_Text"] = journey_df["Journey_Duration"].apply(
        lambda x: f"{int(x.total_seconds() // 3600)} hr "
                  f"{int((x.total_seconds() % 3600) // 60)} min"
    )

    df = df.merge(
        journey_df[
            [
                "Train_No",
                "Journey_Duration",
                "Journey_Duration_Hours",
                "Duration_Text"
            ]
        ],
        on="Train_No",
        how="left"
    )

    # ========================= Task 2.4 - Feature Engineering =========================

    stats_df = (
        df.groupby("Train_No")
        .agg(
            Total_Distance=("Distance", "max"),
            Total_Stops=("Station_Name", "count")
        )
        .reset_index()
    )

    df = df.merge(
        stats_df,
        on="Train_No",
        how="left"
    )

    # ========================= Overview Page Tables =========================

    structure_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Values": df.count().values
    })

    route_df = (
        df.groupby("Train_No")
        .agg(
            Source_Station=("Station_Name", "first"),
            Destination_Station=("Station_Name", "last")
        )
        .reset_index()
    )

    train_stats = (
        df.groupby("Train_No")
        .agg(
            Total_Distance=("Distance", "max"),
            Total_Stops=("Station_Name", "count")
        )
        .reset_index()
    )

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isna().sum().values,
        "Missing (%)": (
                df.isna().sum().values / len(df) * 100
        ).round(2)
    })

    # ========================= Cleaning Page Tables =========================

    train_journey_summary = (
        df.groupby("Train_No")
        .agg(
            Source_Station=("Station_Name", "first"),
            Destination_Station=("Station_Name", "last"),
            Journey_Duration=("Duration_Text", "first"),
            Duration_Hours=("Journey_Duration_Hours", "first"),
            Total_Distance=("Distance", "max"),
            Total_Stops=("Station_Name", "count")
        )
        .reset_index()
    )

    final_clean_dataset = (
        df.groupby("Train_No")
        .agg(
            Source_Station=("Station_Name", "first"),
            Destination_Station=("Station_Name", "last"),
            Source_Code=("Station_Code", "first"),
            Destination_Code=("Station_Code", "last"),
            Route_Number=("Route_Number", "first"),
            Departure_Time=("Departure_Time", "first"),
            Arrival_Time=("Arrival_time", "last"),
            Journey_Duration=("Duration_Text", "first"),
            Total_Distance=("Distance", "max"),
            Total_Stops=("Station_Name", "count")
        )
        .reset_index()
    )

    final_clean_dataset["Departure_Time"] = (
        final_clean_dataset["Departure_Time"].dt.strftime("%H:%M")
    )

    final_clean_dataset["Arrival_Time"] = (
        final_clean_dataset["Arrival_Time"].dt.strftime("%H:%M")
    )

    # ========================= Exploration Data =========================

    station_traffic_df = (
        df.groupby("Station_Name")
        .size()
        .reset_index(name="Traffic")
        .sort_values("Traffic", ascending=False)
    )

    journey_analysis_df = (
        df.groupby("Train_No")
        .agg(
            Total_Distance=("Distance", "max"),
            Journey_Duration_Hours=("Journey_Duration_Hours", "first")
        )
        .reset_index()
    )

    # ========================= Prediction Model Dataset =========================

    model_df = (
        df.groupby("Train_No")
        .agg(
            Total_Distance=("Distance", "max"),
            Total_Stops=("Station_Name", "count"),
            Journey_Duration_Hours=("Journey_Duration_Hours", "first")
        )
        .reset_index()
    )

    model_df = model_df.dropna()

    # ========================= Final Dashboard Data =========================

    distance_chart_df = (
        df.groupby("Train_No")
        .agg(
            Distance=("Distance", "max")
        )
        .reset_index()
        .sort_values("Distance", ascending=False)
        .head(10)
    )

    distance_chart_df["Train_No"] = (
        distance_chart_df["Train_No"].astype(str)
    )

    fare_chart_df = pd.DataFrame({
        "Class": ["1A", "2A", "3A", "SL"],
        "Total Fare": [
            df["1A"].sum(),
            df["2A"].sum(),
            df["3A"].sum(),
            df["SL"].sum()
        ]
    })

    traffic_chart_df = (
        df["Station_Name"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    traffic_chart_df.columns = [
        "Station",
        "Train_Count"
    ]

    distance_duration_df = (
        df.groupby("Train_No")
        .agg(
            Distance=("Total_Distance", "first"),
            Duration=("Journey_Duration_Hours", "first")
        )
        .reset_index()
    )

    return (
        df,
        structure_df,
        route_df,
        train_stats,
        missing_df,
        train_journey_summary,
        final_clean_dataset,
        station_traffic_df,
        journey_analysis_df,
        model_df,
        distance_chart_df,
        fare_chart_df,
        traffic_chart_df,
        distance_duration_df
    )

(df, structure_df, route_df, train_stats, missing_df, train_journey_summary,
 final_clean_dataset, station_traffic_df, journey_analysis_df, model_df,
 distance_chart_df, fare_chart_df, traffic_chart_df, distance_duration_df) = load_data()

params = st.query_params

if "page" not in params:
    params["page"] = "Home"

current_page = params["page"]

# ========================================== CUSTOM SIDEBAR CSS ==========================================

st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #06111f 0%,
        #0b1f3a 50%,
        #12345b 100%
    );
    border-right: 2px solid rgba(255,255,255,0.10);
}

/* Remove Streamlit default padding */
section[data-testid="stSidebar"] > div{
    padding-top: 1rem;
}

/* Title */
.sidebar-title{
    font-size:18px;
    font-weight:700;
    color:#E2E8F0;
    text-align:center;
    margin-bottom:20px;
    font-family:Inter,sans-serif;
    letter-spacing:0.5px;
    width:100%;
    display:block;
}

/* Divider */
.sidebar-divider{
    height:1px;
    background:rgba(255,255,255,0.08);
    margin:10px 0 20px 0;
}

ul.nav.nav-pills.flex-column{
    border:2px solid rgba(255,255,255,0.20);
    border-radius:18px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🚆 Train Route Analysis</div>',
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Overview",
            "Cleaning",
            "Exploration",
            "Visualization",
            "Journey Prediction",
            "Final Dashboard"
        ],
        icons=[
            "house",
            "table",
            "tools",
            "search",
            "bar-chart",
            "graph-up-arrow",
            "speedometer2"
        ],
        default_index=["Home", "Overview", "Cleaning", "Exploration", "Visualization",
                       "Journey Prediction", "Final Dashboard"].index(current_page),

        styles={

            # Container
            "container": {
                "padding": "10px",
                "background-color": "#112D4E",

                "border": "1px solid rgba(255,255,255,0.25)",

                "border-radius": "12px",

                "box-shadow": """
                        0 0 0 1px rgba(255,255,255,0.08),
                        0 0 0 3px rgba(125,211,252,0.04)
                """
            },

            # Icon
            "icon": {
                "color": "#7DD3FC",
                "font-size": "18px",
                "font-weight": "700",
                "transition": "0.2s"
            },

            # Menu Item
            "nav-link": {
                "font-size": "15px",
                "font-weight": "500",
                "text-align": "left",
                "padding": "12px",
                "margin": "6px 0",
                "border-radius": "12px",
                "color": "#BFDBFE",
                "--hover-color": "rgba(59,130,246,0.15)"
            },

            # Selected Item
            "nav-link-selected": {
                "background-color": "rgba(59,130,246,0.25)",
                "color": "#E0F2FE",
                "font-weight": "700",
                "border": "1px solid rgba(255,255,255,0.20)",
                "border-radius": "12px",
                "border-left": "4px solid #60A5FA"
            }
        }
    )

if current_page != selected:
    st.query_params["page"] = selected
    st.rerun()

# ========================================== CUSTOM MAIN PAGE CSS ==========================================

st.markdown("""<style>

/* ==========================================================
   TRAIN JOURNEY ANALYTICS - PREMIUM DASHBOARD UI
========================================================== */

:root{

    --navy-dark:#081225;
    --navy:#0B1F3A;
    --navy-light:#102B52;

    --rail-blue:#3B82F6;
    --rail-blue-light:#93C5FD;

    --success:#22C55E;
    --info:#3B82F6;
    --danger:#EF4444;

    --text:#F8FAFC;
    --muted:#94A3B8;

    --glass:rgba(255,255,255,0.04);
    --glass-hover:rgba(255,255,255,0.06);

    --border:rgba(255,255,255,0.10);

    --radius:20px;

    --shadow:
        0 10px 30px rgba(0,0,0,.25);

    --transition:all .3s ease;

}

/* ==========================================================
   APP BACKGROUND
========================================================== */

.stApp{

    background:

    radial-gradient(
        circle at top left,
        rgba(59,130,246,.12),
        transparent 30%
    ),

    radial-gradient(
        circle at bottom right,
        rgba(147,197,253,.08),
        transparent 35%
    ),

    linear-gradient(
        180deg,
        #081225 0%,
        #0B1F3A 45%,
        #102B52 100%
    );

    color:var(--text);
}

/* ==========================================================
   MAIN CONTAINER
========================================================== */

.main .block-container{

    max-width:95%;

    padding-top:1.5rem;

    padding-bottom:2rem;
}

/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #081225 0%,
        #0B1F3A 40%,
        #102B52 100%
    );

    border-right:
        2px solid rgba(255,255,255,.12);
}

section[data-testid="stSidebar"] *{

    color:white !important;
}

/* ==========================================================
   HEADINGS
========================================================== */

h1{

    font-family:'Inter',sans-serif;

    font-size:3.4rem !important;

    font-weight:800 !important;

    text-align:center;

    color:#FFFFFF !important;

    margin-bottom:10px;

    letter-spacing:.5px;
}

h2{

    color:#E2E8F0 !important;

    font-size:2rem !important;

    font-weight:700 !important;
}

h3{

    color:#CBD5E1 !important;

    font-size:1.4rem !important;

    font-weight:600 !important;
}

/* ==========================================================
   CAPTION
========================================================== */

div[data-testid="stCaptionContainer"]{

    color:#94A3B8 !important;

    font-size:15px;

    font-weight:500;
}

/* ==========================================================
   TEXT
========================================================== */

p,
li,
label{

    color:#CBD5E1 !important;

    font-size:15px;

    line-height:1.7;
}

/* ==========================================================
   PREMIUM DIVIDER
========================================================== */

hr{

    border:none !important;

    height:2px !important;

    border-radius:999px;

    background:

    linear-gradient(
        90deg,
        transparent,
        rgba(147,197,253,.4),
        rgba(59,130,246,.8),
        rgba(147,197,253,.4),
        transparent
    );

    box-shadow:
        0 0 15px rgba(59,130,246,.25);

    margin:24px 0;
}

/* ==========================================================
   GLASS CARD
========================================================== */

.glass-card{

    background:var(--glass);

    backdrop-filter:blur(18px);

    border:1px solid var(--border);

    border-radius:22px;

    padding:24px;

    box-shadow:var(--shadow);
}

/* ==========================================================
   PREMIUM KPI CARD
========================================================== */

.kpi-card{

    position:relative;

    overflow:hidden;

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.06),
        rgba(255,255,255,0.03)
    );

    backdrop-filter:blur(24px);
    -webkit-backdrop-filter:blur(24px);

    border:1px solid rgba(255,255,255,0.10);

    border-radius:22px;

    padding:24px;

    min-height:145px;

    text-align:center;

    transition:all 0.35s ease;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.25);

}

/* Soft Blue Glow Layer */

.kpi-card::before{

    content:"";

    position:absolute;

    top:-50%;

    left:-50%;

    width:200%;

    height:200%;

    background:
        radial-gradient(
            circle,
            rgba(147,197,253,0.08) 0%,
            transparent 70%
        );

    opacity:0;

    transition:0.4s ease;

    pointer-events:none;
}

/* Hover Effect */

.kpi-card:hover{

    transform:
        translateY(-6px)
        scale(1.02);

    border:
        1px solid rgba(147,197,253,0.45);

    box-shadow:
        0 18px 40px rgba(0,0,0,0.35),
        0 0 18px rgba(147,197,253,0.20),
        0 0 35px rgba(59,130,246,0.12);
}

.kpi-card:hover::before{

    opacity:1;
}

/* ==========================================================
   KPI ICON
========================================================== */

.kpi-icon{

    width:64px;

    height:64px;

    border-radius:18px;

    display:flex;

    align-items:center;

    justify-content:center;

    margin:auto;

    margin-bottom:16px;

    font-size:30px;

    color:#BFDBFE;

    background:
        linear-gradient(
            135deg,
            rgba(59,130,246,0.20),
            rgba(147,197,253,0.10)
        );

    border:
        1px solid rgba(147,197,253,0.18);

    box-shadow:
        0 8px 20px rgba(59,130,246,0.12);

    transition:all 0.35s ease;
}

/* Icon Hover */

.kpi-card:hover .kpi-icon{

    transform:scale(1.08);

    box-shadow:
        0 0 18px rgba(147,197,253,0.25),
        0 0 30px rgba(59,130,246,0.15);
}

/* ==========================================================
   KPI TITLE
========================================================== */

.kpi-title{

    color:#94A3B8;

    font-size:12px;

    font-weight:600;

    letter-spacing:1px;

    text-transform:uppercase;

    margin-bottom:8px;
}

/* ==========================================================
   KPI VALUE
========================================================== */

.kpi-value{

    font-size:1.2rem;

    font-weight:700;

    color:#FFFFFF;

    line-height:1.2;

    text-shadow:
        0 0 10px rgba(147,197,253,0.10);
}

/* ====================================
   SUCCESS
==================================== */

.stSuccess{

    background:
    linear-gradient(
        135deg,
        rgba(34,197,94,.18),
        rgba(22,163,74,.10)
    ) !important;

    border-radius:12px !important;
}

/* ====================================
   INNER CONTENT PADDING
==================================== */

.stSuccess > div{

    padding:18px 24px !important;
}

/* ====================================
   TEXT SPACING
==================================== */

.stSuccess p,
.stSuccess li{

    padding-left:8px !important;

    margin-left:4px !important;

    line-height:1.8 !important;
}

/* List Container */

.stSuccess ul,
.stSuccess ol{

    padding-left:28px !important;

    margin-top:10px !important;
}

/* ====================================
   INFO
==================================== */

.stInfo{

    background:
    linear-gradient(
        135deg,
        rgba(59,130,246,.18),
        rgba(37,99,235,.10)
    ) !important;

    border-radius:12px !important;
}

/* ====================================
   INNER CONTENT PADDING
==================================== */

.stInfo > div{

    padding:18px 24px !important;
}

/* ====================================
   TEXT SPACING
==================================== */

.stInfo p,
.stInfo li{

    padding-left:8px !important;

    margin-left:4px !important;

    line-height:1.8 !important;
}

/* List Container */

.stInfo ul,
.stInfo ol{

    padding-left:28px !important;

    margin-top:10px !important;
}

/* ====================================
   WARNING
==================================== */

.stWarning{

    background:
    linear-gradient(
        135deg,
        rgba(245,158,11,.18),
        rgba(217,119,6,.10)
    ) !important;

    border-radius:12px !important;
}

/* ====================================
   INNER CONTENT PADDING
==================================== */

.stWarning > div{

    padding:18px 24px !important;
}

/* ====================================
   TEXT SPACING
==================================== */

.stWarning p,
.stWarning li{

    padding-left:8px !important;

    margin-left:4px !important;

    line-height:1.8 !important;
}

/* List Container */

.stWarning ul,
.stWarning ol{

    padding-left:28px !important;

    margin-top:10px !important;
}


/* ==========================================================
   DATAFRAME - TRANSPARENT
========================================================== */

/* Header */
[data-testid="stDataFrame"] thead tr th {
    background-color: #1E4E8C !important;
    color: white !important;
    font-weight: 700 !important;
}

/* Body */
[data-testid="stDataFrame"] tbody tr td {
    background-color: #071424 !important;
    color: white !important;
}

/* Hover */
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #102A4D !important;
}

/* ==========================================================
   CHARTS
========================================================== */

.js-plotly-plot{

    background:rgba(255,255,255,.03);

    border-radius:22px;

    padding:10px;

    border:1px solid rgba(255,255,255,.08);

    backdrop-filter:blur(15px);

    box-shadow:var(--shadow);
}

.plotly,
.main-svg,
.svg-container{

    background:transparent !important;
}

/* ==========================================================
   SELECTBOX
========================================================== */

.stSelectbox div[data-baseweb="select"]{

    background:rgba(255,255,255,.04);

    border-radius:16px;

    border:1px solid rgba(255,255,255,.10);
}

/* ==========================================================
   NUMBER INPUT
========================================================== */

.stNumberInput input{

    background:rgba(255,255,255,.04) !important;

    color:white !important;

    border-radius:16px !important;

    border:1px solid rgba(255,255,255,.10) !important;
}

/* ==========================================================
   SLIDER
========================================================== */

.stSlider{

    padding-top:8px;
}

.stSlider [data-baseweb="slider"]{

    color:#60A5FA !important;
}

/* ==========================================================
   BUTTONS
========================================================== */

.stButton button{

    width:100%;

    height:52px;

    border:none;

    border-radius:16px;

    font-size:15px;

    font-weight:700;

    color:white !important;

    background:

    linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    );

    box-shadow:
        0 10px 25px rgba(59,130,246,.25);

    transition:all .3s ease;
}

.stButton button:hover{

    transform:translateY(-2px);

    box-shadow:
        0 14px 35px rgba(59,130,246,.35);
}

/* ==========================================================
   METRICS
========================================================== */

div[data-testid="metric-container"]{

    background:
    rgba(255,255,255,.04);

    border:1px solid rgba(255,255,255,.08);

    border-radius:18px;

    padding:15px;

    backdrop-filter:blur(16px);

    box-shadow:var(--shadow);
}

/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{

    width:10px;
}

::-webkit-scrollbar-track{

    background:#081225;
}

::-webkit-scrollbar-thumb{

    background:
    linear-gradient(
        180deg,
        #3B82F6,
        #60A5FA
    );

    border-radius:20px;
}

/* ==========================================================
   REMOVE STREAMLIT HEADER
========================================================== */

header[data-testid="stHeader"]{

    background:transparent;
}

footer{

    visibility:hidden;
}

/* ==========================================================
   PERFORMANCE
========================================================== */

*{

    -webkit-font-smoothing:antialiased;

    -moz-osx-font-smoothing:grayscale;
}

</style>

""", unsafe_allow_html=True)

# ========================= DATAFRAME FUNCTION =========================
def dark_table_style(df):

    def color_rows(x):
        styles = []

        for i in range(len(x)):
            bg = '#071424' if i % 2 == 0 else '#0D1B2A'
            styles.append([f'background-color: {bg}'] * len(x.columns))

        return pd.DataFrame(styles, index=x.index, columns=x.columns)

    return (
        df.style
        .apply(color_rows, axis=None)
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1E4E8C"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center")
                ]
            }
        ])
    )

# ========================= Home Page =========================

def home_page(df):
    st.markdown("""
    <div class="hero-card">

    <h1>🚆 Railway Analytics Hub</h1>

    <p>
    Analyze train routes, station traffic, journey duration and machine learning
    predictions through an intelligent railway analytics platform.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ========================= INTRODUCTION =========================

    st.subheader("📖 Project Overview")

    st.markdown("""
        This project applies **Data Science, Data Analytics, and Machine Learning**
        techniques to railway transportation data in order to uncover operational insights,
        analyze route performance, monitor station activity, and predict train journey duration.

        The dashboard follows a complete Data Science workflow including data inspection,
        cleaning, feature engineering, exploratory analysis, visualization, predictive modeling,
        and final business reporting.
        """)

    st.divider()

    # ========================= KPI SECTION =========================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title">Analysis Levels</div>
            <div class="kpi-value">6</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">📈</div>
            <div class="kpi-title">Visual Dashboards</div>
            <div class="kpi-value">12+</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">🤖</div>
            <div class="kpi-title">ML Model</div>
            <div class="kpi-value">Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon">🚆</div>
            <div class="kpi-title">Domain</div>
            <div class="kpi-value">Railway</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= FEATURES =========================

    st.subheader("✨ Dashboard Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stSuccess">

        <h3>📊 Data Overview</h3>

        • Dataset Inspection<br>
        • Missing Value Analysis<br>
        • Route Summary<br>
        • Data Quality Assessment

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stInfo">

        <h3>🔍 Exploratory Analysis</h3>

        • Route Comparison<br>
        • Traffic Analysis<br>
        • Distance Analysis<br>
        • Operational Insights

        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stWarning">

        <h3>🤖 Machine Learning</h3>

        • Journey Prediction<br>
        • Linear Regression<br>
        • Model Evaluation<br>
        • Real-time Estimation

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= PROJECT WORKFLOW =========================

    st.subheader("⚙️ Project Workflow")

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Level 1</h3>
            <b>Data Overview</b><br><br>
            Understand dataset structure and quality.
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🧹 Level 2</h3>
            <b>Data Cleaning & Feature Engineering</b><br><br>
            Prepare reliable features for analysis.
        </div>
        """, unsafe_allow_html=True)

    with row1_col3:
        st.markdown("""
        <div class="glass-card">
            <h3>🔍 Level 3</h3>
            <b>Data Exploration</b><br><br>
            Discover trends and relationships within railway operations.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ================= SECOND ROW =================

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📈 Level 4</h3>
            <b>Visualization & Pattern Analysis</b><br><br>
            Present insights using interactive visualizations.
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🧠 Level 5</h3>
            <b>Prediction Model Development</b><br><br>
            Build a Linear Regression model to predict journey duration.
        </div>
        """, unsafe_allow_html=True)

    with row2_col3:
        st.markdown("""
        <div class="glass-card">
            <h3>🚆 Level 6</h3>
            <b>Final Analytics Dashboard</b><br><br>
            Combine analysis, insights, and machine learning into a complete decision-support system.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= KEY CAPABILITIES =========================

    st.subheader("🚀 Key Capabilities")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="glass-card">

        <h3>🚆 Railway Analytics</h3>

        ✅ Train Route Analysis<br>
        ✅ Journey Duration Analysis<br>
        ✅ Distance Pattern Analysis<br>
        ✅ Station Traffic Monitoring<br>
        ✅ Route Performance Evaluation

        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">

        <h3>🤖 Machine Learning</h3>

        ✅ Prediction Engine<br>
        ✅ Actual vs Predicted<br>
        ✅ Interactive Visualizations<br>
        ✅ Business Insights<br>
        ✅ Decision Support

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= OBJECTIVE =========================

    st.markdown("""
    <div class="objective-card">

    <h2>🎯 Project Objective</h2>

    Develop an intelligent railway analytics platform capable of transforming
    raw transportation data into actionable insights while accurately predicting
    journey duration using machine learning.

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ========================= FOOTER =========================

    st.markdown("""
    <div class="footer-card">

    💡 Navigate through the sidebar to explore every stage of the Railway Analytics
    Data Science workflow.

    </div>
    """, unsafe_allow_html=True)

# ========================= Overview Page =========================

def overview_page(df):
    st.title("📊 Overview & Inspection")
    st.caption(
        "Analyze the dataset structure, data quality, completeness, and key statistics "
        "to establish a strong foundation for further analysis and modeling."
    )

    st.divider()

    # ========================= KPI CARDS =========================

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📄</div>
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-title">Total Columns</div>
            <div class="kpi-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-title">Missing Values</div>
            <div class="kpi-value">{int(df.isna().sum().sum())}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= Task 1.1 : Dataset Summary =========================

    st.subheader("📌 Dataset Summary")

    st.caption("This section provides a high-level overview of the dataset,"
               "including records, columns, and data types.")

    st.dataframe(dark_table_style(df.head()),use_container_width=True,hide_index=True)

    st.markdown("#### Dataset Structure")

    st.dataframe(dark_table_style(structure_df), use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # ========================= Task 1.2 : Train Route Information =========================

        st.subheader("🚆 Train Route Information")
        st.caption("Display train-wise route details showing source and destination stations.")

        st.dataframe(dark_table_style(route_df), use_container_width=True, hide_index=True, height=425)

    with col2:
        # ========================= Task 1.3 : Distance & Stop Statistics =========================

        st.subheader("📈 Distance & Stop Statistics")
        st.caption("Descriptive statistics help understand the distribution of distance and stop-related information.")

        st.dataframe(dark_table_style(train_stats), use_container_width=True, hide_index=True, height=425)

    st.divider()

    # ========================= Task 1.4 : Missing Value Analysis =========================

    st.subheader("🔍 Missing Value Analysis")
    st.caption("Analyze missing values in each column to assess data completeness "
               "and identify potential data quality issues.")

    st.dataframe(dark_table_style(missing_df), use_container_width=True, hide_index=True)

    st.success("Data Inspection Completed Successfully ✅")

# ========================= Cleaning Page =========================

def cleaning_page(df):
    st.title("🧹 Data Cleaning & Feature Engineering")

    st.caption("This section summarizes the data cleaning process, including the total records retained, "
               "missing values handled, and duplicate records removed.")

    st.divider()

    # ========================= KPI CARDS =========================

    missing_before = int(df.isnull().sum().sum())
    duplicate_before = int(df.duplicated().sum())
    total_records = len(df)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📋</div>
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{total_records:,}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-title">Missing Values Fixed</div>
                <div class="kpi-value">{missing_before:,}</div>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🗑️</div>
                <div class="kpi-title">Duplicates Removed</div>
                <div class="kpi-value">{duplicate_before:,}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ========================= TRAIN JOURNEY SUMMARY =========================

    st.subheader("🚂 Train Journey Performance Summary")

    st.caption("This table presents a train-wise overview of source and destination stations, total journey duration,"
               " distance traveled, and number of stops derived after data cleaning and feature engineering.")

    st.dataframe(dark_table_style(train_journey_summary), use_container_width=True, hide_index=True)

    st.divider()

    # ========================= CLEAN DATASET =========================

    st.subheader("🧹 Final Cleaned & Engineered Dataset")

    st.caption("This dataset contains train-level features prepared for exploratory analysis, "
               "visualization, and machine learning model development."
               )

    st.dataframe(dark_table_style(final_clean_dataset), use_container_width=True, hide_index=True)

    with st.expander("📌 Data Quality & Processing Summary", expanded=True):
        st.success("✅ Level 2 completed successfully. Data cleaning, preprocessing, "
                   "and feature engineering have been performed.")

        st.info("ℹ️ Key features such as Journey Duration, Total Distance,"
                " and Total Stops have been generated to support exploratory analysis and machine learning.")

# ========================= Exploration Page =========================

def exploration_page(df):

    # ========================= COLUMN MAPPING =========================

    station_col = "Station_Name"
    train_col = "Train_No"
    distance_col = "Distance"
    route_col = "Route_Number"

    df_explore = df

    st.title("🚆 Train Route Exploration Dashboard")
    st.caption("Explore train routes, station traffic and distance patterns")

    tab1, tab2 = st.tabs(["🚆 Data Exploration", "📈 Route Comparison"])

    with tab1:
        st.markdown("### 📊 Data Exploration Dashboard")

        station_traffic = (
            df_explore.groupby(station_col)[train_col]
            .count()
            .sort_values(ascending=False)
        )

        busiest_station = station_traffic.index[0]

        least_station = station_traffic.index[-1]

        high_traffic_stations = len(
            station_traffic[station_traffic > 200]
        )

        low_traffic_stations = len(
            station_traffic[station_traffic <= 200]
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🔥</div>
                <div class="kpi-title">High Traffic</div>
                <div class="kpi-value">{high_traffic_stations}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🟢</div>
                <div class="kpi-title">Low Traffic</div>
                <div class="kpi-value">{low_traffic_stations}</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🚉</div>
                <div class="kpi-title">Busiest Station</div>
                <div class="kpi-value">{busiest_station}</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📍</div>
                <div class="kpi-title">Least Busy</div>
                <div class="kpi-value">{least_station}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🚉 Station Traffic Analysis")
        st.caption("Analyze station-wise train traffic to identify the busiest and "
                   "least busy railway stations in the network.")

        traffic1, traffic2 = st.columns([4, 7])

        with traffic1:

            traffic_df = station_traffic_df

            st.dataframe(dark_table_style(traffic_df), use_container_width=True, hide_index=True, height=500)

        with traffic2:
            fig = px.bar(
                traffic_df.head(15),
                x=station_col,
                y="Traffic",
                color="Traffic",
                color_continuous_scale="Viridis",
                title="Top 15 Stations by Train Traffic"
            )

            fig.update_traces(
                texttemplate="%{y}",
                textposition="outside"
            )

            fig.update_layout(
                height=500,
                title_x=0.3,
                xaxis_title="Station Name",
                yaxis_title="Train Traffic",
                coloraxis_showscale=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### 📈 Distance vs Journey Duration Relationship Analysis")
        st.caption("Analyze how train travel distance influences overall journey duration.")

        journey_analysis = journey_analysis_df

        correlation = journey_analysis["Total_Distance"].corr(
            journey_analysis["Journey_Duration_Hours"]
        )

        avg_distance = journey_analysis["Total_Distance"].mean()

        avg_duration = journey_analysis["Journey_Duration_Hours"].mean()

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
            <div class="kpi-icon">📏</div>
            <div class="kpi-title">Average Distance</div>
            <div class="kpi-value">{avg_distance:.0f} KM</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
            <div class="kpi-icon">⏳</div>
            <div class="kpi-title">Average Duration</div>
            <div class="kpi-value">{avg_duration:.1f} hrs</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title">Correlation</div>
            <div class="kpi-value">{correlation:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.dataframe(
            dark_table_style(journey_analysis.rename(
                columns={
                    "Train_No": "Train Number",
                    "Total_Distance": "Total Distance (KM)",
                    "Journey_Duration_Hours": "Journey Duration (Hours)"
                }
            )),
            use_container_width=True, hide_index=True
        )

        if correlation >= 0.7:
            relation = "strong positive"
        elif correlation >= 0.4:
            relation = "moderate positive"
        else:
            relation = "weak"

        st.info(
            f"The analysis shows a {relation} relationship between distance and journey duration. "
            f"The correlation coefficient is {correlation:.2f}, indicating that trains covering "
            f"longer distances generally require more travel time."
        )

    with tab2:
        st.subheader("⚖️ Route Comparison Dashboard")

        # ========================= STATION SELECTION =========================

        col1, col2 = st.columns(2)

        source_station = col1.selectbox("📍 Select Source Station", sorted(df_explore[station_col].unique()))

        destination_station = col2.selectbox("🎯 Select Destination Station", sorted(df_explore[station_col].unique()))

        st.divider()

        # ========================= ROUTE FILTERING =========================

        if source_station != destination_station:
            source_trains = set(df_explore[df_explore[station_col] == source_station][train_col])

            destination_trains = set(df_explore[df_explore[station_col] == destination_station][train_col])

            common_trains = source_trains.intersection(destination_trains)

            if len(common_trains) == 0:

                st.error("❌ No direct trains found between selected stations.")

            else:

                route_data = []

                for train in common_trains:

                    temp = df_explore[df_explore[train_col] == train].copy()

                    temp = temp.sort_values(distance_col)

                    source_rows = temp[temp[station_col] == source_station]

                    destination_rows = temp[temp[station_col] == destination_station]

                    if len(source_rows) > 0 and len(destination_rows) > 0:

                        source_distance = source_rows[distance_col].iloc[0]

                        destination_distance = destination_rows[distance_col].iloc[0]

                        if destination_distance > source_distance:
                            temp = temp[
                                (temp[distance_col] >= source_distance)
                                &
                                (temp[distance_col] <= destination_distance)
                                ]

                            route_data.append(temp)

                if len(route_data) == 0:

                    st.warning("⚠️ No valid routes found in selected direction.")

                else:

                    selected_routes_df = pd.concat(route_data, ignore_index=True)

                    # ========================= KPIs Section =========================

                    total_trains = selected_routes_df[train_col].nunique()

                    max_distance = selected_routes_df.groupby(train_col)[distance_col].max().max()

                    station_traffic = selected_routes_df[station_col].value_counts()

                    busiest_station = station_traffic.idxmax()

                    least_station = station_traffic.idxmin()

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">🚆</div>
                            <div class="kpi-title">Total Trains</div>
                            <div class="kpi-value">{total_trains}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">📏</div>
                            <div class="kpi-title">Maximum Distance</div>
                            <div class="kpi-value">{max_distance:.0f} KM</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">🔥</div>
                            <div class="kpi-title">Highest Traffic</div>
                            <div class="kpi-value">{busiest_station}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-icon">📍</div>
                            <div class="kpi-title">Lowest Traffic</div>
                            <div class="kpi-value">{least_station}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    # ========================= TRAIN TABLE =========================

                    st.markdown("### 📋 Available Train Services for the Selected Route")
                    st.caption("This table provides a summary of all trains operating between the selected source and"
                               " destination stations, including departure and arrival times, total distance, "
                               "number of stops, and estimated journey duration."
                               )

                    # ========================= TRAIN SUMMARY TABLE =========================

                    display_df = (
                        selected_routes_df
                        .sort_values(["Train_No", "Distance"])
                        .groupby("Train_No").agg(
                            Source_Station=("Station_Name", "first"),
                            Destination_Station=("Station_Name", "last"),
                            Source_Code=("Station_Code", "first"),
                            Destination_Code=("Station_Code", "last"),
                            Departure_Time=("Departure_Time", "first"),
                            Arrival_Time=("Arrival_time", "last"),
                            Total_Distance=("Distance", "max"),
                            Total_Stops=("Station_Name", "count"),
                        ).reset_index()
                    )

                    # ========================= CALCULATE ACTUAL JOURNEY DURATION =========================

                    display_df["Departure_Time"] = pd.to_datetime(
                        display_df["Departure_Time"], format="%H:%M", errors="coerce")

                    display_df["Arrival_Time"] = pd.to_datetime(
                        display_df["Arrival_Time"], format="%H:%M", errors="coerce")

                    # Handle next-day arrivals
                    mask = display_df["Arrival_Time"] < display_df["Departure_Time"]

                    display_df.loc[mask, "Arrival_Time"] += pd.Timedelta(days=1)

                    # Calculate duration
                    display_df["Journey_Duration"] = (
                            display_df["Arrival_Time"]
                            - display_df["Departure_Time"]
                    )

                    # Numeric duration for graphs
                    display_df["Duration_Hours"] = (
                            display_df["Journey_Duration"]
                            .dt.total_seconds() / 3600
                    )

                    # Text duration for table
                    display_df["Duration_Text"] = (
                        display_df["Journey_Duration"]
                        .apply(
                            lambda x:
                            f"{int(x.total_seconds() // 3600)} hr "
                            f"{int((x.total_seconds() % 3600) // 60)} min"
                            if pd.notnull(x)
                            else "N/A"
                        )
                    )

                    # ========================= FORMAT TIMES =========================

                    display_df["Departure_Time"] = display_df["Departure_Time"].dt.strftime("%H:%M")

                    display_df["Arrival_Time"] = display_df["Arrival_Time"].dt.strftime("%H:%M")

                    # ========================= DISPLAY TABLE =========================

                    st.dataframe(dark_table_style(display_df), use_container_width=True, hide_index=True)

                    st.divider()

                    # ========================= GRAPH SECTION =========================

                    st.subheader("📈 Distance and Journey Duration Analysis")
                    st.caption("Analyze the relationship between distance, journey duration and "
                               "train stops for the selected route.")

                    duration_df = display_df.copy()

                    duration_df["Train_No"] = duration_df["Train_No"].astype(str)

                    duration_df = duration_df.sort_values("Duration_Hours", ascending=False)

                    # ========================= GRAPH 1 : JOURNEY DURATION COMPARISON =========================

                    fig1 = px.bar(
                        duration_df,
                        x="Train_No",
                        y="Duration_Hours",
                        text="Duration_Hours",
                        color="Duration_Hours",
                        color_continuous_scale="Viridis",
                        hover_data=[
                            "Source_Station",
                            "Destination_Station",
                            "Total_Distance",
                            "Total_Stops"
                        ], title="Journey Duration Comparison"
                    )

                    fig1.update_traces(
                        texttemplate="%{text:.1f} hrs",
                        textposition="outside"
                    )

                    fig1.update_layout(
                        height=550,
                        title_x=0.35,
                        xaxis_title="Train Number",
                        yaxis_title="Duration (Hours)",
                        xaxis=dict(type="category")
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                    st.info(
                        "This chart compares travel time across all trains operating on the selected route."
                    )

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:

                        # ========================= GRAPH 2 : DISTANCE VS JOURNEY DURATION =========================

                        fig2 = px.scatter(
                            display_df,
                            x="Total_Distance",
                            y="Duration_Hours",
                            hover_name="Train_No",
                            hover_data={
                                "Source_Station": True,
                                "Destination_Station": True,
                                "Total_Stops": True
                            },
                            size="Total_Stops",
                            color="Duration_Hours",
                            color_continuous_scale="Turbo",
                            title="Distance vs Journey Duration"
                        )

                        fig2.update_layout(
                            height=550,
                            title_x=0.2,
                            xaxis_title="Distance (KM)",
                            yaxis_title="Journey Duration (Hours)"
                        )

                        st.plotly_chart(fig2, use_container_width=True)

                        st.info(
                            "Longer routes generally require more travel time, indicating a positive relationship between distance and journey duration."
                        )

                    with col2:
                        # ========================= GRAPH 3 : STOPS VS JOURNEY DURATION =========================

                        fig3 = px.scatter(
                            display_df,
                            x="Total_Stops",
                            y="Duration_Hours",
                            size="Total_Distance",
                            color="Total_Distance",
                            color_continuous_scale="Plasma",
                            hover_name="Train_No",
                            hover_data={
                                "Total_Distance": True
                            },
                            title="Impact of Stops on Journey Duration"
                        )

                        fig3.update_layout(
                            height=550,
                            title_x=0.2,
                            xaxis_title="Total Stops",
                            yaxis_title="Journey Duration (Hours)"
                        )

                        st.plotly_chart(fig3, use_container_width=True)

                        st.info(
                            "Trains with more intermediate stops generally take longer to complete their journey.")

                    st.divider()

                    correlation = display_df["Total_Distance"].corr(
                        display_df["Duration_Hours"]
                    )

                    st.subheader("💡 Key Exploratory Insights")

                    longest_train = display_df.loc[
                        display_df["Duration_Hours"].idxmax(),
                        "Train_No"
                    ]

                    shortest_train = display_df.loc[
                        display_df["Duration_Hours"].idxmin(),
                        "Train_No"
                    ]

                    max_duration = display_df["Duration_Hours"].max()

                    min_duration = display_df["Duration_Hours"].min()

                    avg_duration = display_df["Duration_Hours"].mean()

                    avg_distance = display_df["Total_Distance"].mean()

                    avg_stops = display_df["Total_Stops"].mean()

                    st.success(f"""
                    🚆 **{total_trains} trains** operate between **{source_station}** and **{destination_station}**.

                    📏 Average route distance is **{avg_distance:.0f} KM**.

                    ⏱️ Average journey duration is **{avg_duration:.1f} hours**.

                    🛑 Trains make an average of **{avg_stops:.0f} stops** on this route.

                    🏆 Train **{longest_train}** has the longest journey duration (**{max_duration:.1f} hrs**).

                    ⚡ Train **{shortest_train}** has the shortest journey duration (**{min_duration:.1f} hrs**).

                    🔥 The busiest station on this route is **{busiest_station}**.

                    🟢 The least busy station on this route is **{least_station}**.

                    📊 Distance and journey duration show a correlation of **{correlation:.2f}**, indicating that longer routes generally require more travel time.
                        """)

# ========================= Visualization Page =========================

def visualization_page(df):
    st.title("📊 Smart Railway Analytics Dashboard")
    st.caption(
        "Explore train movement, station activity, and distance patterns "
        "through interactive visualizations and railway operational insights."
    )

    st.divider()

    # ==========================
    # FILTERS
    # ==========================

    with st.expander("🔍 Analysis Filters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            selected_stations = st.multiselect(
                "🏢 Stations",
                options=sorted(df["Station_Name"].dropna().unique()),
                default=sorted(df["Station_Name"].head(15).dropna().unique())
            )

        with col2:
            distance_range = st.slider(
                "📏 Distance (KM)",
                min_value=int(df["Distance"].min()),
                max_value=int(df["Distance"].max()),
                value=(
                    int(df["Distance"].min()),
                    int(df["Distance"].max())
                )
            )

    # ==========================
    # APPLY FILTERS
    # ==========================

    filtered_df = df.copy()

    # Station Filter
    if selected_stations:
        filtered_df = filtered_df[
            filtered_df["Station_Name"].isin(selected_stations)
        ]

    # Distance Filter
    filtered_df = filtered_df[
        (filtered_df["Distance"] >= distance_range[0]) &
        (filtered_df["Distance"] <= distance_range[1])
        ]

    # Optional: show filter result summary
    st.caption(
        f"Showing {len(filtered_df):,} records | "
        f"{filtered_df['Train_No'].nunique():,} trains | "
        f"{filtered_df['Station_Name'].nunique():,} stations"
    )

    st.divider()

    st.subheader("🚆 Train Movement Distribution")
    st.caption(
        "Compare station coverage across train services to identify trains "
        "with the highest operational reach."
    )

    train_counts = (
        filtered_df.groupby("Train_No")
        .size()
        .reset_index(name="Station_Stops")
        .sort_values("Station_Stops", ascending=False)
        .head(15)
    )

    # Convert Train Number to String

    train_counts["Train_No"] = train_counts["Train_No"].astype(str)

    fig = px.bar(
        train_counts,
        x="Train_No",
        y="Station_Stops",
        color="Station_Stops",
        text="Station_Stops",
        title="Top Trains by Number of Station Stops",
        color_continuous_scale=[
            "#1E3A8A",
            "#2563EB",
            "#60A5FA"
        ]
    )

    fig.update_xaxes(
        type="category",
        title="Train Number"
    )

    fig.update_yaxes(
        title="Number of Station Stops"
    )

    fig.update_layout(
        showlegend=False,
        title_x=0.3,
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"💡 **Observation:** Multiple top-tier routes are tied at a maximum peak coverage of"
        f" {train_counts['Station_Stops'].max()} station stops.")

    st.divider()

    st.subheader("🏢 Station-Wise Train Traffic Analysis")
    st.caption(
        "Identify high-traffic railway stations based on train stop frequency."
    )

    station_traffic = (
        station_traffic_df[
            station_traffic_df["Station_Name"].isin(selected_stations)
        ]
        if selected_stations
        else station_traffic_df
    )

    station_traffic = station_traffic.head(20)

    station_traffic.columns = ["Station_Name", "Train_Count"]

    fig = px.bar(
        station_traffic,
        x="Station_Name",
        y="Train_Count",
        color="Train_Count",
        text_auto=True,
        title="Top Busiest Railway Stations",
        color_continuous_scale=[
            "#0F172A",  # Dark Navy
            "#1E40AF",  # Railway Blue
            "#2563EB",  # Primary Blue
            "#3B82F6"  # Light Blue
        ]
    )

    fig.update_layout(
        height=500,
        title_x=0.4,
        xaxis=dict(
            title="Railway Station",
            tickangle=-45,
            showgrid=False,
            title_font=dict(size=15),
            tickfont=dict(size=11)
        ),

        yaxis=dict(
            title="Train Traffic Count",
            zeroline=False,
            title_font=dict(size=15)
        ),

        font=dict(
            family="Arial",
            size=12,
            color="#111827"
        ),

        margin=dict(
            l=30,
            r=30,
            t=80,
            b=100
        ),

        coloraxis_showscale=False
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(
            size=11,
            color="white"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"💡 **Observation:** `{station_traffic['Station_Name'].iloc[0]}` dominates overall railway traffic volume with "
        f"{station_traffic['Train_Count'].max()} total scheduled arrivals.")

    st.divider()

    st.subheader("📏 Distance Distribution Across Train Stops")
    st.caption(
        "Explore how train stops are distributed over different travel distances."
    )

    fig = px.histogram(
        filtered_df,
        x="Distance",
        nbins=40,
        title="Distribution of Train Stop Distances",
        color_discrete_sequence=["#3B82F6"]  # Modern Professional Blue
    )

    fig.update_traces(
        opacity=0.9,
        hovertemplate=
        "<b>Distance Range:</b> %{x}<br>"
        "<b>Train Stops:</b> %{y}<extra></extra>"
    )

    fig.update_layout(
        title=dict(
            text="Distribution of Train Stop Distances",
            x=0.5,
            xanchor="center",
        ),

        xaxis=dict(
            title="Distance from Origin Station (KM)",
        ),

        yaxis=dict(
            title="Number of Train Stops",
        ),

        template="plotly",
        height=500,
        bargap=0.03,
        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "💡 **Observation:** Station network density heavily skews toward short-to-midrange local"
        " transit runs before tapering off past 1,000 KM.")
    st.divider()

    st.subheader("⏱️ Distance vs Journey Duration Analysis")
    st.caption(
        "Analyze how journey duration varies with travel distance across train services."
    )

    journey_analysis = journey_analysis_df

    fig = px.scatter(
        journey_analysis,
        x="Total_Distance",
        y="Journey_Duration_Hours",
        color="Journey_Duration_Hours",
        hover_name="Train_No",
        title="Distance vs Journey Duration Relationship",
        trendline="ols",
        color_continuous_scale=[
            "#0B1F5B",  # Dark Blue
            "#1F77B4",  # Blue
            "#A6CEE3",  # Light Blue
            "#FFFFFF"  # White
        ]
    )

    fig.update_traces(
        marker=dict(size=9,
                    line=dict(
                        width=1,
                        color="black"))
    )

    fig.update_layout(
        height=500,
        title_x=0.3,
        xaxis_title="Total Distance (KM)",
        yaxis_title="Journey Duration (Hours)",
        legend_title="Duration Level"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Observation:** The flat OLS trendline reveals significant speed variation, "
        "highlighting distinct segments of express vs slower halting services.")

    st.divider()

    st.success(" 🎯 Conclusion : The visual analysis highlights key patterns in train movement, station utilization, and travel behavior. These insights provide a better understanding of railway operations,"
               " route characteristics, and network performance across the transportation system.")

# ========================= Journey Prediction Page =========================

def prediction_page(df):
    st.title("🚆 Train Journey Duration Prediction")
    st.caption("Predict train journey duration using machine learning based on travel distance and number of stops.")
    st.divider()

    # ========================= MODEL DATASET =========================

    X = model_df[["Total_Distance", "Total_Stops"]]
    y = model_df["Journey_Duration_Hours"]

    # ========================= TRAIN TEST SPLIT =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ========================= LINEAR REGRESSION MODEL =========================

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ========================= MODEL EVALUATION =========================

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    score = model.score(X_test, y_test)

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📉</div>
            <div class="kpi-title">MAE</div>
            <div class="kpi-value">{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-title">RMSE</div>
            <div class="kpi-value">{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🚆</div>
            <div class="kpi-title">R² SCORE</div>
            <div class="kpi-value">{score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========================= USER INPUT =========================

    st.subheader("🧠 Predict Journey Duration")
    st.caption(
        "Enter journey details to estimate the expected travel duration using the trained Linear Regression model.")

    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input(
            "📍 Total Distance (KM)",
            min_value=1,
            value=500,
            step=10
        )

    with col2:
        stops = st.number_input(
            "🚉 Total Stops",
            min_value=1,
            value=10,
            step=1
        )

    st.markdown("")

    predict_btn = st.button(
        "🚀 Predict Duration",
        use_container_width=True
    )

    if predict_btn:
        input_df = pd.DataFrame({
            "Total_Distance": [distance],
            "Total_Stops": [stops]
        })

        prediction = model.predict(input_df)[0]

        hours = int(prediction)
        minutes = int((prediction - hours) * 60)

        st.success(
            f"🚆 Estimated Journey Time: {hours} Hours {minutes} Minutes"
        )

        # ========================= VISUAL COMPARISON =========================

        comparison_df = pd.DataFrame({
            "Distance": X_test["Total_Distance"],
            "Actual Duration": y_test,
            "Predicted Duration": y_pred
        })

        # User selected distance
        selected_distance = distance

        # Filter nearby distances (±100 KM)
        nearby_df = comparison_df[
            (comparison_df["Distance"] >= selected_distance - 100) &
            (comparison_df["Distance"] <= selected_distance + 100)
            ].copy()

        # Sort by closest distance
        nearby_df["Distance_Diff"] = abs(
            nearby_df["Distance"] - selected_distance
        )

        nearby_df = nearby_df.sort_values(
            "Distance_Diff"
        ).head(30)

        fig = px.scatter(
            nearby_df,
            x="Distance",
            y=["Actual Duration", "Predicted Duration"],
            title=f"🚆 Journey Duration Prediction Near {selected_distance} KM",
            labels={
                "Distance": "Distance (KM)",
                "value": "Journey Duration (Hours)",
                "variable": "Duration Type"
            },
        )

        fig.update_traces(
            mode="markers",
            marker=dict(
                size=10,
                line=dict(width=1)
            )
        )

        fig.update_layout(
            height=500,
            title_x=0.27,
            title_font=dict(size=18),
            legend_title_text="Duration",
            hovermode="x unified",
            xaxis=dict(
                title="Distance (KM)",
                showgrid=True
            ),
            yaxis=dict(
                title="Journey Duration (Hours)",
                showgrid=True
            ),
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info("""Observation:
        The model demonstrates a strong correlation between actual and predicted journey durations, showing reliable performance for trains with comparable travel distances.
        """)

    st.divider()

    # ========================= FEATURE IMPORTANCE =========================

    st.subheader("📈 Feature Impact")
    st.caption("Analyze how distance and number of stops influence journey duration predictions.")

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    feature_df["Importance"] = (
        feature_df["Coefficient"].abs()
    )

    feature_df = feature_df.sort_values(
        "Importance",
        ascending=True
    )

    fig2 = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="🎯 Feature Importance Ranking",
        template="plotly_white"
    )

    fig2.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig2.update_layout(
        height=400,
        title_x=0.4,
        xaxis_title="Importance Score",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # ========================= NEAREST TRAIN DATA =========================

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        "Train_No": model_df.loc[X_test.index, "Train_No"],
        "Total_Distance": X_test["Total_Distance"],
        "Total_Stops": X_test["Total_Stops"],
        "Actual_Duration": y_test,
        "Predicted_Duration": y_pred
    })

    # User selected distance
    selected_distance = distance

    # Find nearby distances (±100 KM)
    nearby_df = comparison_df[
        (comparison_df["Total_Distance"] >= selected_distance - 100) &
        (comparison_df["Total_Distance"] <= selected_distance + 100)
        ].copy()

    # Sort by nearest distance
    nearby_df["Distance_Diff"] = abs(
        nearby_df["Total_Distance"] - selected_distance
    )

    nearby_df = nearby_df.sort_values(
        by="Distance_Diff"
    )

    # Keep only required columns
    nearby_df = nearby_df[
        [
            "Train_No",
            "Total_Distance",
            "Total_Stops",
            "Actual_Duration",
            "Predicted_Duration"
        ]
    ]

    nearby_df[["Actual_Duration", "Predicted_Duration"]] = (
        nearby_df[["Actual_Duration", "Predicted_Duration"]]
        .round(2)
    )

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
                <div class="kpi-card">
                <div class="kpi-icon">🗺️</div>
                <div class="kpi-title">Similar Routes</div>
                <div class="kpi-value">{len(nearby_df)}</div>
                </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
                <div class="kpi-card">
                <div class="kpi-icon">📍</div>
                <div class="kpi-title">Avg Distance</div>
                <div class="kpi-value">{nearby_df['Total_Distance'].mean():.0f} KM</div>
                </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
                <div class="kpi-card">
                <div class="kpi-icon">⌛</div>
                <div class="kpi-title">Avg Duration</div>
                <div class="kpi-value">{nearby_df['Actual_Duration'].mean():.1f} Hrs</div>
                </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("🚄 Similar Train Routes")

    st.caption(
        f"Showing train records with distances closest to {selected_distance} KM."
    )

    st.dataframe(dark_table_style(nearby_df),use_container_width=True,hide_index=True)

# ========================= Final Dashboard Page =========================

def final_page(df):
    st.title("🚆 Indian Railway Journey Analytics Dashboard")
    st.caption(
        "Comprehensive analysis of train routes, fares, journey duration, station traffic and machine learning insights."
    )
    st.divider()

    total_trains = df["Train_No"].nunique()

    total_stations = df["Station_Name"].nunique()

    avg_distance = round(df["Distance"].mean(), 2)

    avg_duration = round(
        df["Journey_Duration_Hours"].mean(),
        2
    )

    total_fare_value = (
            df["1A"].sum()
            + df["2A"].sum()
            + df["3A"].sum()
            + df["SL"].sum()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🚄</div>
                <div class="kpi-title">Total Trains</div>
                <div class="kpi-value">{total_trains:,}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📍</div>
                <div class="kpi-title">Total Stations</div>
                <div class="kpi-value">{total_stations:,}</div>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🛤</div>
                <div class="kpi-title">Avg Distance</div>
                <div class="kpi-value">{avg_distance:.0f} KM</div>
            </div>
            """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
               <div class="kpi-card">
                   <div class="kpi-icon">⏱</div>
                   <div class="kpi-title">Avg Duration</div>
                   <div class="kpi-value">{avg_duration:.2f} Hrs</div>
               </div>
               """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
               <div class="kpi-card">
                   <div class="kpi-icon">💰</div>
                   <div class="kpi-title">Total Fare Value</div>
                   <div class="kpi-value">₹{total_fare_value:,.0f}</div>
               </div>
               """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Operational Analytics & Route Insights")

    st.caption(
        "Explore key performance indicators, route statistics, fare distribution, station activity, and travel behavior patterns extracted from railway journey data."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            distance_chart_df,
            x="Train_No",
            y="Distance",
            title="🚆 Top 10 Long Distance Trains",
            text="Distance",
            color="Distance",
            color_continuous_scale=[
                "#1E3A8A",  # Deep Blue
                "#2563EB",  # Royal Blue
                "#38BDF8"  # Sky Blue
            ]
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=
            "<b>Train No:</b> %{x}<br>" +
            "<b>Distance:</b> %{y:,.0f} KM<extra></extra>"
        )

        fig.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            title={
                "x": 0.5,
                "xanchor": "center",
                "font": {
                    "family": "Arial Black"
                }
            },

            xaxis_title="Train Number",
            yaxis_title="Distance (KM)",

            coloraxis_showscale=False,

            font=dict(
                size=14
            ),

            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            ),

            xaxis=dict(
                type="category",
                showgrid=False,
                tickangle=-20
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(180,180,180,0.2)",
                zeroline=False
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Insight: These trains cover the longest travel distances in the dataset, indicating major intercity or long-haul routes that contribute significantly to network connectivity and passenger movement."
        )

    with col2:
        fig2 = px.pie(
            fare_chart_df,
            names="Class",
            values="Total Fare",
            hole=0.55,
            title="💰 Fare Distribution By Class",
            color="Class",
            color_discrete_map={
                "1A": "#1E3A8A",  # Premium Blue
                "2A": "#2563EB",  # Royal Blue
                "3A": "#60A5FA",  # Light Blue
                "SL": "#93C5FD"  # Sky Blue
            }
        )

        fig2.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=
            "<b>%{label}</b><br>"
            "Fare: ₹%{value:,.0f}<br>"
            "Share: %{percent}<extra></extra>",
        )

        fig2.update_layout(
            title={
                "text": "💰 Fare Distribution By Class",
                "x": 0.5,
                "xanchor": "center"
            },
            font=dict(
                size=14
            ),
            legend=dict(
                orientation="h",
                y=-0.15,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(
                t=70,
                b=50,
                l=20,
                r=20
            ),
            height=500
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.info(
            "Insight: This distribution highlights the contribution of each ticket class (1A, 2A, 3A, and Sleeper) to the overall fare value, providing an overview of pricing structure across train categories."
        )

    st.divider()

    fig3 = px.bar(
        traffic_chart_df.sort_values(
            "Train_Count",
            ascending=True
        ),
        x="Train_Count",
        y="Station",
        orientation="h",
        text="Train_Count",
        title="🚉 Top 15 Busiest Railway Stations",
        color="Train_Count",
        color_continuous_scale="Blues"
    )

    fig3.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    fig3.update_layout(
        height=550,
        title_x=0.4,
        xaxis_title="Number of Trains",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.info(
        "Insight: These stations experience the highest train traffic and serve as critical transportation hubs within the railway network."
    )

    st.divider()

    fig4 = px.scatter(
        distance_duration_df,
        x="Distance",
        y="Duration",
        title="🚆 Distance vs Journey Duration Analysis",
        color="Duration",
        color_continuous_scale="Turbo",
        size="Duration",
        size_max=18
    )

    fig4.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(
                width=1.5,
                color="rgba(30,30,30,0.4)"
            )
        ),
        hovertemplate=
        "<b>📍 Distance:</b> %{x} KM<br>" +
        "<b>⏱ Duration:</b> %{y:.2f} Hours<br>" +
        "<extra></extra>"
    )

    fig4.update_layout(
        height=500,
        title_x=0.3,
        template="plotly_dark",
        hovermode="closest",
        coloraxis_colorbar=dict(
            title="Hours"
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        ),
        xaxis=dict(
            title="Distance (KM)",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(255,255,255,0.08)"
        ),
        yaxis=dict(
            title="Journey Duration (Hours)",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(255,255,255,0.08)"
        )
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.info(
        "Insight: Journey duration generally increases with travel distance. The positive correlation observed here validates distance as one of the strongest predictors used in the machine learning model."
    )

    st.divider()

    s, i = st.columns(2)

    with s:
        st.success(
            f"📊 Analysis Summary: The dataset includes {total_trains:,} unique train routes and {total_stations:,} stations, with an average journey distance of {avg_distance:.0f} KM and average travel duration of {avg_duration:.2f} hours, highlighting a strong relationship between route distance, stops, and journey time."
        )

    with i:
        st.info(
            "🎯 Project Status: Successfully completed an end-to-end Data Science workflow including data preprocessing, feature engineering, exploratory analysis, visualization, machine learning model development, journey duration prediction, and interactive dashboard reporting."
        )


# ===========================================================================
# ALL FUNCTION CALL
# ===========================================================================


# ========================= Home Page =========================

if current_page == "Home": home_page(df)

# ========================= Level 1: Data Overview =========================

if current_page == "Overview": overview_page(df)

# ========================= Level 2: Data Cleaning & Feature Engineering =========================

if current_page == "Cleaning": cleaning_page(df)

# ========================= Level 3: Data Exploration =========================

if current_page == "Exploration": exploration_page(df)

# ========================= Level 4: Visualization & Pattern Analysis =========================

if current_page == "Visualization": visualization_page(df)

# ========================= Level 5: Prediction Model Development =========================

if current_page == "Journey Prediction": prediction_page(df)

# ========================= Level 6: Final Data Science Project =========================

if current_page == "Final Dashboard": final_page(df)