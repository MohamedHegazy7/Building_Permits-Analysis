import streamlit as st
import pandas as pd
import plotly.express as px


# Page configuration
st.set_page_config(
    page_title="Building Permits — Overview",
    page_icon="🏗️",
    layout="wide",
)


# --- styles
st.markdown(
    """
    <style>
    .title-main {font-size:28px; font-weight:700; color:#1f77b4;}
    .section-header {font-size:18px; font-weight:700; margin-top:18px; margin-bottom:8px;}
    .info {background:#f0f2f6; padding:12px; border-radius:8px;}
    .col-desc {background:#eef5fb; padding:10px; border-radius:6px; margin-bottom:6px}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- data loader
@st.cache_data
def load_data(path: str = "Building_Permits.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        # return empty df on failure
        return pd.DataFrame()
    return df


df = load_data()

st.markdown("<div class='title-main'>🏗️ San Francisco Building Permits</div>", unsafe_allow_html=True)

if df.empty:
    st.error("Could not load 'Building_Permits.csv' — place file in project folder.")
    st.stop()


# ---------- Dataset overview
st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Records", f"{len(df):,}")

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    # try to parse a common date column for date range
    if "Permit Creation Date" in df.columns:
        try:
            dates = pd.to_datetime(df["Permit Creation Date"], errors="coerce")
            if dates.notna().any():
                st.metric("Date range", f"{dates.min().year} - {dates.max().year}")
            else:
                st.metric("Date range", "N/A")
        except Exception:
            st.metric("Date range", "N/A")
    else:
        st.metric("Date range", "N/A")

with col4:
    st.metric("Missing values", f"{df.isnull().sum().sum():,}")


# ---------- Column descriptions
st.markdown("<div class='section-header'>Column Descriptions</div>", unsafe_allow_html=True)
with st.expander("Show full column dictionary", expanded=False):
    # full mapping based on dataset columns
    column_descriptions = {
        "Permit Number": "Unique identifier/tracking number for each building permit",
        "Permit Type": "Type/category of the permit issued",
        "Permit Type Definition": "Detailed definition explaining the permit type",
        "Permit Creation Date": "Date when the permit was first created/submitted",
        "Block": "City block number where the project is located",
        "Lot": "Lot number within the block",
        "Street Number": "Street address number",
        "Street Number Suffix": "Suffix for the street number when present",
        "Street Name": "Name of the street where the project is located",
        "Street Suffix": "Street suffix (St, Ave, Blvd, etc.)",
        "Unit": "Unit or apartment number",
        "Unit Suffix": "Suffix for unit when present",
        "Description": "Description of the work/project the permit is for",
        "Current Status": "Current permit status (issued, complete, withdrawn, etc.)",
        "Current Status Date": "Date when current status was recorded",
        "Filed Date": "Date the permit was filed",
        "Issued Date": "Date the permit was issued",
        "Completed Date": "Date the permit's work was marked completed",
        "First Construction Document Date": "Date of the first construction document submission",
        "Structural Notification": "Indicates if structural notification was required",
        "Number of Existing Stories": "Existing building stories",
        "Number of Proposed Stories": "Proposed number of stories",
        "Voluntary Soft-Story Retrofit": "Flag for voluntary retrofit",
        "Fire Only Permit": "Flag when the permit is fire-only",
        "Permit Expiration Date": "Expiration date for the permit",
        "Estimated Cost": "Estimated cost of the project (USD)",
        "Revised Cost": "Revised/actual cost (USD)",
        "Existing Use": "Existing use (e.g., 1 family dwelling)",
        "Existing Units": "Number of existing units",
        "Proposed Use": "Proposed use after work",
        "Proposed Units": "Number of proposed units",
        "Plansets": "Number of plan sets submitted",
        "TIDF Compliance": "Transit impact fee compliance flag",
        "Existing Construction Type": "Code for existing construction type",
        "Existing Construction Type Description": "Description of existing construction type",
        "Proposed Construction Type": "Code for proposed construction type",
        "Proposed Construction Type Description": "Description of proposed construction type",
        "Site Permit": "Flag indicating site permit",
        "Supervisor District": "Supervisor district number",
        "Neighborhoods - Analysis Boundaries": "Neighborhood name used for analysis",
        "Zipcode": "ZIP code of the property",
        "Location": "Latitude/longitude coordinates string",
        "Record ID": "Internal record identifier",
    }

    # render descriptions only for columns that exist
    cols = st.columns(2)
    i = 0
    for name, desc in column_descriptions.items():
        if name in df.columns:
            with cols[i % 2]:
                st.markdown(f"<div class='col-desc'><strong>{name}</strong><br>{desc}</div>", unsafe_allow_html=True)
            i += 1


# ---------- Data preview
st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
all_cols = df.columns.tolist()
default_preview = [c for c in ["Permit Number", "Permit Type", "Current Status", "Street Name", "Neighborhoods - Analysis Boundaries", "Estimated Cost"] if c in all_cols]
preview_default = default_preview if default_preview else all_cols[:6]

preview_cols = st.multiselect("Select columns to preview:", options=all_cols, default=preview_default)
if preview_cols:
    st.dataframe(df[preview_cols].head(10), use_container_width=True)


# ---------- Quick statistics
st.markdown("<div class='section-header'>Statistics</div>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Numeric", "Categorical", "Missing"])

with tab1:
    num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if num:
        st.dataframe(df[num].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found")

with tab2:
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    if cat:
        sel = st.selectbox("Categorical column:", cat)
        vc = df[sel].value_counts().head(20)
        fig = px.bar(x=vc.index, y=vc.values, labels={"x": sel, "y": "count"}, title=f"Top values — {sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns found")

with tab3:
    miss = pd.DataFrame({"column": df.columns, "missing_count": df.isnull().sum().values})
    miss = miss[miss["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if not miss.empty:
        st.dataframe(miss, use_container_width=True)
        fig = px.bar(miss, x="column", y="missing_count", title="Missing values by column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found")


# ---------- Key insights
st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
cols = st.columns(2)
with cols[0]:
    if "Current Status" in df.columns:
        sc = df["Current Status"].value_counts().head(20)
        fig = px.pie(values=sc.values, names=sc.index, title="Permit status distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Current Status column not found")

with cols[1]:
    if "Neighborhoods - Analysis Boundaries" in df.columns:
        tn = df["Neighborhoods - Analysis Boundaries"].value_counts().head(10)
        fig = px.bar(x=tn.values, y=tn.index, orientation="h", labels={"x": "count", "y": "neighborhood"}, title="Top neighborhoods by permits")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Neighborhoods column not found")


# ---------- Footer
st.markdown("---")
st.markdown("Data loaded from Building_Permits.csv — explore columns and summaries above.")
