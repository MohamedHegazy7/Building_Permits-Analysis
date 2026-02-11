import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit.components.v1 as components

# Try to import ydata_profiling, but make it optional to prevent app crashes
try:
    from ydata_profiling import ProfileReport
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

# --- 1. Configure page settings ---
st.set_page_config(page_title="Data Exploration & Profiling", layout="wide")

# --- 2. Helper Functions ---

# Function to load data with caching for performance
@st.cache_data
def load_data():
    try:
        # Ensure the filename matches your actual file in the repository
        df = pd.read_csv('Building_Permits.csv')
        return df
    except Exception as e:
        st.error(f"Error: 'Building_Permits.csv' not found. Please check the file path. {e}")
        return pd.DataFrame()

# Function to render the ydata-profiling report
def st_profile_report(profile_report):
    try:
        report_html = profile_report.to_html()
        components.html(report_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Error rendering report: {e}")

# --- 3. Title and Introduction ---
st.title("🔬 Data Exploration & Profiling Guide")
st.markdown("---")

st.header("📚 What is Data Exploration & Profiling?")
st.markdown("""
**Data Exploration** is the first critical step in data analysis where we systematically examine and understand 
the characteristics, patterns, and quality of a dataset before diving into modeling.
""")

st.markdown("---")

# --- 4. Load the dataset ---
df = load_data()

# Stop execution if data didn't load
if df.empty:
    st.stop()

# --- 5. Display basic dataset information ---
st.header("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", f"{len(df):,}")
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
    st.metric("Overall Missing Values", f"{missing_pct:.2f}%")

# --- 6. Data Preview (Fixed TypeError: width='stretch') ---
st.subheader("📊 Data Preview (First 10 Rows)")
st.dataframe(df.head(10), use_container_width=True)

# --- 7. Data Types ---
st.subheader("🏷️ Column Data Types")
st.dataframe(df.dtypes.astype(str), use_container_width=True)

# --- 8. Statistical Summary ---
st.subheader("📈 Statistical Summary")
st.dataframe(df.describe().round(2), use_container_width=True)

st.markdown("---")

# --- 9. Missing Data Analysis ---
st.header("🔍 Missing Data Analysis")

missing_data = df.isna().sum().reset_index()
missing_data.columns = ['Column', 'Missing Count']
missing_data['Missing %'] = (missing_data['Missing Count'] / len(df) * 100).round(2)
missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if not missing_data.empty:
    fig_missing = px.bar(
        missing_data,
        x='Column',
        y='Missing %',
        title='Missing Values by Column (%)',
        color='Missing %',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_missing, use_container_width=True)
    st.dataframe(missing_data, use_container_width=True)
else:
    st.success("✅ No missing values detected!")

st.markdown("---")

# --- 10. Duplicate Analysis ---
st.header("🔂 Duplicate Analysis")
duplicate_count = df.duplicated().sum()
st.metric("Total Duplicate Rows", duplicate_count)

st.markdown("---")

# --- 11. Correlation Analysis (Numerical Columns Only) ---
st.header("📊 Correlation Matrix")

numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        title='Correlation Matrix (Numerical)',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No numerical columns available for correlation analysis.")

st.markdown("---")

# --- 12. Automated Profiling Report ---
st.header("📑 Automated Profiling Report")

if HAS_PROFILING:
    st.info("Note: Generating the report may take a few minutes for large datasets.")
    if st.button("🚀 Generate Full Profile Report"):
        with st.spinner("Processing data and generating report..."):
            try:
                # Using minimal=True is recommended for large datasets to avoid memory issues
                profile = ProfileReport(df, explorative=True, minimal=True)
                st_profile_report(profile)
                st.success("✅ Profile report generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating profile report: {e}")
else:
    st.warning("⚠️ 'ydata-profiling' is not installed. Add it to your requirements.txt to enable this feature.")

st.markdown("---")
st.info("📌 **Tip:** Use this overview to identify data quality issues before proceeding to cleaning and modeling.")