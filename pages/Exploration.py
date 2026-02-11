import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# --- 1. Configure page settings ---
st.set_page_config(page_title="Data Exploration & Profiling", layout="wide")

# --- 2. Helper Functions ---

# Function to load data
@st.cache_data
def load_data():
    try:
        # Update this filename to match your actual file
        df = pd.read_csv('Building_Permits.csv')
        return df
    except FileNotFoundError:
        st.error("File 'Building_Permits.csv' not found. Please upload it or check the path.")
        return pd.DataFrame()

# Function to render the ydata-profiling report in Streamlit
def st_profile_report(profile_report):
    try:
        # Convert the report to HTML
        report_html = profile_report.to_html()
        # Render the HTML using Streamlit components
        components.html(report_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Error generating report: {e}")

# --- 3. Title and Introduction ---
st.title("🔬 Data Exploration & Profiling Guide")
st.markdown("---")

st.header("📚 What is Data Exploration & Profiling?")
st.markdown("""
**Data Exploration** is the first critical step in data analysis where we systematically examine and understand 
the characteristics, patterns, and quality of a dataset.
""")

st.markdown("---")

# --- 4. Load the dataset ---
df = load_data()

# Stop execution if data didn't load
if df.empty:
    st.stop()

st.header("🎯 Key Aspects of Data Exploration")

# --- 5. Create Tabs (Added Tab 7 for Automated Profiling) ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dataset Info",
    "Univariate Analysis",
    "Missing Data",
    "Duplicates",
    "Categorical Analysis",
    "Numerical Analysis",
    "Automated Profiling"  # <--- New Tab
])

# --- TAB 1: Dataset Information ---
with tab1:
    st.subheader("1️⃣ Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.write("**Data Types Distribution:**")
    dtype_counts = df.dtypes.value_counts()
    
    # FIX: Convert index to string for Plotly
    fig = px.bar(x=dtype_counts.index.astype(str), 
                 y=dtype_counts.values, 
                 labels={'x': 'Data Type', 'y': 'Count'},
                 title='Data Types in Dataset')
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 View Detailed Dataset Info"):
        info_data = []
        for col in df.columns:
            info_data.append({
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].notna().sum(),
                'Null Count': df[col].isnull().sum(),
                'Unique Values': df[col].nunique()
            })
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True, height=600)

# --- TAB 2: Univariate Analysis ---
with tab2:
    st.subheader("2️⃣ Univariate Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select a numeric column:", numeric_cols)
        
        if selected_col:
            col_data = df[selected_col].dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Mean", f"{col_data.mean():.2f}")
            with col2: st.metric("Median", f"{col_data.median():.2f}")
            with col3: st.metric("Std Dev", f"{col_data.std():.2f}")
            with col4: st.metric("Range", f"{col_data.max() - col_data.min():.2f}")
            
            fig = px.histogram(df, x=selected_col, nbins=50, 
                              title=f"Distribution of {selected_col}",
                              marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found.")

# --- TAB 3: Missing Data Analysis ---
with tab3:
    st.subheader("3️⃣ Missing Data Analysis")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing %', ascending=False)
    
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    if len(missing_data) > 0:
        fig = px.bar(missing_data, x='Column', y='Missing %',
                    title='Missing Data Percentage by Column',
                    color='Missing %',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_data, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")

# --- TAB 4: Duplicate Records Analysis ---
with tab4:
    st.subheader("4️⃣ Duplicate Records Analysis")
    
    complete_dupes = df.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1: st.metric("Complete Duplicates", complete_dupes)
    with col2: st.metric("Duplicate Percentage", f"{(complete_dupes/len(df)*100):.2f}%")
    
    st.write("**Duplicates in Key Columns:**")
    key_columns = ['Permit Number', 'Record ID'] # Ensure these match your CSV
    
    duplicate_analysis = []
    for col in key_columns:
        if col in df.columns:
            dup_count = df.duplicated(subset=[col]).sum()
            duplicate_analysis.append({
                'Column': col,
                'Duplicate Count': dup_count,
                'Unique Values': df[col].nunique()
            })
    
    if duplicate_analysis:
        st.dataframe(pd.DataFrame(duplicate_analysis), use_container_width=True)

# --- TAB 5: Categorical Data Analysis ---
with tab5:
    st.subheader("5️⃣ Categorical Data Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        selected_cat = st.selectbox("Select a categorical column:", categorical_cols)
        
        if selected_cat:
            value_counts = df[selected_cat].value_counts().head(15)
            st.metric("Unique Values", df[selected_cat].nunique())
            
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Top 15 Categories in {selected_cat}",
                        labels={'x': selected_cat, 'y': 'Frequency'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No categorical columns found.")

# --- TAB 6: Numerical Data Analysis ---
with tab6:
    st.subheader("6️⃣ Numerical Data Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_num_cols = st.multiselect(
            "Select numeric columns for correlation analysis:",
            numeric_cols,
            default=numeric_cols[:5]
        )
        
        if selected_num_cols:
            corr_matrix = df[selected_num_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0, zmin=-1, zmax=1
            ))
            fig.update_layout(title='Correlation Matrix', height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df[selected_num_cols].describe().T, use_container_width=True)
    else:
        st.warning("No numeric columns found.")

# --- TAB 7: Automated Profiling (YData Profiling) ---
with tab7:
    st.subheader("7️⃣ Automated Profiling Report")
    st.markdown("""
    This section generates a complete HTML report using **ydata-profiling**.
    *Note: This process might take a few moments for large datasets.*
    """)
    
    # We use a button to prevent the report from generating automatically on page load
    if st.button("Generate Profile Report"):
        with st.spinner("Generating Report... Please wait."):
            try:
                # Configuration to fix the "ValueError" (Disable continuous interactions)
                # 'explorative=True' enables advanced features
                pr = ProfileReport(df, 
                                   title="Building Permits Profiling Report",
                                   explorative=True,
                                   interactions={'continuous': False}) # <--- CRITICAL FIX
                
                # Display the report
                st_profile_report(pr)
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("*Last updated: February 11, 2026*")