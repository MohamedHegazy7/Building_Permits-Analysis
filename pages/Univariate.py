import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Univariate Analysis", page_icon="📊", layout="wide")


@st.cache_data
def load_data(path: str = "Building_Permits.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series: pd.Series, threshold: int = 20) -> bool:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) <= threshold:
            return True
    except Exception:
        pass
    return False


df = load_data()

st.title("Univariate Analysis")

if df.empty:
    st.error("Could not load Building_Permits.csv. Place the file in the project folder.")
    st.stop()


cols = df.columns.tolist()

with st.sidebar:
    st.header("Settings")
    column = st.selectbox("Select column", options=cols)
    cat_threshold = st.number_input("Treat numeric as categorical if unique <=", min_value=1, max_value=1000, value=20, step=1)
    sample_limit = st.number_input("Max sample points for histogram (0 = no sampling)", min_value=0, max_value=200000, value=0, step=500)
    st.markdown("---")
    st.caption("Numerical: histogram + box. Categorical: bar + pie.")


series = df[column]
is_num = is_numeric(series)
is_cat = is_categorical(series, threshold=cat_threshold)

st.markdown(f"**Column:** {column} — {'categorical' if is_cat else 'numeric' if is_num else str(series.dtype)}")

plot_df = df[[column]].copy()
plot_df = plot_df[plot_df[column].notna()]

if is_num and not is_cat:
    st.subheader("Numerical column: Histogram & Box plot")

    n = len(plot_df)
    if sample_limit > 0 and n > sample_limit:
        plot_df = plot_df.sample(sample_limit, random_state=42)

    # Histogram
    fig_h = px.histogram(plot_df, x=column, nbins=60, title=f"Histogram — {column}", labels={column: column}, template='plotly_white')
    fig_h.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_h, use_container_width=True)

    # Box plot
    fig_b = px.box(plot_df, y=column, points='outliers', title=f"Box plot — {column}", template='plotly_white')
    fig_b.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_b, use_container_width=True)

    # Stats
    st.subheader("Summary statistics")
    st.dataframe(plot_df.describe().T)

else:
    st.subheader("Categorical column: Bar chart & Pie chart")

    # collapse to top-k categories for display
    top_k = st.slider("Top K categories to display (others grouped as 'Other')", 3, 100, 25)
    vc = plot_df[column].value_counts(dropna=True)
    top = vc.nlargest(top_k).index
    plot_df['category_trunc'] = plot_df[column].where(plot_df[column].isin(top), other='Other')

    counts = plot_df['category_trunc'].value_counts().reset_index()
    counts.columns = [column, 'count']

    # Bar chart
    fig_bar = px.bar(counts, x=column, y='count', title=f"Bar chart — {column}", labels={column: column, 'count': 'count'}, template='plotly_white')
    fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=120))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart
    fig_pie = px.pie(counts, names=column, values='count', title=f"Pie chart — {column}", template='plotly_white')
    st.plotly_chart(fig_pie, use_container_width=True)

    # Show table of top categories
    with st.expander("Show counts table"):
        st.dataframe(counts)
