import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Bivariate Analysis", page_icon="🔎", layout="wide")


@st.cache_data
def load_data(path: str = "Building_Permits.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series: pd.Series, cat_threshold: int = 20) -> bool:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    try:
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) <= cat_threshold:
            return True
    except Exception:
        pass
    return False


def safe_dropna(subdf: pd.DataFrame, cols):
    return subdf.loc[subdf[list(cols)].notna().all(axis=1), :]


df = load_data()

st.title("Bivariate Analysis")

if df.empty:
    st.error("Could not load Building_Permits.csv. Place the file in the project folder.")
    st.stop()


all_columns = df.columns.tolist()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    x_col = st.selectbox("X-axis column", options=all_columns, index=0)
    y_col = st.selectbox("Y-axis column", options=all_columns, index=1 if len(all_columns) > 1 else 0)
    color_col = st.selectbox("Optional: color / group by", options=[None] + all_columns, index=0)
    sample_limit = st.number_input("Max points for scatter (0 = no sampling)", min_value=0, max_value=200000, value=5000, step=500)
    cat_threshold = st.number_input("Treat numeric as categorical if unique <=", min_value=1, max_value=1000, value=20, step=1)
    st.markdown("---")
    st.markdown("Advanced display")
    prefer_categorical = st.selectbox("If both categorical, default visualization", ["Heatmap", "Stacked Bar", "Grouped Bar"], index=0)
    show_counts = st.checkbox("Show counts on categorical bars", value=True)
    st.caption("For large cardinality, the app will fall back to grouped/stacked bars for performance.")


# Basic validation
if x_col == y_col:
    st.warning("X and Y are the same column — showing single-column summary instead.")

col_x = df[x_col]
col_y = df[y_col]

num_x = is_numeric(col_x)
num_y = is_numeric(col_y)
cat_x = is_categorical(col_x, cat_threshold)
cat_y = is_categorical(col_y, cat_threshold)

st.markdown(f"**X:** {x_col} — {'categorical' if cat_x else 'numeric'} | **Y:** {y_col} — {'categorical' if cat_y else 'numeric'}")

# Prepare plot dataframe and drop NA in selected columns
plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].copy()
plot_df = safe_dropna(plot_df, [x_col, y_col] + ([color_col] if color_col else []))

# Helper to limit categories (top-k + other)
def top_k_with_other(series: pd.Series, k: int = 20):
    vc = series.value_counts(dropna=True)
    top = vc.nlargest(k).index
    return series.where(series.isin(top), other='Other')


# Set common style options
color_seq = px.colors.qualitative.Plotly
template = 'plotly_white'


def render_scatter(df_plot: pd.DataFrame):
    n = len(df_plot)
    if sample_limit > 0 and n > sample_limit:
        df_plot = df_plot.sample(sample_limit, random_state=42)
    try:
        fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_col if color_col else None, labels={x_col: x_col, y_col: y_col, color_col: color_col} if color_col else {x_col: x_col, y_col: y_col}, color_discrete_sequence=color_seq, template=template, opacity=0.75)
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render scatter plot: {e}")


def render_density_heatmap(df_plot: pd.DataFrame):
    try:
        fig = px.density_heatmap(df_plot, x=x_col, y=y_col, nbinsx=40, nbinsy=40, labels={x_col: x_col, y_col: y_col}, color_continuous_scale='Viridis', template=template)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render density heatmap: {e}")


def render_line(df_plot: pd.DataFrame):
    try:
        # try to sort x if datetime-like
        try:
            df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors='coerce')
            df_plot = df_plot.dropna(subset=[x_col])
            df_plot = df_plot.sort_values(x_col)
        except Exception:
            pass
        agg = df_plot.groupby(x_col)[y_col].mean().reset_index()
        fig = px.line(agg, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, markers=True, template=template)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render line plot: {e}")


def render_box(df_plot: pd.DataFrame, cat_is_x: bool = True):
    try:
        if not cat_is_x:
            fig = px.box(df_plot, x=y_col, y=x_col, color=color_col if color_col else None, labels={x_col: x_col, y_col: y_col, color_col: color_col} if color_col else {x_col: x_col, y_col: y_col}, template=template)
        else:
            fig = px.box(df_plot, x=x_col, y=y_col, color=color_col if color_col else None, labels={x_col: x_col, y_col: y_col, color_col: color_col} if color_col else {x_col: x_col, y_col: y_col}, template=template)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render box plot: {e}")


def render_bar_agg(df_plot: pd.DataFrame, categorical_on_x: bool = True, agg_func: str = 'count'):
    try:
        if categorical_on_x:
            if agg_func == 'count':
                df_bar = df_plot.groupby(x_col).size().reset_index(name='count')
                fig = px.bar(df_bar, x=x_col, y='count', labels={x_col: x_col, 'count': 'count'}, template=template, color_discrete_sequence=color_seq)
            else:
                df_bar = df_plot.groupby(x_col)[y_col].agg(agg_func).reset_index()
                fig = px.bar(df_bar, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, template=template, color_discrete_sequence=color_seq)
        else:
            if agg_func == 'count':
                df_bar = df_plot.groupby(y_col).size().reset_index(name='count')
                fig = px.bar(df_bar, x='count', y=y_col, orientation='h', labels={y_col: y_col, 'count': 'count'}, template=template, color_discrete_sequence=color_seq)
            else:
                df_bar = df_plot.groupby(y_col)[x_col].agg(agg_func).reset_index()
                fig = px.bar(df_bar, x=x_col, y=y_col, labels={x_col: x_col, y_col: y_col}, template=template, color_discrete_sequence=color_seq)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to render bar chart: {e}")


def render_cat_vs_cat(df_plot: pd.DataFrame):
    ux = df_plot[x_col].nunique()
    uy = df_plot[y_col].nunique()
    # collapse to top categories if too many
    max_display = 40
    if ux > max_display:
        df_plot[x_col] = top_k_with_other(df_plot[x_col], k=max_display)
    if uy > max_display:
        df_plot[y_col] = top_k_with_other(df_plot[y_col], k=max_display)

    if prefer_categorical == 'Heatmap' and ux <= 60 and uy <= 60:
        try:
            ct = pd.crosstab(df_plot[y_col], df_plot[x_col])
            fig = px.imshow(ct, labels=dict(x=x_col, y=y_col, color='count'), text_auto=True if ct.size <= 400 else False, template=template, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Unable to render heatmap: {e}")
    else:
        barmode = 'stack' if prefer_categorical == 'Stacked Bar' else 'group'
        try:
            fig = px.histogram(df_plot, x=x_col, color=y_col, barmode=barmode, text_auto=show_counts, labels={x_col: x_col, y_col: y_col}, template=template, color_discrete_sequence=color_seq)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Unable to render categorical histogram: {e}")


# Main decision flow
if num_x and num_y:
    st.subheader("Numeric vs Numeric")
    plot_choice = st.selectbox("Plot type", ["Scatter", "Heatmap", "Line"], index=0)
    if plot_choice == 'Scatter':
        render_scatter(plot_df)
    elif plot_choice == 'Heatmap':
        render_density_heatmap(plot_df)
    else:
        render_line(plot_df)

elif (cat_x and num_y) or (num_x and cat_y):
    st.subheader("Numeric vs Categorical")
    chart_choice = st.selectbox("Chart type", ["Box", "Bar"], index=0)
    if chart_choice == 'Box':
        # ensure categorical on x
        render_box(plot_df, cat_is_x=cat_x)
    else:
        agg_func = st.selectbox("Aggregation", ['count', 'mean', 'median', 'sum'], index=0)
        render_bar_agg(plot_df, categorical_on_x=cat_x, agg_func=agg_func)

else:
    st.subheader("Categorical vs Categorical")
    render_cat_vs_cat(plot_df)

st.markdown('---')
st.caption('Tips: sampling helps for scatter plots. Use the color/group option for extra grouping.')
