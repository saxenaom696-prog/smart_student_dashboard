import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# CONFIG
st.set_page_config(page_title="Student Dashboard", layout="wide")

# LOAD DATA
df = pd.read_csv("student_performance.csv")
df.columns = df.columns.str.lower()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# MODEL
X = numeric_df.iloc[:, :-1]
y = numeric_df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

# SIDEBAR
st.sidebar.title("Filters")
selected_col = st.sidebar.selectbox("Select Feature", numeric_df.columns)

# MAIN TITLE
st.title("Student Performance Dashboard")

# GAUGE FUNCTION
def gauge(title, value):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={'axis': {'range': [0, 100]}}
        )
    )

# TOP METRICS
st.subheader("Performance Overview")

cols = st.columns(4)
means = numeric_df.mean()

for i in range(min(4, len(means))):
    cols[i].plotly_chart(
        gauge(means.index[i], means[i]),
        use_container_width=True
    )

# CHART
st.subheader("Feature Analysis")
fig = px.histogram(df, x=selected_col)
st.plotly_chart(fig, use_container_width=True)

# BAR CHART
st.subheader("Average Scores")
fig_bar = px.bar(x=means.index, y=means.values, color=means.values)
st.plotly_chart(fig_bar, use_container_width=True)

# PREDICTION
st.subheader("Predict Student Score")

col1, col2 = st.columns(2)
inputs = []

for i, col in enumerate(X.columns):
    if i % 2 == 0:
        val = col1.number_input(f"{col}", value=0)
    else:
        val = col2.number_input(f"{col}", value=0)
    inputs.append(val)

if st.button("Predict"):
    pred = model.predict([inputs])
    st.success(f"Predicted Score: {round(pred[0], 2)}")

# DATA PREVIEW
st.subheader("Dataset Preview")
st.dataframe(df.head())