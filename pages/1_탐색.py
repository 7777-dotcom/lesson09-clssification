import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/7777-dotcom/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df, column, labels):
    """column 값(0/1)별 뇌졸중 비율(%)을 표로 만든다."""
    rate = df.groupby(column)["stroke"].mean().mul(100).reset_index()
    rate["구분"] = rate[column].map(labels)
    rate = rate.rename(columns={"stroke": "뇌졸중 비율(%)"})
    return rate[["구분", "뇌졸중 비율(%)"]]


st.title("🔍 탐색")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 1. 나이와 평균 혈당의 분포
st.header("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포", labels={"age": "나이"})
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)

# 2. 뇌졸중 여부에 따른 비교
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
group_df = df.copy()
group_df["뇌졸중 여부"] = group_df["stroke"].map({0: "겪지 않음", 1: "겪음"})
order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}

col1, col2 = st.columns(2)
with col1:
    fig = px.box(
        group_df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders=order,
        title="나이 비교",
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(
        group_df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders=order,
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    group_df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["겪지 않음", "겪음"])
    .round(2)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당 평균"})
    .reset_index()
)
st.subheader("두 그룹의 평균값")
st.dataframe(mean_table, hide_index=True, use_container_width=True)

# 3. 고혈압, 심장병과 뇌졸중 비율
st.header("3. 고혈압·심장병과 뇌졸중 비율")
col1, col2 = st.columns(2)
with col1:
    rate = stroke_rate_by(df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"})
    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        color="구분",
        text_auto=".1f",
        title="고혈압 여부별 뇌졸중 비율",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    rate = stroke_rate_by(df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"})
    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        color="구분",
        text_auto=".1f",
        title="심장병 여부별 뇌졸중 비율",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# 4. bmi가 빈 사람들
st.header("4. bmi가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [len(bmi_missing), len(df)],
        "뇌졸중 사람 수": [int(bmi_missing["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [
            round(bmi_missing["stroke"].mean() * 100, 2) if len(bmi_missing) else 0.0,
            round(df["stroke"].mean() * 100, 2),
        ],
    }
)
st.dataframe(bmi_table, hide_index=True, use_container_width=True)

# 5. 흡연 상태
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True, use_container_width=True)
