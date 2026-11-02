import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/7777-dotcom/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_record():
    if RECORD_FILE.exists():
        try:
            data = json.loads(RECORD_FILE.read_text(encoding="utf-8"))
            return {
                "meanings": data.get("meanings", {}),
                "source": data.get("source", ""),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"meanings": {}, "source": ""}


def save_record(record):
    RECORD_DIR.mkdir(exist_ok=True)
    RECORD_FILE.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def describe_values(series):
    """열의 '값의 종류'를 짧은 글로 만든다."""
    values = series.dropna()
    unique = sorted(values.unique(), key=lambda v: str(v))
    if pd.api.types.is_numeric_dtype(values) and len(unique) > 10:
        return f"숫자 ({values.min():g} ~ {values.max():g})"
    return ", ".join(str(v) for v in unique)


def card(label, value):
    return f"""
    <div style="background:#f0f4fa;border-radius:12px;padding:20px 12px;text-align:center;">
        <div style="font-size:15px;color:#555;">{label}</div>
        <div style="font-size:40px;font-weight:700;color:#1f3a68;">{value}</div>
    </div>
    """


st.title("🧠 뇌졸중 예측 실습실")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

record = load_record()

# 1. 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.markdown(card("전체 사람 수", f"{total:,}명"), unsafe_allow_html=True)
c2.markdown(card("열 개수", f"{n_cols}개"), unsafe_allow_html=True)
c3.markdown(card("stroke가 1인 사람 수", f"{n_stroke:,}명"), unsafe_allow_html=True)
c4.markdown(card("stroke가 1인 비율", f"{ratio:.1f}%"), unsafe_allow_html=True)

# 2. 열 설명 표
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 더블클릭해서 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [record["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)
edited = st.data_editor(
    table,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="meaning_table",
)

new_meanings = {
    row["열 이름"]: (row["우리말 뜻"] or "") for _, row in edited.iterrows()
}

# 3. 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# 4. 데이터 출처
st.subheader("데이터 출처")
new_source = st.text_area(
    "교재에 나온 출처를 적어 주세요.",
    value=record["source"],
    height=100,
    key="source_text",
)

# 5. 바뀐 내용이 있으면 json으로 저장
new_record = {"meanings": new_meanings, "source": new_source}
if new_record != record:
    save_record(new_record)
