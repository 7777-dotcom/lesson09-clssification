import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/7777-dotcom/data/refs/heads/main/stroke.csv"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 보지 않고 많은 쪽으로만 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
MAX_ROWS = 20

st.set_page_config(page_title="채점표 - 뇌졸중 예측 실습실", page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def confusion_counts(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 TP, FP, FN, TN 사람 수를 센다."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def confusion_table(tp, fp, fn, tn):
    """행은 실제, 열은 예측인 2×2 표에 행·열 합계를 붙인다."""
    return pd.DataFrame(
        [
            [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
            [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fp + fn + tn:,}명"],
        ],
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
        columns=["예측: 뇌졸중", "예측: 아님", "합계"],
    )


def four_metrics(tp, fp, fn, tn):
    """정확도, 재현율, 정밀도, F1을 구한다. 분모가 0이면 None."""
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def pct(value):
    return NOT_AVAILABLE if value is None else f"{value * 100:.2f}%"


def person_list(rows):
    """번호, 나이, 평균 혈당, 고혈압 여부, 심장병 여부를 나이 많은 순서로 정리한다."""
    rows = rows.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": rows["id"].to_numpy(),
            "나이": rows["age"].to_numpy(),
            "평균 혈당": rows["avg_glucose_level"].to_numpy(),
            "고혈압 여부": rows["hypertension"].map({1: "있음", 0: "없음"}).to_numpy(),
            "심장병 여부": rows["heart_disease"].map({1: "있음", 0: "없음"}).to_numpy(),
        }
    )


def show_person_list(title, caption, rows):
    total = len(rows)
    suffix = f"전체 {total:,}명" + (f", 앞 {MAX_ROWS}줄만 보여요" if total > MAX_ROWS else "")
    st.subheader(f"{title} ({suffix})")
    st.caption(caption)
    if total == 0:
        st.write("해당하는 사람이 없어요.")
    else:
        st.dataframe(person_list(rows).head(MAX_ROWS), hide_index=True, use_container_width=True)


st.title("📋 채점표")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 1. '분류 모델' 페이지와 똑같이: 입력 속성 고르기
selected = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOR[c],
)
features = [f for f in FEATURES if f in selected]
if len(features) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요.")
    st.stop()

# 2. 번호 순으로 정렬해 열 명씩 묶고, 묶음의 앞 세 명을 테스트용으로 고정
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3
x_train, y_train = data.loc[~is_test, features].copy(), data.loc[~is_test, "stroke"]
x_test, y_test = data.loc[is_test, features].copy(), data.loc[is_test, "stroke"]
test_rows = data.loc[is_test]  # 목록에 보여 줄 원래 값 (x_test와 같은 순서)

# 3. 체질량지수를 골랐을 때만 훈련용 중앙값으로 빈 값 채우기
if "bmi" in features:
    bmi_median = x_train["bmi"].median()
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)

# 4. 크기 맞추기(훈련용으로만)와 모델 학습
scaler = StandardScaler().fit(x_train)
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(x_train), y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(x_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

pred = {
    LR_NAME: lr.predict(scaler.transform(x_test)),
    TREE_NAME: tree.predict(x_test),
    BASE_NAME: base.predict(x_test),
}
counts = {name: confusion_counts(y_test, p) for name, p in pred.items()}

n_test = len(y_test)
n_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_stroke:,}명이에요.")

# 5. 혼동행렬
st.header("혼동행렬 (행: 실제, 열: 예측)")
left, right = st.columns(2)
for col, name in zip([left, right], [LR_NAME, TREE_NAME]):
    with col:
        st.subheader(name)
        st.table(confusion_table(*counts[name]))
st.caption(
    "뇌졸중을 양성으로 두었어요. TP: 뇌졸중인 사람을 뇌졸중이라고 맞힘, "
    "FP: 아닌 사람을 뇌졸중이라고 잘못 예측, FN: 뇌졸중인 사람을 아니라고 놓침, "
    "TN: 아닌 사람을 아니라고 맞힘."
)

# 6. 네 지표를 한 표에
st.header("네 가지 지표 비교")
rows = []
for name in [LR_NAME, TREE_NAME, BASE_NAME]:
    accuracy, recall, precision, f1 = four_metrics(*counts[name])
    rows.append(
        {"모델": name, "정확도": pct(accuracy), "재현율": pct(recall),
         "정밀도": pct(precision), "F1": pct(f1)}
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체, 재현율 = TP ÷ (TP+FN)(실제 뇌졸중인 사람 수), "
    "정밀도 = TP ÷ (TP+FP)(뇌졸중이라 예측한 사람 수), "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# 7. 놓친 환자와 헛짚은 사람
st.header("놓친 환자와 헛짚은 사람")
chosen = st.selectbox("목록을 볼 모델", [LR_NAME, TREE_NAME])
actual = y_test.to_numpy()
predicted = pred[chosen]
missed = test_rows.loc[(actual == 1) & (predicted == 0)]
wrong = test_rows.loc[(actual == 0) & (predicted == 1)]

col_left, col_right = st.columns(2)
with col_left:
    show_person_list("놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", missed)
with col_right:
    show_person_list("헛짚은 사람", "아닌데 뇌졸중이라고 예측한 사람", wrong)

tp, fp, fn, tn = counts[chosen]
same = len(missed) == fn and len(wrong) == fp
st.write(
    f"두 목록의 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong):,}명)는 "
    f"혼동행렬의 FN 칸({fn:,}명)·FP 칸({fp:,}명)과 "
    + ("같아요." if same else "달라요. 계산을 다시 확인해 주세요.")
)
