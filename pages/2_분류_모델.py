import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 보지 않고 많은 쪽으로만 답하는 모델)"

NEG_PALE, POS_PALE = "#dbe9f6", "#f8d7d7"  # 칸 색 (아님 / 뇌졸중)
NEG_COLOR, POS_COLOR = "#1f77b4", "#d62728"  # 점 색 (아님 / 뇌졸중)
FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif"

st.set_page_config(page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def acc_card(name, train_acc, test_acc):
    return f"""
    <div style="background:#f0f4fa;border-radius:12px;padding:16px 12px;text-align:center;">
        <div style="font-size:14px;color:#555;min-height:42px;">{name}</div>
        <div style="font-size:40px;font-weight:700;color:#1f3a68;">{test_acc * 100:.2f}%</div>
        <div style="font-size:12px;color:#666;">테스트 데이터 정확도</div>
        <div style="display:flex;justify-content:space-around;margin-top:10px;font-size:13px;color:#444;">
            <div>훈련 데이터<br><b>{train_acc * 100:.2f}%</b></div>
            <div>테스트 데이터<br><b>{test_acc * 100:.2f}%</b></div>
        </div>
    </div>
    """


def fixed_value_text(col, value):
    if col in BINARY and value in (0, 1):
        return f"{KOR[col]} {int(value)}({'있음' if value == 1 else '없음'})"
    return f"{KOR[col]} {value:g}"


def axis_range(series, col):
    lo, hi = float(series.min()), float(series.max())
    pad = 0.25 if col in BINARY else ((hi - lo) * 0.05 or 1.0)
    return lo - pad, hi + pad


def build_dot(tree, x_train, y_train, features):
    """의사결정트리를 graphviz DOT 문자열로 만든다."""
    t = tree.tree_
    path = tree.decision_path(x_train)
    n_node = np.asarray(path.sum(axis=0)).ravel().astype(int)
    pos_node = np.asarray(path.T.dot(y_train.to_numpy())).ravel().astype(int)

    lines = [
        "digraph Tree {",
        f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=12, '
        'width=2.2, height=0.8, margin="0.15,0.1"];',
        f'edge [fontname="{FONT}", fontsize=12];',
    ]
    for node in range(t.node_count):
        n, p = n_node[node], pos_node[node]
        counts = f"훈련용 {n:,}명\\n뇌졸중 {p:,}명 ({p / n * 100:.1f}%)"
        left, right = t.children_left[node], t.children_right[node]
        if left == right:  # 더 묻지 않고 답을 내는 마디
            answer = int(np.argmax(t.value[node, 0]))
            title = "답: 뇌졸중" if answer == 1 else "답: 아님"
            fill = POS_PALE if answer == 1 else NEG_PALE
            lines.append(f'n{node} [label="{title}\\n{counts}", fillcolor="{fill}"];')
        else:
            col = features[t.feature[node]]
            if col in BINARY:
                title = f"{KOR[col]}이 있나요?"
                yes_child, no_child = right, left
            else:
                title = f"{KOR[col]} ≤ {t.threshold[node]:.1f}?"
                yes_child, no_child = left, right
            lines.append(f'n{node} [label="{title}\\n{counts}", fillcolor="#f3f3f3"];')
            lines.append(f'n{node} -> n{yes_child} [label="예"];')
            lines.append(f'n{node} -> n{no_child} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines)


st.title("🤖 분류 모델")
st.write("뇌졸중(stroke=1)을 **양성**으로 두고, 두 가지 모델이 뇌졸중인지 아닌지 맞히게 해 봐요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 1. 입력 속성 고르기
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

# 2. 훈련용·테스트용 나누기: 번호 순으로 정렬해 열 명씩 묶고, 묶음의 앞 세 명을 테스트용으로 고정
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3
x_train, y_train = data.loc[~is_test, features].copy(), data.loc[~is_test, "stroke"]
x_test, y_test = data.loc[is_test, features].copy(), data.loc[is_test, "stroke"]
st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"훈련용 {len(x_train):,}명, 테스트용 {len(x_test):,}명이에요."
)

# 3. 체질량지수를 골랐을 때만 훈련용 중앙값으로 빈 값 채우기
if "bmi" in features:
    bmi_median = x_train["bmi"].median()
    n_fill_train = int(x_train["bmi"].isna().sum())
    n_fill_test = int(x_test["bmi"].isna().sum())
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수가 빈 사람(훈련용 {n_fill_train}명, 테스트용 {n_fill_test}명)은 "
        f"훈련용의 중앙값 {bmi_median:.1f}로 채웠어요."
    )

# 4. 크기 맞추기(훈련용으로만 학습)와 모델 만들기
# 로지스틱 회귀는 크기를 맞춘 값으로, 의사결정트리는 질문을 읽기 쉽도록 원래 값 그대로 학습해요.
scaler = StandardScaler().fit(x_train)
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(x_train), y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(x_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

# 5. 정확도 카드
st.header("모델별 정확도")
cards = [
    (LR_NAME, accuracy_score(y_train, lr.predict(scaler.transform(x_train))),
     accuracy_score(y_test, lr.predict(scaler.transform(x_test)))),
    (TREE_NAME, accuracy_score(y_train, tree.predict(x_train)),
     accuracy_score(y_test, tree.predict(x_test))),
    (BASE_NAME, accuracy_score(y_train, base.predict(x_train)),
     accuracy_score(y_test, base.predict(x_test))),
]
for col, (name, tr, te) in zip(st.columns(3), cards):
    col.markdown(acc_card(name, tr, te), unsafe_allow_html=True)

# 6. 두 속성으로 그리는 산점도
st.header("테스트 데이터와 모델이 가르는 선")
c1, c2 = st.columns(2)
x_col = c1.selectbox("가로축", features, index=0, format_func=lambda c: KOR[c])
y_options = [f for f in features if f != x_col]
y_col = c2.selectbox("세로축", y_options, index=0, format_func=lambda c: KOR[c])

x_lo, x_hi = axis_range(x_test[x_col], x_col)
y_lo, y_hi = axis_range(x_test[y_col], y_col)
xs = np.linspace(x_lo, x_hi, 150)
ys = np.linspace(y_lo, y_hi, 150)
gx, gy = np.meshgrid(xs, ys)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다
others = [f for f in features if f not in (x_col, y_col)]
medians = {f: float(x_test[f].median()) for f in others}
grid = pd.DataFrame({f: np.full(gx.size, medians.get(f, 0.0)) for f in features})
grid[x_col] = gx.ravel()
grid[y_col] = gy.ravel()
grid = grid[features]

prob = lr.predict_proba(scaler.transform(grid))[:, 1].reshape(gx.shape)
tree_zone = tree.predict(grid).reshape(gx.shape)

fig = go.Figure()
# 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=tree_zone, zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        colorscale=[[0, NEG_PALE], [0.5, NEG_PALE], [0.5, POS_PALE], [1, POS_PALE]],
    )
)
for label, color in [("의사결정트리: '아님'이라고 답하는 칸", NEG_PALE),
                     ("의사결정트리: '뇌졸중'이라고 답하는 칸", POS_PALE)]:
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", name=label,
                   marker=dict(symbol="square", size=14, color=color))
    )

# 로지스틱 회귀가 확률 0.5로 가르는 선
line_in_picture = bool(prob.min() < 0.5 < prob.max())
if line_in_picture:
    fig.add_trace(
        go.Contour(
            x=xs, y=ys, z=prob, showscale=False, hoverinfo="skip",
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            colorscale=[[0, "black"], [1, "black"]],
            line=dict(width=3), showlegend=True, name="로지스틱 회귀: 확률 0.5로 가르는 선",
        )
    )

# 테스트 데이터 점 (아님을 먼저, 뇌졸중을 나중에 찍어 가려지지 않게)
rng = np.random.default_rng(0)


def plot_values(col):
    values = x_test[col].to_numpy(dtype=float)
    if col in BINARY:  # 0과 1뿐이라 겹치지 않게 살짝 흩어 찍는다
        values = values + rng.uniform(-0.1, 0.1, len(values))
    return values


px_all, py_all = plot_values(x_col), plot_values(y_col)
for value, name, color, size in [(0, "실제: 아님", NEG_COLOR, 5), (1, "실제: 뇌졸중", POS_COLOR, 8)]:
    mask = (y_test == value).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=px_all[mask], y=py_all[mask], mode="markers", name=name,
            marker=dict(color=color, size=size, opacity=0.6 if value == 0 else 0.95,
                        line=dict(width=0.5, color="white")),
        )
    )

fig.update_layout(
    xaxis=dict(title=KOR[x_col], range=[x_lo, x_hi]),
    yaxis=dict(title=KOR[y_col], range=[y_lo, y_hi]),
    legend=dict(orientation="h", yanchor="top", y=-0.18),
    height=560,
)
st.plotly_chart(fig, use_container_width=True)

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 값이 0과 1뿐이라, 점이 겹쳐 보이지 않게 살짝 흩어서 찍었어요.")
if others:
    fixed = ", ".join(fixed_value_text(f, medians[f]) for f in others)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}")
else:
    st.write("두 축 말고 사용한 속성이 없어서, 따로 세워 둔 값은 없어요.")

if not line_in_picture:
    where = "낮아서" if prob.max() <= 0.5 else "높아서"
    answer = "아님" if prob.max() <= 0.5 else "뇌졸중"
    st.write(
        f"로지스틱 회귀가 확률 0.5로 가르는 선은 이 그림 밖에 있어요. "
        f"그림 안에서 뇌졸중일 확률은 {prob.min():.2%}~{prob.max():.2%}로 모두 0.5보다 {where}, "
        f"그림 어디에서나 '{answer}'이라고 답해요."
    )

# 7. 의사결정트리가 던진 질문
st.header("의사결정트리가 던진 질문")
st.graphviz_chart(build_dot(tree, x_train, y_train, features), use_container_width=True)

t = tree.tree_
leaf_nodes = [n for n in range(t.node_count) if t.children_left[n] == t.children_right[n]]
n_no = sum(int(np.argmax(t.value[n, 0])) == 0 for n in leaf_nodes)
st.write(f"답을 내는 마디는 모두 {len(leaf_nodes)}칸이고, 그중 {n_no}칸이 '아님'이라고 답해요.")

used = [features[i] for i in sorted(set(t.feature[t.feature >= 0]))]
if used:
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것: {', '.join(KOR[c] for c in used)}")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것: 없음(아무것도 묻지 않고 바로 답을 내요)")
