import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
NEG_COLOR, POS_COLOR = "#1f77b4", "#d62728"  # 점 색 (아님 / 뇌졸중)
BEFORE_COLOR, AFTER_COLOR = "#2ca02c", "#ff7f0e"  # 경계선·평면 색 (맞추기 전 / 맞춘 뒤)

st.set_page_config(page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 혼동행렬과 지표 ----------
def confusion_counts(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 TP, FP, FN, TN 사람 수를 센다."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


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


def before_after_table(before, after):
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    rows = [
        ("TP · 찾아낸 환자", f"{before[0]:,}명", f"{after[0]:,}명"),
        ("FN · 놓친 환자", f"{before[2]:,}명", f"{after[2]:,}명"),
        ("FP · 헛짚은 사람", f"{before[1]:,}명", f"{after[1]:,}명"),
        ("TN · 아닌 사람을 아니라고 맞힘", f"{before[3]:,}명", f"{after[3]:,}명"),
    ]
    for label, b, a in zip(["정확도", "재현율", "정밀도", "F1"], four_metrics(*before), four_metrics(*after)):
        rows.append((label, pct(b), pct(a)))
    return pd.DataFrame(
        [[b, a] for _, b, a in rows],
        index=[label for label, _, _ in rows],
        columns=["가중치 맞추기 전", "가중치 맞춘 뒤"],
    )


# ---------- 로지스틱 회귀의 0.5 경계 ----------
def linear_form(lr, scaler, features):
    """로짓을 '절편 + 계수×원래 값'으로 바꾼다. 로짓 = 0인 곳이 확률 0.5인 곳이다."""
    coef = lr.coef_[0] / scaler.scale_
    intercept = float(lr.intercept_[0] - np.sum(coef * scaler.mean_))
    return dict(zip(features, (float(c) for c in coef))), intercept


def logit_range(const, terms):
    """상자 안에서 로짓이 가질 수 있는 가장 작은 값과 큰 값. terms: [(계수, (최소, 최대)), ...]"""
    low = const + sum(min(a * lo, a * hi) for a, (lo, hi) in terms)
    high = const + sum(max(a * lo, a * hi) for a, (lo, hi) in terms)
    return low, high


def outside_message(label, shape_word, place, low, high):
    """경계가 그림 안을 지나가면 None, 밖에 있으면 그 사실을 알리는 글."""
    if low < 0 < high:
        return None
    p_low, p_high = 1 / (1 + np.exp(-low)), 1 / (1 + np.exp(-high))
    if high <= 0:
        side, answer = "낮아서", "아님"
    else:
        side, answer = "높아서", "뇌졸중"
    return (
        f"{label}의 0.5 {shape_word}은 {place} 밖에 있어요. 그림 안에서 뇌졸중일 확률은 "
        f"{p_low:.2%}~{p_high:.2%}로 모두 0.5보다 {side}, 그림 어디에서나 '{answer}'이라고 답해요."
    )


def boundary_line(ax, ay, const, x_rng, y_rng, n=600):
    """ax·x + ay·y + const = 0 인 직선 가운데 그림 안에 들어오는 부분."""
    if abs(ay) >= abs(ax) and abs(ay) > 1e-12:
        xs = np.linspace(x_rng[0], x_rng[1], n)
        ys = -(const + ax * xs) / ay
        ys = np.where((ys >= y_rng[0]) & (ys <= y_rng[1]), ys, np.nan)
    elif abs(ax) > 1e-12:
        ys = np.linspace(y_rng[0], y_rng[1], n)
        xs = -(const + ay * ys) / ax
        xs = np.where((xs >= x_rng[0]) & (xs <= x_rng[1]), xs, np.nan)
    else:
        return None
    return xs, ys


def boundary_plane(coefs, const, ranges, n=40):
    """로짓 = 0 인 평면 가운데 상자 안에 들어오는 부분 (x, y, z 격자)."""
    k = int(np.argmax(np.abs(coefs)))  # 계수가 가장 큰 축을 나머지 두 축으로 계산한다
    if abs(coefs[k]) < 1e-12:
        return None
    i, j = [m for m in range(3) if m != k]
    u = np.linspace(ranges[i][0], ranges[i][1], n)
    v = np.linspace(ranges[j][0], ranges[j][1], n)
    grid_u, grid_v = np.meshgrid(u, v)
    w = -(const + coefs[i] * grid_u + coefs[j] * grid_v) / coefs[k]
    w = np.where((w >= ranges[k][0]) & (w <= ranges[k][1]), w, np.nan)
    pts = [None, None, None]
    pts[i], pts[j], pts[k] = grid_u, grid_v, w
    return pts


# ---------- 그림 도우미 ----------
def axis_range(series, col):
    lo, hi = float(series.min()), float(series.max())
    pad = 0.25 if col in BINARY else ((hi - lo) * 0.05 or 1.0)
    return lo - pad, hi + pad


def fixed_value_text(col, value):
    if col in BINARY and value in (0, 1):
        return f"{KOR[col]} {int(value)}({'있음' if value == 1 else '없음'})"
    return f"{KOR[col]} {value:g}"


jitter_rng = np.random.default_rng(0)


def plot_values(series, col):
    values = series.to_numpy(dtype=float)
    if col in BINARY:  # 0과 1뿐이라 겹치지 않게 살짝 흩어 찍는다
        values = values + jitter_rng.uniform(-0.1, 0.1, len(values))
    return values


st.title("⚖️ 성능 높이기")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 1. 입력 속성 고르기 ('분류 모델' 페이지와 똑같이)
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

# 3. 체질량지수를 골랐을 때만 훈련용 중앙값으로 빈 값 채우기
if "bmi" in features:
    bmi_median = x_train["bmi"].median()
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)
    st.caption(f"체질량지수가 빈 사람은 훈련용의 중앙값 {bmi_median:.1f}로 채웠어요.")

# 4. 크기 맞추기(훈련용으로만)와 모델 학습: 가중치를 맞추기 전과 맞춘 뒤
scaler = StandardScaler().fit(x_train)
x_train_scaled, x_test_scaled = scaler.transform(x_train), scaler.transform(x_test)

lr_before = LogisticRegression(max_iter=1000).fit(x_train_scaled, y_train)
lr_after = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train_scaled, y_train)
tree_params = dict(max_depth=3, min_samples_leaf=5, random_state=42)
tree_before = DecisionTreeClassifier(**tree_params).fit(x_train, y_train)
tree_after = DecisionTreeClassifier(class_weight="balanced", **tree_params).fit(x_train, y_train)

n_test, n_stroke = len(y_test), int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_stroke:,}명이에요.")
st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어서, 훈련할 때 뇌졸중인 사람과 아닌 사람의 가중치를 "
    "같은 힘이 되도록 맞춘 모델(class_weight='balanced')과 맞추지 않은 모델을 비교해요."
)

# 5. 전·후 표
st.header("가중치를 맞추기 전과 맞춘 뒤")
results = {
    LR_NAME: (
        confusion_counts(y_test, lr_before.predict(x_test_scaled)),
        confusion_counts(y_test, lr_after.predict(x_test_scaled)),
    ),
    TREE_NAME: (
        confusion_counts(y_test, tree_before.predict(x_test)),
        confusion_counts(y_test, tree_after.predict(x_test)),
    ),
}
left, right = st.columns(2)
for col, name in zip([left, right], [LR_NAME, TREE_NAME]):
    before, after = results[name]
    with col:
        st.subheader(name)
        st.table(before_after_table(before, after))
        st.write(
            f"찾아낸 환자 {before[0]:,}명 → {after[0]:,}명, "
            f"놓친 환자 {before[2]:,}명 → {after[2]:,}명, "
            f"헛짚은 사람 {before[1]:,}명 → {after[1]:,}명"
        )

form_before, intercept_before = linear_form(lr_before, scaler, features)
form_after, intercept_after = linear_form(lr_after, scaler, features)

# 6. 두 속성으로 그리는 산점도와 0.5 경계선
st.header("2차원: 0.5 경계선이 어떻게 움직였을까")
st.caption(f"{LR_NAME}이 뇌졸중일 확률을 0.5로 가르는 선을 맞추기 전과 맞춘 뒤로 겹쳐 그려요.")
c1, c2 = st.columns(2)
x_col = c1.selectbox("가로축", features, index=0, format_func=lambda c: KOR[c])
y_options = [f for f in features if f != x_col]
y_col = c2.selectbox("세로축", y_options, index=0, format_func=lambda c: KOR[c])

x_rng = axis_range(x_test[x_col], x_col)
y_rng = axis_range(x_test[y_col], y_col)
others2 = [f for f in features if f not in (x_col, y_col)]
medians2 = {f: float(x_test[f].median()) for f in others2}  # 두 축이 아닌 속성은 중앙값에 세운다

fig2 = go.Figure()
for value, name, color, size in [(0, "실제: 아님", NEG_COLOR, 5), (1, "실제: 뇌졸중", POS_COLOR, 8)]:
    mask = (y_test == value).to_numpy()
    fig2.add_trace(
        go.Scatter(
            x=plot_values(x_test[x_col], x_col)[mask],
            y=plot_values(x_test[y_col], y_col)[mask],
            mode="markers", name=name,
            marker=dict(color=color, size=size, opacity=0.6 if value == 0 else 0.95,
                        line=dict(width=0.5, color="white")),
        )
    )

messages2 = []
for label, form, intercept, color, dash in [
    ("가중치를 맞추기 전", form_before, intercept_before, BEFORE_COLOR, "dash"),
    ("가중치를 맞춘 뒤", form_after, intercept_after, AFTER_COLOR, "solid"),
]:
    const = intercept + sum(form[f] * medians2[f] for f in others2)
    low, high = logit_range(const, [(form[x_col], x_rng), (form[y_col], y_rng)])
    message = outside_message(label, "경계선", "그림", low, high)
    if message:
        messages2.append(message)
        continue
    line = boundary_line(form[x_col], form[y_col], const, x_rng, y_rng)
    if line is not None:
        fig2.add_trace(
            go.Scatter(x=line[0], y=line[1], mode="lines", name=f"0.5 경계선({label})",
                       line=dict(color=color, width=3, dash=dash))
        )

fig2.update_layout(
    xaxis=dict(title=KOR[x_col], range=list(x_rng)),
    yaxis=dict(title=KOR[y_col], range=list(y_rng)),
    legend=dict(orientation="h", yanchor="top", y=-0.18),
    height=560,
)
st.plotly_chart(fig2, use_container_width=True)

if x_col in BINARY or y_col in BINARY:
    st.caption("고혈압·심장병은 값이 0과 1뿐이라, 점이 겹쳐 보이지 않게 살짝 흩어서 찍었어요.")
if others2:
    fixed = ", ".join(fixed_value_text(f, medians2[f]) for f in others2)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}")
else:
    st.write("두 축 말고 사용한 속성이 없어서, 따로 세워 둔 값은 없어요.")
for message in messages2:
    st.write(message)

# 7. 속성을 셋 이상 골랐을 때: 3차원 산점도와 0.5 평면
if len(features) >= 3:
    st.header("3차원: 0.5 평면이 어떻게 움직였을까")
    st.caption("마우스로 끌어서 돌려 보고, 휠로 확대·축소할 수 있어요.")
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("가로축(X)", features, index=0, format_func=lambda c: KOR[c])
    y3 = d2.selectbox("세로축(Y)", [f for f in features if f != x3], index=0, format_func=lambda c: KOR[c])
    z3 = d3.selectbox(
        "깊이축(Z)", [f for f in features if f not in (x3, y3)], index=0, format_func=lambda c: KOR[c]
    )
    axes3 = [x3, y3, z3]
    ranges3 = [axis_range(x_test[c], c) for c in axes3]
    others3 = [f for f in features if f not in axes3]
    medians3 = {f: float(x_test[f].median()) for f in others3}

    fig3 = go.Figure()
    for value, name, color, size in [(0, "실제: 아님", NEG_COLOR, 2.5), (1, "실제: 뇌졸중", POS_COLOR, 5)]:
        mask = (y_test == value).to_numpy()
        fig3.add_trace(
            go.Scatter3d(
                x=plot_values(x_test[x3], x3)[mask],
                y=plot_values(x_test[y3], y3)[mask],
                z=plot_values(x_test[z3], z3)[mask],
                mode="markers", name=name,
                marker=dict(color=color, size=size, opacity=0.6 if value == 0 else 0.95),
            )
        )

    messages3 = []
    for label, form, intercept, color in [
        ("가중치를 맞추기 전", form_before, intercept_before, BEFORE_COLOR),
        ("가중치를 맞춘 뒤", form_after, intercept_after, AFTER_COLOR),
    ]:
        const = intercept + sum(form[f] * medians3[f] for f in others3)
        coefs = [form[c] for c in axes3]
        low, high = logit_range(const, list(zip(coefs, ranges3)))
        message = outside_message(label, "평면", "그림 상자", low, high)
        if message:
            messages3.append(message)
            continue
        plane = boundary_plane(coefs, const, ranges3)
        if plane is not None:
            fig3.add_trace(
                go.Surface(
                    x=plane[0], y=plane[1], z=plane[2], opacity=0.5, showscale=False,
                    colorscale=[[0, color], [1, color]], hoverinfo="skip",
                    showlegend=True, name=f"0.5 평면({label})",
                )
            )

    fig3.update_layout(
        scene=dict(
            xaxis=dict(title=KOR[x3], range=list(ranges3[0])),
            yaxis=dict(title=KOR[y3], range=list(ranges3[1])),
            zaxis=dict(title=KOR[z3], range=list(ranges3[2])),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.02),
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("초록색 면은 가중치를 맞추기 전, 주황색 면은 맞춘 뒤의 0.5 평면이에요.")
    if any(c in BINARY for c in axes3):
        st.caption("고혈압·심장병은 값이 0과 1뿐이라, 점이 겹쳐 보이지 않게 살짝 흩어서 찍었어요.")
    if others3:
        fixed = ", ".join(fixed_value_text(f, medians3[f]) for f in others3)
        st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}")
    else:
        st.write("세 축 말고 사용한 속성이 없어서, 따로 세워 둔 값은 없어요.")
    for message in messages3:
        st.write(message)
else:
    st.caption("속성을 셋 이상 고르면 3차원 그림도 볼 수 있어요.")
