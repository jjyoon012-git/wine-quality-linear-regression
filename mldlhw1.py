import streamlit as st
import pandas as pd
import numpy as np
import time  # 학습 시간을 측정하여 계산 효율성 비교
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Dataset을 활용한 Direct Solution과 Gradient Descent 비교", layout="wide")

TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
RANDOM_STATE = 42
EPOCHS = 1000
STANDARDIZE = True
RIDGE_LAMBDA = 0.1  # Ridge Regularization (고정)

# bias 추가
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

# direct solution 함수 (Ridge Regularization 사용)
def direct_solution_ridge(X: np.ndarray, y: np.ndarray, ridge_lambda: float = RIDGE_LAMBDA):
    Xb = add_bias(X)
    d = Xb.shape[1]
    I = np.eye(d)
    I[0, 0] = 0.0
    A = Xb.T @ Xb + ridge_lambda * I
    b = Xb.T @ y.reshape(-1, 1)
    w = np.linalg.pinv(A) @ b
    return w.ravel()

def gd_train(X: np.ndarray, y: np.ndarray, lr: float, epochs: int = EPOCHS):
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    curve = []
    for ep in range(epochs):
        y_pred = Xb @ w
        residual = y - y_pred
        grad = -(2.0 / n) * (Xb.T @ residual)
        w -= lr * grad
        if ep % 10 == 0:
            curve.append(mean_squared_error(y, y_pred))
    return w, curve

# 사이드바를 활용해서 Learning Rate, Feature Map 차수 결정
st.sidebar.header("사이드바를 활용한 값 설정")
lr = st.sidebar.slider("Learning rate (Gradient Descent)", 0.001, 0.1, 0.01, step=0.001)
degree = st.sidebar.slider("Feature mapping 차수", 1, 5, 1, step=1)

# wine quality dataset
@st.cache_data
def load_data():
    red = pd.read_csv("winequality-red.csv", sep=";")
    red["wine_type"] = "red"
    try:
        white = pd.read_csv("winequality-white.csv", sep=";")
        white["wine_type"] = "white"
        df = pd.concat([red, white], ignore_index=True)
    except Exception:
        df = red
    return df

st.title(" Wine Quality - Direct Solution vs Gradient Descent")
st.caption("Train:Test = 8:2, 표준화 적용. Sidebar에서 Learning rate와 Feature mapping 차수 설정 가능")

df = load_data()
st.markdown("#### 데이터 head, tail 확인")
st.dataframe(df.head())
st.dataframe(df.tail())

# 데이터 전처리 & 분할

y = df["quality"].astype(float).values
X = df.drop(columns=["quality"])

if "wine_type" in X.columns:
    X = pd.get_dummies(X, columns=["wine_type"], drop_first=True)

if degree > 1:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly.fit_transform(X)
else:
    X = X.values if isinstance(X, pd.DataFrame) else X

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

if STANDARDIZE:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)


# Direct Solution - 시간
t0 = time.time() 
w_dir = direct_solution_ridge(X_tr, y_tr)  # Ridge Regularization
t_direct = time.time() - t0
yhat_dir_te = add_bias(X_te) @ w_dir

# Direct Solution - 평가 지표
mse_dir_te = mean_squared_error(y_te, yhat_dir_te)
mae_dir_te = mean_absolute_error(y_te, yhat_dir_te)
rmse_dir_te = float(np.sqrt(mse_dir_te))
r2_dir_te = r2_score(y_te, yhat_dir_te)

# Gradient Descent - 시간
t0 = time.time()
w_gd, curve = gd_train(X_tr, y_tr, lr=lr, epochs=EPOCHS)
t_gd = time.time() - t0
yhat_gd_te = add_bias(X_te) @ w_gd

# Gradient Descent - 평가 지표
mse_gd_te = mean_squared_error(y_te, yhat_gd_te)
mae_gd_te = mean_absolute_error(y_te, yhat_gd_te)
rmse_gd_te = float(np.sqrt(mse_gd_te))
r2_gd_te = r2_score(y_te, yhat_gd_te)

# 결과 시각화
st.markdown("### Test Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Direct Solution - MSE", f"{mse_dir_te:.4f}")
    st.metric("Direct Solution - MAE", f"{mae_dir_te:.4f}")
    st.metric("Direct Solution - RMSE", f"{rmse_dir_te:.4f}")
    st.metric("Direct Solution - R²", f"{r2_dir_te:.4f}")
    st.metric("Direct Solution - Time (s)", f"{t_direct:.4f}")
with col2:
    st.metric("Gradient Descent - MSE", f"{mse_gd_te:.4f}")
    st.metric("Gradient Descent - MAE", f"{mae_gd_te:.4f}")
    st.metric("Gradient Descent - RMSE", f"{rmse_gd_te:.4f}")
    st.metric("Gradient Descent - R²", f"{r2_gd_te:.4f}")
    st.metric("Gradient Descent - Time (s)", f"{t_gd:.4f}")

st.markdown("#### Test Set 비교")
st.table(pd.DataFrame({
    "Method": ["Direct Solution", "Gradient Descent"],
    "MSE": [mse_dir_te, mse_gd_te],
    "MAE": [mae_dir_te, mae_gd_te],
    "RMSE": [rmse_dir_te, rmse_gd_te],
    "R²": [r2_dir_te, r2_gd_te],
    "Time (s)": [t_direct, t_gd],
    "Settings": [f"degree={degree}, λ={RIDGE_LAMBDA}",
                 f"degree={degree}, lr={lr}, epochs={EPOCHS}"]
}))

# Gradient Descent 학습곡선 시각화
st.markdown("### Gradient Descent 학습곡선 시각화")
fig1, ax1 = plt.subplots()
ax1.plot(curve)
ax1.set_xlabel("Checkpoint (every 10 epochs)")
ax1.set_ylabel("Train MSE")
ax1.set_title("GD Convergence")
st.pyplot(fig1)

# Prediction vs True (Test)
st.markdown("### Prediction vs True (Test)")
fig2, ax2 = plt.subplots()
ax2.scatter(y_te, yhat_dir_te, label="Direct Solution")
ax2.scatter(y_te, yhat_gd_te, label="Gradient Descent")
mn, mx = min(y_te.min(), yhat_dir_te.min(), yhat_gd_te.min()), max(y_te.max(), yhat_dir_te.max(), yhat_gd_te.max())
ax2.plot([mn, mx], [mn, mx], linestyle="--")
ax2.set_xlabel("True")
ax2.set_ylabel("Predicted")
ax2.legend()
st.pyplot(fig2)

st.success("비교 완료")