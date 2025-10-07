# 🍷 wine-quality-linear-regression
**Direct Solution vs Gradient Descent (with Ridge Regularization & Feature Mapping)**  
Streamlit을 활용하여 Linear Regression의 두 가지 최적화 방법을 실험적으로 비교한 과제입니다.

---

## 개요

**UCI Machine Learning Repository**의 [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)을 기반으로,  
**Direct Solution**과 **Gradient Descent**의 학습 성능 및 계산 효율성을 비교합니다.

- **Direct Solution**: 정규방정식을 통해 닫힌 해(closed-form solution)를 한 번에 계산  
- **Gradient Descent**: 손실 함수의 기울기를 이용해 반복적으로 가중치를 갱신하는 방식  

또한 **Ridge Regularization**(L2 penalty)과 **Feature Mapping**(Polynomial Expansion)을 적용하여  
모델의 일반화 성능과 수렴 특성을 함께 분석하였습니다.

---

## 주요 비교 항목

| 항목 | Direct Solution | Gradient Descent |
|------|-----------------|------------------|
| 학습 방식 | 정규방정식 기반 닫힌 해 계산 | 반복적 기울기 하강(Gradient 업데이트) |
| 정규화 | Ridge Regularization  | 미적용 |
| 조정 가능 하이퍼파라미터 | 없음 | Learning rate, Epochs |
| 계산 효율성 | 빠름 (O(n³)) | 느림 (반복 계산 필요) |
| 데이터 규모 적합성 | 소규모 데이터에 적합 | 대규모 데이터에 효율적 |

---

## 기능 요약

- **Ridge Regularization**: 과적합 방지를 위한 L2 penalty 추가  
- **Feature Mapping**: Polynomial feature 확장을 통한 비선형 관계 학습  
- **Streamlit UI**:  
  - Sidebar에서 **Learning rate** 및 **Feature mapping 차수** 실시간 조절  
  - MSE, MAE, RMSE, R² 등 다양한 평가지표 자동 계산  
  - 학습곡선(Convergence) 및 예측결과 시각화  

---

## How to Use

```bash
git clone https://github.com/your-username/wine-quality-linear-regression.git
cd wine-quality-linear-regression
pip install -r requirements.txt
streamlit run mldlhw1.py

