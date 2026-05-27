import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1. 실제 데이터 및 모델 불러오기 ---
@st.cache_data
def load_data():
    # 데이터 불러오기 (보내주신 CSV 파일)
    return pd.read_csv('lung_cancer.csv')

@st.cache_resource
def load_models():
    # 모델과 스케일러 불러오기
    model = joblib.load('lung_cancer.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    return model, scaler

try:
    df = load_data()
    model, scaler = load_models()
except Exception as e:
    st.error(f"파일을 불러오지 못했습니다. 에러 내용: {e}")
    st.stop()

# --- 2. 앱 화면 구성 ---
st.title("🏥 환자 결과 예측 및 데이터 시각화")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 환자 정보 입력")
    # 입력창 구성
    age_input = st.number_input("나이 입력:", min_value=0.0, value=30.0, step=1.0)
    smoke_input = st.number_input("흡연량 입력:", min_value=0.0, value=15.0, step=1.0)
    drink_input = st.number_input("음주량 입력:", min_value=0.0, value=10.0, step=1.0)
    
    predict_button = st.button("결과 예측 및 시각화", use_container_width=True)

with col2:
    st.subheader("📉 예측 결과 및 시각화")
    
    if predict_button:
        # 3. 모델 예측을 위한 데이터프레임 생성
        new_patient = pd.DataFrame(
            [[age_input, smoke_input, drink_input]], 
            columns=['나이', '흡연량', '음주량']
        )
        
        # 스케일링 및 예측
        new_patient_scaled = scaler.transform(new_patient)
        pred_result = model.predict(new_patient_scaled)
        
        # 결과 출력 (Result 값이 0 또는 1이므로 그에 맞춰 출력)
        st.success(f"🎉 예측 완료! 이 환자의 Result(결과)는 **{pred_result[0]}** 입니다.")
        
        # --- 4. 시각화 영역 ---
        st.subheader("📍 전체 데이터 중 새 환자의 위치")
        
        fig = plt.figure(figsize=(8, 6))
        
        # [핵심 수정!] c=df['cluster'] 를 c=df['Result'] 로 변경했습니다!
        plt.scatter(df['나이'], df['흡연량'], c=df['Result'], alpha=0.5, cmap='viridis')
        
        # 새 환자 위치 표시 (X축: 나이, Y축: 흡연량)
        plt.scatter(age_input, smoke_input, c='black', s=300, marker='X')
        
        # 축 라벨 설정
        plt.xlabel('Age')
        plt.ylabel('Smoking')
        
        # 스트림릿에 그래프 띄우기
        st.pyplot(fig)
        
    else:
        st.info("왼쪽에서 정보를 입력하고 버튼을 눌러주세요!")