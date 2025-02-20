# 🚀 AI 프로젝트 포트폴리오  

## 📌 프로젝트 개요  
본 저장소는 Python을 활용한 데이터 분석, AI 모델 개발
우주 천체 은하(타원, 나선, 나선막대) 3종류의 패턴을 학습해 분류(classfication)하는 모델

## 🛠 기술 스택  
- 프로그래밍 언어: Python  
- 데이터 분석 & 처리: Pandas, NumPy, OpenCV  
- 머신러닝 & AI: TensorFlow, Keras, Scikit-learn, pytorch  
- 웹 크롤링 & 자동화: Selenium  
- 개발 환경: pycharm, Google Colab


## 📂 프로젝트 목록  

### [AI 기반 천체 분류 모델 (Galaxy ML)](./galaxy_ml/)**
📌 개요: 머신러닝을 활용하여 천체 이미지를 분류하는 AI 모델 개발  
🔧 사용 기술: TensorFlow, CNN, OpenCV  
📊 결과: 분류 정확도 85% 이상 달성  

python
# 예제 코드: CNN 모델 구축
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


dataset
  test - 3000
  train - 30000

추가파일 & 데이터셋 : https://drive.google.com/file/d/1pfqD37Q6PtJV58iqrqLUHPPrd_JzJS3L/view?usp=drive_link
