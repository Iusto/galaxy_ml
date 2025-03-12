# 🚀 AI 프로젝트 포트폴리오

# 📌 프로젝트 개요  
    본 저장소는 Python을 활용한 데이터 분석, AI 모델 개발 </br>
    우주 천체 은하(타원, 나선, 나선막대) 3종류의 패턴을 학습해 분류(classfication)하는 모델

## 🛠 기술 스택  
    - 프로그래밍 언어: Python  
    - 데이터 분석 & 처리: Pandas, NumPy, OpenCV  
    - 머신러닝 & AI: TensorFlow, Keras, Scikit-learn, pytorch  
    - 웹 크롤링 & 자동화: Selenium  
    - 개발 환경: pycharm, Google Colab

## 📂 프로젝트 목록  

### [AI 기반 천체 분류 모델](https://github.com/Iusto/galaxy_ml)
    📌 개요: 머신러닝을 활용하여 천체 이미지를 분류하는 AI 모델 개발  
    🔧 사용 기술: TensorFlow, CNN, OpenCV  
    📊 결과: 분류 정확도 75% 이상 달성  

## 예제 코드: CNN 모델 구축
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])


## dataset
    타원은하 (E): 59,071개
    나선은하 (S): 53,334개
    나선막대은하 (SB): 51,008개

    https://drive.google.com/drive/folders/1WcU0HNk9OJnWv1Jijs93IIgEm22qYuzh?usp=drive_link


## 최근 변경 사항(Improve ML 참조)
    1. 데이터 증강 추가 (ImageDataGenerator에 다양한 변형 옵션 추가)
    2. ResNet50 사전 학습 모델 사용 (transfer learning 적용)
    3. ReduceLROnPlateau 추가 (학습률 감소 적용)
    4. SGD(momentum=0.9) 옵티마이저로 변경
    5. BatchNormalization 추가 (모델 성능 안정화)
    6. 클래스 불균형 고려하여 class_weight 적용
    위와같은 파라미터 조정 후 머신러닝 진행 중
