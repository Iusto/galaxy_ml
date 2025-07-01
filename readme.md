# 🚀 AI 프로젝트 포트폴리오: Galaxy ML

## 📌 프로젝트 개요

본 프로젝트는 **딥러닝 기반 천체 은하 이미지 분류 시스템**을 개발하는 것을 목표로 합니다.  
초기에는 Selenium을 이용해 Google 이미지 검색으로 약 3,000장의 은하 사진을 수집한 후 직접 라벨링하여 학습에 활용하였으나, 데이터 양의 한계로 정확도는 약 **55% 수준**에 머물렀습니다.  
이후 더 나은 품질의 데이터 확보를 위해 Galaxy Zoo 프로젝트의 데이터를 도입하였고, 대규모 수작업 라벨링 및 자동화를 통해 **정확도 75%의 분류 모델**을 완성하였습니다.

### 🎯 주요 성과
- **3가지 은하 유형** 분류: 타원은하(E), 나선은하(S), 나선막대은하(SB)
- **ResNet50 기반 Transfer Learning** 적용으로 성능 향상
- **데이터 증강 및 클래스 가중치** 적용으로 불균형 데이터 문제 해결
- **Early Stopping 및 Learning Rate Scheduling** 적용으로 과적합 방지

---

## 🔍 데이터 수집 및 전처리 전략

### 데이터 수집
- **Galaxy Zoo 데이터셋**을 활용하되, 초기 라벨링 품질 이슈로 인해 팀원 4명이 약 **4,000장 수작업 라벨링** 진행
- 수작업 데이터를 기반으로 나머지 전체 데이터에 대해 **자동 라벨링 시스템** 구성
- **ImageDataGenerator**를 활용한 데이터 증강으로 부족한 데이터 보완

### 데이터 증강 기법
```python
ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### 최종 데이터셋 구성
| 클래스        | 훈련 데이터 | 테스트 데이터 | 총 이미지 수 |
|---------------|-------------|---------------|-------------|
| 타원은하 (E)   | 50,000장    | 1,000장       | 51,000장    |
| 나선은하 (S)   | 50,000장    | 1,000장       | 51,000장    |
| 나선막대은하 (SB) | 50,000장  | 1,000장       | 51,000장    |
| **총계**      | **150,000장** | **3,000장**   | **153,000장** |

---

## ⚗️ 모델 아키텍처 및 학습 전략

### 기본 CNN 모델 (ml.py)
```python
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])
```

### 개선된 ResNet50 모델 (improve_ML.py)
```python
# 사전 학습된 ResNet50 기반 Transfer Learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```

### 핵심 학습 전략
- **Transfer Learning**: ImageNet 사전 학습 가중치 활용
- **데이터 증강**: 회전, 이동, 확대/축소, 플립 등 적용
- **클래스 가중치**: 불균형 데이터 문제 해결
- **배치 정규화**: 학습 안정성 향상
- **드롭아웃**: 과적합 방지
- **학습률 스케줄링**: ReduceLROnPlateau 적용
- **조기 종료**: EarlyStopping으로 과적합 방지

### 하이퍼파라미터
- **옵티마이저**: SGD (learning_rate=0.001, momentum=0.9)
- **배치 크기**: 16
- **이미지 크기**: 128×128×3
- **에폭**: 최대 50 (Early Stopping 적용)

---

## 📊 모델 성능 비교 및 분석

### 성능 요약
| 모델                    | 정확도 | Loss | 특징                                     |
|-------------------------|--------|------|------------------------------------------|
| 기본 CNN (ml.py)        | 55%    | 0.91 | 단순 구조, 소규모 데이터셋               |
| 개선 ResNet50 (improve_ML.py) | 75%    | 0.48 | Transfer Learning + 데이터 증강 + 클래스 가중치 |

### 성능 향상 요인 분석
1. **Transfer Learning 효과**: ImageNet 사전 학습 가중치 활용으로 20% 성능 향상
2. **데이터 증강**: 회전, 이동, 플립 등으로 데이터 다양성 확보
3. **클래스 가중치**: 불균형 데이터 문제 해결로 안정적 학습
4. **정규화 기법**: BatchNormalization + Dropout으로 과적합 방지
5. **학습률 스케줄링**: 적응적 학습률 조정으로 최적화 개선

📈 **정확도 향상 추이 (Epoch별)**

![Accuracy Chart](https://github.com/user-attachments/assets/00d29b8c-d845-4886-af7f-f30d0b84d17e)

### 학습 곡선 분석
- **초기 10 에폭**: 급격한 성능 향상 (Transfer Learning 효과)
- **중기 10-30 에폭**: 안정적 성능 개선
- **후기 30+ 에폭**: Early Stopping으로 과적합 방지

---

## 🔎 모델 성능 상세 분석

### Confusion Matrix 분석
![Confusion Matrix](https://github.com/user-attachments/assets/74a8f6cf-6f9b-4e7c-8c87-03b67c2b5b48)

### 클래스별 성능 분석
| 클래스 | 정확도 | 주요 혼동 클래스 | 혼동률 |
|--------|--------|------------------|--------|
| 타원은하 (E) | 86% | 나선은하 (S) | 8% |
| 나선은하 (S) | 68% | 나선막대은하 (SB) | 17% |
| 나선막대은하 (SB) | 86% | 나선은하 (S) | 9% |

### 분류 난이도 분석
- **타원은하(E)**: 독특한 형태로 인해 높은 분류 정확도
- **나선은하(S) vs 나선막대은하(SB)**: 구조적 유사성으로 인한 높은 혼동률
- **천문학적 관점**: 실제 천문학에서도 S와 SB 구분은 전문가도 어려워하는 과제

### 오분류 원인 분석
1. **이미지 품질**: 저해상도 또는 노이즈가 많은 이미지
2. **각도 문제**: 은하의 관측 각도에 따른 형태 변화
3. **중간 형태**: S와 SB의 중간적 특성을 가진 은하
4. **라벨링 오류**: 수작업 라벨링 과정에서의 불일치

---

## 🧠 프로젝트 회고 및 학습 인사이트

### 핵심 학습 내용
1. **데이터의 중요성**
   - 모델 성능은 알고리즘보다 데이터 품질에 더 크게 의존
   - 수작업 라벨링의 일관성이 모델 성능에 직접적 영향
   - 데이터 증강을 통한 효과적인 데이터 확장 전략 습득

2. **Transfer Learning의 효과**
   - ImageNet 사전 학습 모델 활용으로 20% 성능 향상 달성
   - 적은 데이터로도 높은 성능을 얻을 수 있는 효과적 방법
   - 도메인 특화 Fine-tuning의 중요성 인식

3. **실험적 접근의 가치**
   - 다양한 하이퍼파라미터 조합 실험을 통한 최적화
   - 클래스 가중치, 정규화 기법 등의 실질적 효과 확인
   - 체계적인 실험 설계와 결과 분석의 중요성

### 기술적 도전과 해결
- **클래스 불균형**: compute_class_weight를 활용한 가중치 조정
- **과적합 문제**: Dropout, BatchNormalization, Early Stopping 적용
- **학습 불안정**: Learning Rate Scheduling으로 해결
- **데이터 부족**: 데이터 증강과 Transfer Learning으로 보완

### 향후 개선 방향
- **앙상블 모델**: 여러 모델의 예측 결과 결합으로 성능 향상
- **Attention 메커니즘**: 은하의 핵심 특징에 집중하는 모델 개발
- **준지도 학습**: 라벨이 없는 데이터 활용 방안 모색
- **실시간 추론**: 모델 경량화 및 최적화 연구

---

## 🛠 기술 스택 및 개발 환경

### 핵심 기술 스택
- **언어**: Python 3.8+
- **딥러닝 프레임워크**: TensorFlow 2.x, Keras
- **데이터 처리**: NumPy, Pandas, OpenCV
- **머신러닝**: Scikit-learn
- **시각화**: Matplotlib, Seaborn
- **이미지 처리**: PIL, ImageDataGenerator

### 주요 라이브러리 버전
```python
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
```

### 개발 환경
- **클라우드**: Google Colab (GPU 가속)
- **로컬**: PyCharm (디버깅 및 코드 관리)
- **버전 관리**: Git
- **데이터 저장**: Google Drive

### 하드웨어 요구사항
- **GPU**: NVIDIA GPU (CUDA 지원) 권장
- **메모리**: 최소 8GB RAM
- **저장공간**: 최소 10GB (데이터셋 포함)

---

## 🗂️ 프로젝트 구조

```bash
galaxy_ml-master/
├── docs/                           # 프로젝트 문서
│   └── AI머신러닝을_활용한_천체구별_프로젝트.pptx
├── models/                         # 학습된 모델 파일
│   └── galaxy_model.keras         # 최종 학습된 모델
├── src/                           # 소스 코드
│   ├── preprocessing/             # 데이터 전처리
│   │   └── image_exaggerate.py   # 이미지 증강 스크립트
│   └── training/                  # 모델 학습
│       ├── ml.py                 # 기본 CNN 모델
│       └── improve_ML.py         # 개선된 ResNet50 모델
├── dataset/                       # 데이터셋 (별도 다운로드)
│   ├── train/                    # 훈련 데이터
│   │   ├── E/                   # 타원은하
│   │   ├── S/                   # 나선은하
│   │   └── SB/                  # 나선막대은하
│   └── test/                     # 테스트 데이터
│       ├── E/
│       ├── S/
│       └── SB/
└── README.md                      # 프로젝트 문서
```

### 파일별 상세 설명
- **ml.py**: 기본 CNN 아키텍처 구현 및 학습
- **improve_ML.py**: ResNet50 기반 Transfer Learning 모델
- **image_exaggerate.py**: 데이터 증강을 통한 데이터셋 확장
- **galaxy_model.keras**: 최종 학습된 모델 가중치

---

## 💻 핵심 코드 예제

### 1. 데이터 로딩 및 전처리
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 증강 설정
imageDataGen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 로더 생성
train_generator = imageDataGen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)
```

### 2. ResNet50 기반 모델 구성
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

# 사전 학습된 모델 로드
base_model = ResNet50(weights='imagenet', include_top=False, 
                      input_shape=(128, 128, 3))
base_model.trainable = False

# 분류 헤드 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```

### 3. 모델 컴파일 및 학습
```python
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 모델 컴파일
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# 모델 학습
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=50,
                    class_weight=class_weight_dict,
                    callbacks=callbacks)
```

---

## 📸 모델 사용법 및 예측 예시

### 모델 로드 및 예측
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 학습된 모델 로드
model = load_model('models/galaxy_model.keras')

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 예측 함수
def predict_galaxy_type(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_names = ['타원은하(E)', '나선은하(S)', '나선막대은하(SB)']
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    return class_names[predicted_class], confidence

# 사용 예시
result, confidence = predict_galaxy_type('test_image.jpg')
print(f'예측 결과: {result} (신뢰도: {confidence:.2%})')
```

### 예측 결과 예시
```bash
[입력 이미지]              → [예측 결과]           [신뢰도]
spiral_galaxy_001.jpg      → 나선은하 (S)         → 87.3%
barred_spiral_004.jpg      → 나선막대은하 (SB)    → 92.1%
elliptical_galaxy_100.jpg  → 타원은하 (E)         → 95.8%
```

### 배치 예측
```python
# 여러 이미지 동시 예측
def batch_predict(image_paths):
    results = []
    for path in image_paths:
        result, confidence = predict_galaxy_type(path)
        results.append((path, result, confidence))
    return results
```

---

## 🔗 데이터셋 및 모델 다운로드

### 데이터셋 다운로드
👉 [Google Drive에서 다운로드](https://drive.google.com/file/d/1IJxSUsAFV3cUPBROa9uWaJI6zQm6UgcC/view?usp=sharing)

### 데이터셋 구성
- **총 크기**: 약 2.5GB
- **이미지 형식**: JPG (128×128 픽셀)
- **클래스**: 3개 (E, S, SB)
- **분할**: Train/Test (95:5 비율)

### 설치 및 실행 가이드
1. **저장소 클론**
   ```bash
   git clone https://github.com/your-repo/galaxy_ml.git
   cd galaxy_ml
   ```

2. **의존성 설치**
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn opencv-python
   ```

3. **데이터셋 다운로드 및 압축 해제**
   - 위 Google Drive 링크에서 데이터셋 다운로드
   - 프로젝트 루트에 `dataset/` 폴더로 압축 해제

4. **모델 학습 실행**
   ```bash
   # 기본 CNN 모델
   python src/training/ml.py
   
   # 개선된 ResNet50 모델
   python src/training/improve_ML.py
   ```

5. **데이터 증강 (선택사항)**
   ```bash
   python src/preprocessing/image_exaggerate.py
   ```

---

## 📅 향후 개선 및 확장 계획

### 단기 개선 목표 (1-3개월)
1. **모델 성능 향상**
   - EfficientNet, Vision Transformer 등 최신 아키텍처 실험
   - 앙상블 모델 구현으로 정확도 80% 이상 달성
   - 하드 샘플 마이닝을 통한 어려운 케이스 집중 학습

2. **데이터 품질 개선**
   - 전문가 검증을 통한 라벨링 품질 향상
   - 활성 학습(Active Learning)을 통한 효율적 라벨링
   - 노이즈 제거 및 이미지 품질 향상 전처리

### 중기 확장 계획 (3-6개월)
1. **기능 확장**
   - 은하 형태학적 특성 추가 분류 (나선팔 개수, 밝기 등)
   - 객체 탐지 모델로 은하 위치 및 크기 예측
   - 시계열 데이터 활용한 은하 진화 예측

2. **시스템 최적화**
   - 모델 경량화 (TensorFlow Lite, ONNX)
   - 실시간 추론 API 개발
   - 웹 인터페이스 구축

### 장기 비전 (6개월+)
1. **연구 확장**
   - 다중 파장 이미지 분석 (적외선, X선 등)
   - 3D 은하 구조 분석
   - 논문 발표 및 오픈소스 기여

2. **실용화**
   - 천문대 관측 시스템 연동
   - 교육용 도구 개발
   - 시민 과학 프로젝트 연계

---

## 🌐 실서비스 연동 및 배포 방안

### API 서버 연동 구조
본 프로젝트는 **실제 서비스 환경에서의 활용**을 고려하여 설계되었습니다.

```python
# Flask API 예시
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('models/galaxy_model.keras')

@app.route('/predict', methods=['POST'])
def predict_galaxy():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    # 이미지 전처리 및 예측
    result, confidence = predict_galaxy_type(image)
    
    return jsonify({
        'galaxy_type': result,
        'confidence': float(confidence),
        'timestamp': datetime.now().isoformat()
    })
```

### 배포 환경 옵션
1. **클라우드 배포**
   - AWS SageMaker: 모델 호스팅 및 자동 스케일링
   - Google Cloud AI Platform: 실시간 예측 API
   - Azure Machine Learning: 엔드투엔드 MLOps

2. **컨테이너 배포**
   ```dockerfile
   FROM tensorflow/tensorflow:2.8.0
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["python", "app.py"]
   ```

3. **엣지 배포**
   - TensorFlow Lite로 모델 경량화
   - 모바일 앱 또는 임베디드 시스템 연동

### 성능 모니터링
- **예측 정확도 추적**: 실시간 성능 지표 모니터링
- **응답 시간 측정**: API 응답 속도 최적화
- **사용량 분석**: 트래픽 패턴 및 리소스 사용량 추적

### 확장성 고려사항
- **로드 밸런싱**: 다중 인스턴스 배포
- **캐싱 전략**: 자주 요청되는 결과 캐싱
- **배치 처리**: 대량 이미지 일괄 처리 지원

---

### 기여 방법
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

## 🙏 감사의 말

- **Galaxy Zoo 프로젝트**: 고품질 은하 이미지 데이터 제공
- **TensorFlow 팀**: 강력한 딥러닝 프레임워크 제공
- **천문학 커뮤니티**: 도메인 지식 및 피드백 제공
- **오픈소스 커뮤니티**: 다양한 라이브러리 및 도구 제공

---

*이 프로젝트는 천문학과 인공지능의 융합을 통해 우주에 대한 이해를 넓히는 것을 목표로 합니다. 🌌*
