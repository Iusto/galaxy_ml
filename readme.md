
# 🚀 AI 프로젝트 포트폴리오: Galaxy ML

## 📌 프로젝트 개요

본 프로젝트는 Python과 머신러닝 기반으로 **천체 은하 이미지를 분류(Classification)** 하는 AI 모델을 개발하는 것을 목표로 하였습니다.  
초기에는 Selenium을 이용해 Google 이미지 검색으로 약 3,000장의 은하 사진을 수집한 후 직접 라벨링하여 학습에 활용하였으나, 데이터 양의 한계로 정확도는 약 **55% 수준**에 머물렀습니다.  
이후 더 나은 품질의 데이터 확보를 위해 Galaxy Zoo 프로젝트의 데이터를 도입하였고, 대규모 수작업 라벨링 및 자동화를 통해 **정확도 75%의 분류 모델**을 완성하였습니다.

---

## 🔍 데이터 수집 및 라벨링 전략

- **Galaxy Zoo 데이터셋**을 활용하되, 초기 라벨링 품질 이슈로 인해 팀원 4명이 약 **4,000장 수작업 라벨링** 진행
- 수작업 데이터를 기반으로 나머지 전체 데이터에 대해 **자동 라벨링 시스템** 구성
- 최종 구축된 라벨링 데이터셋 규모:

| 클래스        | 이미지 수 |
|---------------|------------|
| 타원은하 (E)   | 59,071장   |
| 나선은하 (S)   | 53,334장   |
| 나선막대은하 (SB) | 51,008장   |

---

## ⚗️ 데이터 전처리 및 모델 학습 전략

- `ImageDataGenerator`를 통한 이미지 증강
- `ResNet50` 기반 Transfer Learning 적용
- `ReduceLROnPlateau` → 에폭별 학습률 조정
- 옵티마이저: `SGD(momentum=0.9)`
- `BatchNormalization`으로 학습 안정화
- 클래스 불균형 해소를 위한 `class_weight` 적용

최종적으로 수작업 라벨링 기반 학습 데이터만으로도 **검증 정확도 약 75%**를 달성했습니다.  
특히, 구조적으로 유사한 **나선은하(S)** 와 **나선막대은하(SB)** 간의 혼동률이 높았으며, 이 과정에서 **데이터 품질과 전처리 전략의 중요성**을 체감했습니다.

---

## 📊 모델 비교 성능 요약

| 모델                    | 정확도 | Loss | 특이사항                                 |
|-------------------------|--------|------|------------------------------------------|
| 베이스라인 CNN          | 55%    | 0.91 | 단순 구조, 라벨 수 3,000장               |
| 개선 모델 (ResNet50 기반) | 75%    | 0.48 | 데이터 증강 + 전이학습 + class_weight 적용 |

📈 정확도 향상 추이 (Epoch별)

![Accuracy Chart](https://github.com/user-attachments/assets/00d29b8c-d845-4886-af7f-f30d0b84d17e)

---

## 🔎 Confusion Matrix 분석

![Confusion Matrix](https://github.com/user-attachments/assets/74a8f6cf-6f9b-4e7c-8c87-03b67c2b5b48)

- 타원은하(E)와 나선막대은하(SB)는 **정확도 86% 이상**으로 높은 성능을 보임
- 반면 나선은하(S)는 **SB와의 혼동률이 17%**로 가장 많은 오차 발생
- 해당 분류 난이도는 실제 천문학 도메인에서도 자주 언급되는 과제로, **모델 보완 여지** 존재

---

## 🧠 회고 및 인사이트

- **‘좋은 데이터’가 곧 좋은 모델의 핵심 자산**임을 체감함
- 수작업 라벨링의 한계를 직접 경험하며, **라벨링 자동화 전략**의 필요성을 인식
- Transfer Learning, class_weight, 하이퍼파라미터 조정 등을 통해 **실험적 접근**의 중요성 확인
- 단순 구현보다 **실제 데이터 품질 + 운영 고려 설계**가 중요한 것임을 학습

---

## 🛠 기술 스택

- **언어**: Python
- **데이터 처리**: Pandas, NumPy, OpenCV
- **모델링**: TensorFlow, Keras, Scikit-learn, PyTorch
- **크롤링**: Selenium
- **환경**: Google Colab (클라우드 GPU), PyCharm (로컬 디버깅)

---

## 🗂️ 프로젝트 구조

```bash
project-root/
├── data/                # 원본 및 라벨링된 이미지 데이터
├── models/              # 학습된 모델 저장
├── src/
│   ├── preprocessing/   # 데이터 전처리
│   ├── training/        # 모델 학습 모듈
├── README.md
```

---

## 💻 예제 코드 (CNN 기반)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])
```

---

## 📸 예측 예시

```bash
[입력 이미지]          → [예측 결과]
spiral_001.jpg         → 나선은하 (S)
barred_spiral_004.jpg  → 나선막대은하 (SB)
elliptical_100.jpg     → 타원은하 (E)
```

---

## 🔗 데이터셋 다운로드

👉 [Google Drive에서 다운로드](https://drive.google.com/file/d/1IJxSUsAFV3cUPBROa9uWaJI6zQm6UgcC/view?usp=sharing)

---

## 📅 향후 개선 방향

- 더 정교한 라벨링 자동화 알고리즘 구축
- 사전학습 모델(MobileNet 등) 성능 비교 실험
- confusion matrix 기반 클래스 간 혼동 해결 전략 강화
- 외부 벤치마크(Kaggle 등)와 성능 비교로 추가 개선 여지 탐색

---

## 🌐 운영 연동 가능성

본 프로젝트는 단순 연구용이 아닌, **Spring Boot 기반 API 서버와의 연동을 전제로 설계**되었습니다.  
이미지 업로드 시 모델이 자동 분류하여 게시 가능한 구조를 실험하였고, 실서비스 연계를 위한 구조 설계까지 고려하였습니다.
