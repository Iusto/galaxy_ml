# 🚀 AI 프로젝트 포트폴리오

## 📌 프로젝트 개요

이 프로젝트는 Python과 머신러닝 기술을 활용하여 천체 은하 이미지를 분류(Classification)하는 AI 모델을 개발하는 것을 목표로 했습니다. 초기에는 Selenium을 활용해 Google 이미지 검색을 통해 은하 사진을 수집하고 직접 라벨링하여 학습에 활용하려 했습니다. 하지만 약 3000장 수준의 적은 이미지 수로는 모델의 성능이 제한적이었고, 실제로 타원은하(E), 나선은하(S), 나선막대은하(SB)로 분류한 초기 모델은 약 55%의 낮은 정확도를 보였습니다.

## 🔍 데이터 수집 및 라벨링 과정

이후 더 나은 데이터를 찾던 중 Galaxy Zoo 프로젝트의 데이터셋을 발견하였고, 이를 활용하기로 결정했습니다. 해당 데이터셋 역시 라벨링이 완벽히 정리된 형태는 아니었기 때문에, 팀원 4명이 약 4000장의 이미지를 나눠 수작업으로 라벨링하였고, 이를 기반으로 나머지 데이터에 대해 자동 라벨링 시스템을 구성했습니다. 결과적으로 아래와 같은 대규모 라벨링 데이터셋을 완성했습니다:

* 타원은하 (E): 59,071장
* 나선은하 (S): 53,334장
* 나선막대은하 (SB): 51,008장

## ⚗️ 데이터 전처리 및 학습 전략

* `ImageDataGenerator`를 통한 데이터 증강
* ResNet50을 활용한 Transfer Learning 적용
* `ReduceLROnPlateau`로 학습률 감소 조정
* SGD(momentum=0.9) 옵티마이저 사용
* `BatchNormalization`으로 학습 안정화
* 클래스 불균형 문제 해결을 위한 `class_weight` 적용

총 4000장의 수작업 라벨링 데이터를 바탕으로 훈련된 모델은 **검증 데이터 기준 약 75%의 정확도**를 달성했습니다. 특히 나선은하와 나선막대은하는 구조적으로 유사한 점이 많아 분류가 어려웠으며, 이 과정에서 데이터 전처리와 라벨링 품질이 모델 성능에 얼마나 큰 영향을 미치는지를 실감하게 되었습니다.

### 📊 모델 비교 성능 표

| 모델               | 정확도 | Loss | 특이사항                           |
| ---------------- | --- | ---- | ------------------------------ |
| 베이스라인 CNN        | 55% | 0.91 | 수작업 라벨링 3000장, 기본 구조           |
| 개선 모델 (ResNet50) | 75% | 0.48 | 데이터 증강, 전이학습, class\_weight 적용 |

## 🧠 회고

* 적은 데이터로 학습하는 한계를 직접 경험하며, '좋은 데이터'의 중요성을 체감함
* 수작업 라벨링의 어려움을 체감하고, 향후 효율적인 라벨링 자동화 방법에 대한 고민을 시작함
* Transfer Learning과 하이퍼파라미터 튜닝을 통해 제한된 데이터에서도 일정 수준 이상의 성능을 낼 수 있음을 실험적으로 확인함

## 🛠 기술 스택

* 언어: Python
* 데이터 처리: Pandas, NumPy, OpenCV
* 모델링: TensorFlow, Keras, Scikit-learn, PyTorch
* 크롤링: Selenium
* 환경: PyCharm, Google Colab

## 📎 개발 환경 및 실행

본 프로젝트는 주로 **Google Colab** 환경에서 실행 및 실험되었으며, 클라우드 GPU를 적극 활용하여 모델 학습을 병렬로 수행하였습니다. 로컬 환경에서는 PyCharm 기반 디버깅 및 코드 정리를 병행하였습니다.

## 🗂️ 폴더 구조

```bash
project-root/
├── data/                # 원본 및 라벨링된 이미지 데이터
├── notebooks/           # 실험용 Jupyter 노트북
├── models/              # 학습된 모델 파일 저장
├── src/                 # 주요 학습 및 추론 코드
│   ├── preprocessing/   # 전처리 스크립트
│   ├── training/        # 모델 학습 모듈
│   ├── inference/       # 예측 로직
├── utils/               # 유틸 함수
├── README.md
```

## 💻 예제 코드 (CNN)

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

## 🔗 데이터셋 링크

[Google Drive 다운로드 링크](https://drive.google.com/file/d/1IJxSUsAFV3cUPBROa9uWaJI6zQm6UgcC/view?usp=sharing)

## 📸 예측 결과

![performance_comparison](https://github.com/user-attachments/assets/00d29b8c-d845-4886-af7f-f30d0b84d17e)
[성능 개선 그래프]

![confusion_matrix](https://github.com/user-attachments/assets/74a8f6cf-6f9b-4e7c-8c87-03b67c2b5b48)
[confusion matrix]


```bash
[입력 이미지]         → [예측 결과]
spiral_001.jpg         → 나선은하 (S)
barred_spiral_004.jpg → 나선막대은하 (SB)
elliptical_100.jpg     → 타원은하 (E)
```

## 📅 향후 개선 방향

* 더 정교한 자동 라벨링 알고리즘 구축
* 추가적인 사전학습 모델(MobileNet 등) 실험
* 학습 곡선 시각화 및 confusion matrix 기반 성능 분석
* Kaggle 등 외부 벤치마크와의 성능 비교
