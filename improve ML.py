import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os

# 데이터 증강 적용
imageDataGen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = imageDataGen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

val_generator = imageDataGen.flow_from_directory(
    'dataset/test',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

# 클래스 가중치 계산
num_classes = len(train_generator.class_indices)
y_train_labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = {i: class_weights[i] for i in range(num_classes)}

# 사전 학습된 모델 불러오기
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # 가중치 고정

# 모델 구성
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 옵티마이저 및 학습률 조정
optimizer = SGD(learning_rate=0.001, momentum=0.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# 모델 학습
hist = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# 모델 저장
model.save('improved_galaxy.keras')

# 학습 과정 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'r', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 모델 평가
test_loss, test_acc = model.evaluate(val_generator, verbose=0)
print(f'손실: {test_loss:.2f}')
print(f'정확도: {test_acc * 100:.2f}%')
