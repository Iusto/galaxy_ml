import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# 1. 데이터 준비하기
# 1) 훈련셋 준비하기(데이터 + 정답)
imageDataGen = ImageDataGenerator(rescale=1/255)


# (50000, 50000, 50000) 총 15만개 이미지 훈련셋

train_generator = imageDataGen.flow_from_directory('dataset/train',
                                                  target_size=(128,128),
                                                  batch_size=8,
                                                  class_mode='categorical',
                                                  shuffle=True)

test_generator= imageDataGen.flow_from_directory('dataset/test',
                                                        target_size=(128,128),
                                                        batch_size=8,
                                                        class_mode ='categorical',
                                                        shuffle=True)

#리스트로 저장
x_train_list=[]
y_train_list=[]
x_test_list=[]
y_test_list=[]


# # 1) 훈련셋
for i in range(3750): #8 * 3750
    img, label = train_generator.next()
    x_train_list.extend(img)
    y_train_list.extend(label)
    
    
# # 2) 테스트셋
for i in range(375): #테스트는 3000개 데이터
    img, label = test_generator.next()
    x_test_list.extend(img)
    y_test_list.extend(label)


# numpy 배열로 변경
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                   stratify=y_train,
                                                   test_size=0.1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
print('-' * 20)

model = load_model('galaxy.keras')

# # 모델 구조 변경
# model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# from tensorflow.keras.optimizers import Adam
# model.compile(loss='categorical_crossentropy', 
#               optimizer=Adam(lr=0.0001),
#               metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True)

# 5. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=16,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping])

model.save('galaxy.keras')

# 6. 모델 학습과정 살펴보기
# 그래프 기본 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (8, 8)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'r', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')

loss_ax.set_xlabel('epoch')
loss_ax.set_xlabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()



# 모델 평가하기
scores = model.evaluate(x_test, y_test, verbose=0)

print('손실: %.2f' % scores[0])
print('정확도: %.2f%%' % (scores[1] * 100))

# 예측하기
yhat = model.predict(x_test[:10])
yhat = np.argmax(yhat, axis=1)

y_test = np.argmax(y_test, axis = 1)

# 정답 및 예측 결과 출력
print('정답:', y_test[:10])
print('예측:', yhat[:10])

# 일부 데이터 시각화
plt_row = 2
plt_col = 5

plt.rcParams['figure.figsize'] = (10, 4)
fig, axarr = plt.subplots(plt_row, plt_col)

for i in range(10):
    sub_plt = axarr[i // 5, i % 5]
    sub_plt.imshow(x_test[i])
    sub_plt.set_title('R:%s  P:%s' % (y_test[i], yhat[i]))
    sub_plt.axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()
