import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import os
import random



folder_path = "galaxy_ori/test/SB"

# image_name = '5.jpg'
# img = load_img(os.path.join(folder_path, image_name))
# x = np.array(img)
# print(x.shape)


def check_image_existence(folder_path, image_name):
    files = os.listdir(folder_path)
    for file in files:
        if file == image_name:
            return True
    return False

imgDataGen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    channel_shift_range=0.2
)

image_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

# 5000개 이미지가 생성될때까지 반복
while True:
    if image_count >= 5000 :
        break
    random_number = random.randint(1, 251)
    image_name = '%d.jpg' % random_number

    if check_image_existence(folder_path, image_name):
        img = load_img(os.path.join(folder_path, image_name))
        x = np.array(img)
        x = x.reshape(1, 227, 227, 3)

        # 이미지 1개 생성
        tx = imgDataGen.flow(x, batch_size=1, save_to_dir="galaxy_ori/test/SB",
                              save_prefix='%d' % random_number,
                              save_format='jpg').next()

        # 무한 루프에서 빠져나가도록 설정
        image_count += 1
