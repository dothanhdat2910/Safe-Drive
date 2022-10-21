import os
import cv2

path = 'data/images'
list_file = os.listdir(os.path.expanduser(path))
try:
    list_file.remove('__pycache__')
except:
    pass

x = 2166


for file in list_file:
    background = cv2.imread('data/background.jpg')
    # print(background.shape[:2])
    background = cv2.resize(background, (224 * 3, 320 * 3))
    h, w = background.shape[:2]
    center_h = int(h / 2)
    center_w = int(w / 2)
    img = cv2.imread(f'{path}/{file}')

    h1, w1 = img.shape[:2]
    center_h1 = int(h1 / 2)
    center_w1 = int(w1 / 2)

    background[center_h - center_h1:center_h + center_h1, center_w - center_w1:center_w + center_w1] = img
    cv2.imwrite(f'data/images_face/img_face_{x}.jpg', background)
    x += 1
