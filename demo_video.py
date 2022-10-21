import cv2

from library.inference import run_inference_for_single_image
from library.detect_face import detect_face
from library.drowsiness import drow
from library.playsound_wav import play_sound_wav
from threading import Thread
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

# Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load("export_model/saved_model")

# predict

category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt", use_display_name=True)

# cap = cv2.VideoCapture('data/test/3_Trim.mp4')
cap = cv2.VideoCapture(0)


def crop(img):
    # print(img.shape)

    img_crop = img[:, 120:520, :]

    return img_crop


count_phone = 0
count_drinks = 0
count_drowsiness = 0
count_seatbelt = 0
count_frame = 0
alarmed = False
result_drowsiness = False
while (cap.isOpened()):

    ret, frame = cap.read()
    # print(type(frame))
    if not ret:
        break
    image_np = crop(frame)
    # image_np = frame
    # print("Done load image ")
    # image_np = cv2.resize(image_np, dsize=(480,640), fx=0.5, fy=0.5)
    # image_np = cv2.resize(image_np, dsize=None, fx=0.5, fy=0.5)
    output_dict = run_inference_for_single_image(model, image_np)
    # print("Done inference")
    boxes = detect_face(image_np, output_dict, thresding=0.4)

    if len(boxes) == 4:
        image_np, result_drowsiness = drow(image_np, boxes)
    if result_drowsiness:
        count_drowsiness += 1
    else:
        count_drowsiness = 0
    print(count_drowsiness)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'][:4],
        output_dict['detection_classes'][:4],
        output_dict['detection_scores'][:4],
        category_index,
        min_score_thresh=.4,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=3)
    if count_drowsiness >= 60:
        count_drowsiness = 0
        try:
            if not alarmed:
                alarmed = True
                # Duong dan den file wav

                # Tien hanh phat am thanh trong 1 luong rieng
                t = Thread(target=play_sound_wav, args=())
                t.deamon = True
                t.start()
        except:
            pass
    else:
        alarmed = False
    # print("Done draw on image ")
    # image_np = cv2.resize(image_np, dsize=None, fx=0.5, fy=0.5)
    # img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', image_np)
    cv2.waitKey(10)
print('Done!')
cap.release()
