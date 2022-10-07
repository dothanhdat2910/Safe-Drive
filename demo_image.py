import cv2
from IPython.display import display
from library.inference import load_image_into_numpy_array, run_inference_for_single_image

from PIL import Image
from library.detect_face import detect_face
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import numpy as np

# Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load("export_model/saved_model")

# predict

category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt", use_display_name=True)

# image_path = '/content/drive/MyDrive/Safe-Drive/data/split_data/test/img_1413.jpg'
image_path = 'data/test/img_288.jpg'
print(type(image_path))
image_np = load_image_into_numpy_array(image_path)
print(type(image_np))
print("Done load image ")
# image_np = cv2.resize(image_np, dsize=(224,224), fx=1, fy=1)
# image_np = cv2.resize(image_np, dsize=None, fx=2, fy=2)
output_dict = run_inference_for_single_image(model, image_np)
print("Done inference")
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    min_score_thresh=.5,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=3)
# for i in range(len(output_dict['detection_classes'])):
#   if output_dict['detection_classes'][i] == 2:
#     print(output_dict['detection_boxes'][i], ':',i)

print("Done draw on image ")
# print(output_dict)
im_height, im_width = image_np.shape[:2]



boxes = detect_face(image_np,output_dict)

print('boxes: ',boxes)
img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
cv2.imshow('window', img)
box = [0.28926688, 0.36827117, 0.48066103, 0.66214865]
cv2.waitKey(0)
