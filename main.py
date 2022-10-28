import threading

from library.lib import *
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import time


class system:
    def __init__(self, video):
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load("export_model/saved_model")
        self.video = video
        self.category_index = label_map_util.create_category_index_from_labelmap("data/label_map.txt",
                                                                                 use_display_name=True)
        self.img = None
        self.count_drowsiness = 0
        self.alarmed = False
        self.result_drowsiness = False
        self.cap = cv2.VideoCapture(self.video)
        threading.Thread(target=self.run_video).start()
        threading.Thread(target=self.check_safe).start()

    def run_video(self, scale=1):

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            image_np = frame
            image_np = cv2.resize(image_np, dsize=None, fx=scale, fy=scale)
            self.img = image_np
            # cv2.imshow('window', self.img)
            # cv2.waitKey(10)

    def check_safe(self):
        print('check')
        while self.img is None:
            time.sleep(0.1)
        print('check1')
        while True:
            output_dict = run_inference_for_single_image(self.model, self.img)
            boxes = detect_face(self.img, output_dict, thresding=0.5)

            if len(boxes) == 4:
                self.img, result_drowsiness = drow(self.img, boxes)
            if self.result_drowsiness:
                self.count_drowsiness += 1
            else:
                self.count_drowsiness = 0
            print(self.count_drowsiness)
            vis_util.visualize_boxes_and_labels_on_image_array(
                self.img,
                output_dict['detection_boxes'][:4],
                output_dict['detection_classes'][:4],
                output_dict['detection_scores'][:4],
                self.category_index,
                min_score_thresh=.5,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=3)
            if self.count_drowsiness >= 60:
                self.count_drowsiness = 0
                try:
                    if not self.alarmed:
                        self.alarmed = True
                        # Duong dan den file wav

                        # Tien hanh phat am thanh trong 1 luong rieng
                        t = Thread(target=play_sound_wav, args=())
                        t.deamon = True
                        t.start()
                except:
                    pass
            else:
                self.alarmed = False
            cv2.imshow('window', self.img)
            cv2.waitKey(10)


if __name__ == '__main__':
    safe = system(0)
