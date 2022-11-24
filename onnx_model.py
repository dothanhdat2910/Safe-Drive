import numpy as np
import onnxruntime
from models.research.object_detection.utils import ops as utils_ops
import tensorflow as tf



class predict_ONNX:
    def __init__(self,
                 model_path: str,
                 providers: list = None):
        self.model_path = model_path
        self.input_shape = [112, 112]
        if providers is None:
            self.providers = ['CUDAExecutionProvider',
                              'CPUExecutionProvider', ]
        else:
            self.providers = providers
        self.sess = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        print(self.sess.get_providers())

    def run(self, image: np.ndarray):
        return self.sess.run(None, {'input_tensor': image})

    def dictionary_output(self, output_list):
        dicts = {
            'raw_detection_scores': output_list[7],
            'detection_boxes': output_list[1],
            'detection_multiclass_scores': output_list[3],
            'detection_anchor_indices': output_list[0],
            'detection_scores': output_list[4],
            'detection_classes': output_list[2],
            'raw_detection_boxes': output_list[6],
            'num_detections': output_list[5]
        }
        # print(dicts)
        return dicts

    def predict_model(self,image):
        output_dict = self.run(image)
        output_dict = self.dictionary_output(output_dict)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections]
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict
