#!/usr/bin/env python3
"""
module which contains you only look once algorithm
"""
import tensorflow.keras as K


class Yolo:
    """
    uses the You Only Look Once v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        * model_path is the path to where a Darknet Keras model is stored
        * classes_path is the path to where the list of class names used for
          the Darknet model, listed in order of index, can be found
        * class_t is a float representing the box score threshold
          for the initial filtering step
        * nms_t is a float representing the IOU threshold
          for non-max suppression
        * anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
          containing all of the anchor boxes:
            - outputs is the number of outputs (predictions)
              made by the Darknet model
            - anchor_boxes is the number of anchor boxes used
              for each prediction
              2 => [anchor_box_width, anchor_box_height]
        * Public instance attributes:
            - model: the Darknet Keras model
            - class_names: a list of the class names for the model
            - class_t: the box score threshold for the initial filtering step
            - nms_t: the IOU threshold for non-max suppression
            - anchors: the anchor boxes
        """
        if '.h5' != model_path[-3:]:
            model_path + '.h5'
        if '.txt' != classes_path[-4:]:
            classes_path += '.txt'

        self.model = K.models.load_model(model_path)
        with open(file=classes_path, mode='r', encoding='utf-8') as f:
            self.class_names = list(filter(None, f.read().split('\n')))
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors