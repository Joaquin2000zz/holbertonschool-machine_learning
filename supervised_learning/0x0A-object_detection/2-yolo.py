#!/usr/bin/env python3
"""
module which contains you only look once algorithm
"""
import tensorflow.keras as K
import numpy as np


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

    def __sigmoid(self, Z):
        """
        sigmoid activation
        """
        return 1 / (1 + np.exp(-Z))

    def process_outputs(self, outputs, image_size):
        """
        * outputs is a list of numpy.ndarrays containing the predictions from
         the Darknet model for a single image:
        * Each output will have the shape
          (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            - grid_height & grid_width => the height and width of the grid
                                          used for the output
            - anchor_boxes => the number of anchor boxes used
            - 4 => (t_x, t_y, t_w, t_h)
            - 1 => box_confidence
            - classes => class probabilities for all classes
        * image_size is a numpy.ndarray containing the image’s original size
          [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            - boxes: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, 4) containing
              the processed boundary boxes for each output, respectively:
                + 4 => (x1, y1, x2, y2)
                + (x1, y1, x2, y2) should represent the boundary
                  box relative to original image
            - box_confidences: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, 1) containing
              the box confidences for each output, respectively
            - box_class_probs: a list of numpy.ndarrays of shape
              (grid_height, grid_width, anchor_boxes, classes) containing
              the box’s class probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):

            # normalizing values between 0 and 1 as the sigmoid function does
            confidence = self.__sigmoid(output[..., 4])
            box_confidences.append(np.expand_dims(confidence, axis=-1))
            box_class_probs.append(self.__sigmoid(output[..., 5:]))

            # obtaining t_x t_y t_w t_h
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            # corresponding anchor box
            anchors = self.anchors[i]

            # getting grid width and height: number of grid cells
            g_h, g_w = output.shape[:2]

            # grid cell indices
            grid = np.tile(np.indices((g_h, g_w)).T, 3)
            grid = grid.reshape((g_w, g_h) + anchors.shape)

            # given the extracted values, let's start with the calculations
            # more information https://www.youtube.com/watch?v=vRqSO6RsptU

            # computing center of each bounding box
            b_xy = self.__sigmoid(t_xy) + grid

            # computing width and height of each bounding box
            b_wh = anchors * np.exp(t_wh)

            # normalizing b_xy and b_wh
            # b_xy: divided by grid's size
            # b_wh: divided by model's input shape
            b_xy /= [g_h, g_w]
            b_wh /= self.model.inputs[0].shape.as_list()[1:-1]

            # convert box center and weight in top left
            # and bottom right corners
            # top left
            b_xy1 = b_xy - (b_wh / 2)
            # bottom right
            b_xy2 = b_xy + (b_wh / 2)

            box = np.concatenate((b_xy1, b_xy2), axis=-1)

            # multiplying by the original image size
            box = box * np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        * boxes: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 4)
          containing the processed boundary boxes for each output
        * box_confidences: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 1)
          containing the processed box confidences for each output
        * box_class_probs: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, classes)
          containing the processed box class probabilities for each output
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
          - filtered_boxes: a numpy.ndarray of shape (?, 4) containing
                            all of the filtered bounding boxes
          - box_classes: a numpy.ndarray of shape (?,) containing
                         the class number that each box in
                         filtered_boxes predicts
          - box_scores: a numpy.ndarray of shape (?) containing
                        the box scores for each box in
                        filtered_boxes, respectively
        """

        box_scores = []
        # The paper of YOLO https://arxiv.org/pdf/1506.02640v5.pdf says:
        # At test time we multiply the conditional class probabilities
        # and the individual box confidence predictions,
        # Pr(Classi|Object) ∗ Pr(Object)
        for classi, confidence in zip(box_class_probs, box_confidences):
            box_scores.append(classi * confidence)

        # obtaining max value and their corresponding idx
        # as the exercise requieres
        max_scores = []
        idx_scores = []
        for score in box_scores:
            # takes max values of last dimentions be reduced to 1 dimention
            max_scores.append(np.amax(score, -1).reshape(-1))
            idx_scores.append(np.argmax(score, -1).reshape(-1))

        max_scores = np.concatenate(max_scores)
        idx_scores = np.concatenate(idx_scores)

        conditional_idx = np.where(max_scores >= self.class_t)

        # preparing boundig boxes according with the required shape
        # (?, 4)
        boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])

        # making each box of shape
        filtered_boxes = boxes[conditional_idx]
        box_classes = idx_scores[conditional_idx]
        box_scores = max_scores[conditional_idx]

        return filtered_boxes, box_classes, box_scores
