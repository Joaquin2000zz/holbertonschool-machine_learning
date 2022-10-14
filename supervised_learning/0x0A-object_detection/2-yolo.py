#!/usr/bin/env python3
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
        model_path = model_path if '.h5' == model_path[-3:] else model_path + '.h5'
        # using conventional if to pass pycodestyle because exceeds 80 characters
        # (ridiculous)
        if '.txt' != classes_path[-4:]:
            classes_path += '.txt'
        self.model = K.models.load_model(model_path)
        with open(file=classes_path, mode='r', encoding='utf-8') as f:
            self.class_names = f.read().split('\n')
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
        boc_class_probs = []

        # obtaining input shape to then make the calculation needed to the boxes
        # for more detail https://www.youtube.com/watch?v=vRqSO6RsptU
        _, input_width, input_height, _ = self.model.input.shape

        for i, output in enumerate(outputs):

            # normalizing values between 0 and 1 as the sigmoid function does
            box_confidences.append(self.__sigmoid(output[..., 4]))
            boc_class_probs.append(self.__sigmoid(output[..., :5]))

            # isolating t_x t_y t_w t_h from last dimention of the output
            box = output[..., :4]

            # obtaining grid shape and anchor boxes from output
            grid_height, grid_width, anchors = output.shape[:3]

            # getting each offset of cells
            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchors, axis=2)
            c_y = np.arange(grid_width).reshape(1, grid_width)
            c_y = np.repeat(c_y, grid_height, axis=0).T
            c_y = np.repeat(c_y[..., np.newaxis], anchors, axis=2)

            # obtaining t_x t_y t_w t_h individualy
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            # width and height of its corresponding anchor box
            p_w, p_h = self.anchors[i, :, 0], self.anchors[i, :, 1]

            # given the extracted values, let's start with the calculations
            # for more information https://www.youtube.com/watch?v=vRqSO6RsptU

            # bounding boxes
            b_x = self.__sigmoid(t_x) + c_x / input_width
            b_y = self.__sigmoid(t_y) + c_y / input_height
            b_w = p_w * np.exp(t_w) / input_width
            b_h = p_h * np.exp(t_h) / input_height

            # adjunsting for CV2 (top left and bottom right point of image)
            x_1 = (b_x - (b_w / 2)) * image_size[1]
            y_1 = (b_y - (b_h / 2)) * image_size[0]
            x_2 = (b_x + (b_w / 2)) * image_size[1]
            y_2 = (b_y + (b_h / 2)) * image_size[0]

            # setting values in its corresponding places
            box[..., 0] = x_1
            box[..., 1] = y_1
            box[..., 2] = x_2
            box[..., 3] = y_2

            boxes.append(box)

        return boxes, box_confidences, boc_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        
        """