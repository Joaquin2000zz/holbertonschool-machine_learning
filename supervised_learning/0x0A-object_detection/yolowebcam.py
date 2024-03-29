#!/usr/bin/env python3
"""
module which contains you only look once algorithm
"""
import cv2
from glob import glob
import numpy as np
import tensorflow.keras as K
import os


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        applies IoU to discard leftovers bounding boxes
        - filtered_boxes: a numpy.ndarray of shape (?, 4)
                          containing all of the filtered bounding boxes:
        - box_classes: a numpy.ndarray of shape (?,) containing the class
                       number for the class that filtered_boxes
                       predicts, respectively
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
                      for each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
                            predicted_box_scores):
        - box_predictions: a numpy.ndarray of shape (?, 4)
          containing all of the predicted bounding boxes ordered
          by class and box score
        - predicted_box_classes: a numpy.ndarray of shape (?,) containing
                                 the class number for box_predictions ordered
                                 by class and box
          score, respectively
        - predicted_box_scores: a numpy.ndarray of shape (?) containing the
                                box scores for box_predictions ordered by
                                class and box score, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        classes = np.unique(box_classes)

        for cls in classes:
            i = np.where(cls == box_classes)

            boxes = filtered_boxes[i]
            obj = box_classes[i]
            scores = box_scores[i]

            which = self.IoU(boxes, scores)

            box_predictions.append(boxes[which])
            predicted_box_classes.append(obj[which])
            predicted_box_scores.append(scores[which])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def IoU(self, boxes, scores):
        """
        applies intersection over union
        - boxes: filtered boxes of specific class
        - scores: filtered scores of specific class
        Returns: slice object to choose correct boundig box
        """
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)[::-1]

        # keep looping while some indexes still remain in the indexes
        # list
        while idxs.size > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes
            last = idxs[0]
            pick.append(last)

            # corners of the box
            xx1 = np.maximum(x1[last], x1[idxs[1:]])
            yy1 = np.maximum(y1[last], y1[idxs[1:]])
            xx2 = np.minimum(x2[last], x2[idxs[1:]])
            yy2 = np.minimum(y2[last], y2[idxs[1:]])

            # compute the width and height of the bounding box
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = (w * h) / (area[last] + area[idxs[1:]] - (w * h))

            # set which values choose acording the nms thershold
            choose = np.where(overlap <= self.nms_t)[0]

            idxs = idxs[choose + 1]

        return pick

    @staticmethod
    def load_images(folder_path):
        """
        - folder_path: a string representing the path to the folder
          holding all the images to load
        Returns a tuple of (images, image_paths):
            + images: a list of images as numpy.ndarrays
            + image_paths: a list of paths to the individual images in images
        """
        images_paths = glob(folder_path + '/*.jpg')
        return [cv2.imread(file) for file in images_paths], images_paths

    def preprocess_images(self, images):
        """
        - images: a list of images as numpy.ndarrays
        - Resize the images with inter-cubic interpolation
        - Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
              containing all of the preprocessed images
                + ni: the number of images that were preprocessed
                + input_h: the input height for the Darknet model Note:
                           this can vary by model
                + input_w: the input width for the Darknet model Note:
                           this can vary by model
                + 3: number of color channels
        - image_shapes: a numpy.ndarray of shape (ni, 2) containing
                        the original height and width of the images
            + 2 => (image_height, image_width)
        """
        # extracting n images and input shape
        n = len(images)
        _, h, w, c = self.model.input.shape

        # np.zeros to call np just once
        pimages = np.zeros(shape=(n, h, w, c))
        image_shapes = np.zeros(shape=(n, 2), dtype=int)

        for i, image in enumerate(images):
            # using inter cubic because of their advantages
            # https://chadrick-kwag.net/cv2-resize-interpolation-methods/
            resized = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
            resized = resized / 255

            pimages[i] = resized#np.expand_dims(resized, axis=-1)
            image_shapes[i] = [image.shape[0], image.shape[1]]

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        - image: a numpy.ndarray containing an unprocessed image
        - boxes: a numpy.ndarray containing the boundary boxes for the image
        - box_classes: a numpy.ndarray containing the class indices
          for each box
        - box_scores: a numpy.ndarray containing the box scores for each box
        - file_name: the file path where the original image is stored
        - Displays the image with all boundary boxes, class names,
          and box scores
        - Boxes should be drawn as with a blue line of thickness 2
        - Class names and box scores should be drawn above each box in red
            + Box scores should be rounded to 2 decimal places
            + Text should be written 5 pixels above the top left
              corner of the box
            + Text should be written in FONT_HERSHEY_SIMPLEX
            + Font scale should be 0.5
            + Line thickness should be 1
            + You should use LINE_AA as the line type
        - The window name should be the same as file_name
        - If the s key is pressed:
            + The image should be saved in the directory detections,
              located in the current directory
            + If detections does not exist, create it
            + The saved image should have the file name file_name
            + The image window should be closed
        - If any key besides s is pressed, the image window should
          be closed without saving
        """
        for box, cls, score in zip(boxes, box_classes, box_scores):
            # each box contains x1, x2, y1, y2
            x1, x2 = [int(x) for x in box[:2]]
            y1, y2 = [int(y) for y in box[2:]]
            text = "{} {:0.2f}".format(self.class_names[cls], score)
            image = cv2.rectangle(img=image, pt1=(x1, x2), pt2=(y1, y2),
                                  color=(255, 0, 0), thickness=1)
            image = cv2.putText(image, text, (x1, x2 -5),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=(0, 0, 255), thickness=1,
                                lineType=cv2.LINE_AA)
        
        cv2.imshow(file_name, image)
        scape = cv2.waitKey(0)

        if scape == ord('s'):
            if not os.path.exists('detections'):
                os.mkdir('detections')

            cv2.imwrite(os.path.join('./detections', file_name), image)
            cv2.destroyAllWindows()
        else:
            cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        - folder_path: a string representing the path to the folder
                     holding all the images to predict
        - All image windows should be named after the corresponding
          image filename without its full path
        - Displays all images using the show_boxes method
        - Returns: a tuple of (predictions, image_paths):
        predictions: a list of tuples for each image of
                     (boxes, box_classes, box_scores)
        - image_paths: a list of image paths corresponding to each
                       prediction in predictions
        """
        
        images, images_path = self.load_images(folder_path=folder_path)

        pimages, image_shapes = self.preprocess_images(images=images)

        results = self.model.predict(pimages)

        file_names = []
        predictions = []
        for i, image in enumerate(images):
            outi = [result[i] for result in results]
            print(image_shapes)
            boxes, box_confidences, box_class_probs = self.process_outputs(outi, image_shapes[i])

            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)

            box_p, p_box_classes, p_box_scores = self.non_max_suppression(filtered_boxes, box_classes, box_scores)

            file_name = images_path[i].split('\\')[-1]
            file_names.append(file_names)
            predictions.append((boxes, box_classes, box_scores))

            self.show_boxes(image, box_p, p_box_classes, p_box_scores,
                            file_name)

        return predictions, glob(folder_path + '/*.jpg')

    def predict_webcam(self):
        """
        predicts in real time
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            pimages, image_shapes = self.preprocess_images(images=[frame])
            results = self.model.predict(pimages)

            boxes, box_confidences, box_class_probs = self.process_outputs(results, image_shapes)

            filtered_boxes, box_classes, box_scores = self.filter_boxes(boxes, box_confidences, box_class_probs)

            box_p, p_box_classes, p_box_scores = self.non_max_suppression(filtered_boxes, box_classes, box_scores)
            for box, cls, score in zip(box_p, p_box_classes, p_box_scores):
                # each box contains x1, x2, y1, y2
                x1, x2 = [int(x) for x in box[:2]]
                y1, y2 = [int(y) for y in box[2:]]
                text = "{} {:0.2f}".format(self.class_names[cls], score)
                frame = cv2.rectangle(img=frame, pt1=(x1, x2), pt2=(y1, y2),
                                      color=(255, 0, 0), thickness=1)
                frame = cv2.putText(frame, text, (x1, x2 -5),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                    color=(0, 0, 255), thickness=1,
                                    lineType=cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('You Only Look Once aka YOLO', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()