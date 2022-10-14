#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('2-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('../data/yolo.h5', '../data/coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)