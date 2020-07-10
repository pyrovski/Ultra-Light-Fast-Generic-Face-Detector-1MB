"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import sys
#import time

import argparse
import cv2
import numpy as np
import onnx
from caffe2.python.onnx import backend
import onnxruntime as ort
import vision.utils.box_utils_numpy as box_utils


def predict(width,
            height,
            confidences,
            boxes,
            prob_threshold,
            iou_threshold=0.3,
            top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(
            box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(
        np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="face detection")
    parser.add_argument(
        "--input", type=str, default="", help="input file path")
    return parser.parse_args()


def main():

    args = parse_args()

    onnx_path = "models/onnx/version-RFB-320.onnx"

    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.input)

    threshold = 0.7

    frame = 0
    while True:
        # TODO: pipeline reading/conversion and inference
        _, orig_image = cap.read()
        if orig_image is None:
            print("no img", file=sys.stderr)
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # confidences, boxes = predictor.run(image)
        #time_time = time.time()
        confidences, boxes = ort_session.run(None, {input_name: image})
        #        print("cost time: {}".format(time.time() - time_time))
        boxes, _, _ = predict(orig_image.shape[1], orig_image.shape[0],
                              confidences, boxes, threshold)
        print("frame %06d box count: %d" % (frame, len(boxes)))
        frame = frame + 1


if __name__ == '__main__':
    main()
