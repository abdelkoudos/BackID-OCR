import os
import cv2
import numpy as np
import tensorflow as tf

class BackID_Recognizer:
    def __init__(self, config, model_path="BackID_model.h5", image_path="test.jpg"):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.image_path = image_path

    def is_valid_image(self, image):
        return image is not None and image.shape[0] > 0 and image.shape[1] > 0

    def preprocess_image(self, image):
        config = self.config
        image = cv2.resize(image, (config['resize_width'], config['resize_height']))
        original_resized = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, config['blockSize'], config['C']
        )
        thresh2 = cv2.adaptiveThreshold(
            thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 11
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = gray.shape
        valid_contours, boxes = [], []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 1 * img_w * img_h:
                valid_contours.append(c)
                boxes.append((x, y, w, h))
        filtered = sorted(zip(valid_contours, boxes), key=lambda cb: cb[1][2]*cb[1][3], reverse=True)
        top_contours = sorted(filtered[:14], key=lambda cb: cb[1][0])
        return top_contours, original_resized

    def iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0: return 0.0
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter_area / (area1 + area2 - inter_area)

    def non_max_suppression(self, boxes, iou_thresh=0.01, img_width=760):
        center_x = img_width // 2
        boxes = sorted(boxes, key=lambda b: abs((b[0] + b[2]) // 2 - center_x))
        selected = []
        for box in boxes:
            if all(self.iou(box, kept) < iou_thresh for kept in selected):
                selected.append(box)
        return selected

    def highlight_final_boxes(self, top_contours, image):
        config = self.config
        img_h, img_w, _ = image.shape
        raw_boxes = []
        for _, (x, y, w, h) in top_contours:
            cx = (x + w // 2)
            crop_x1 = max(cx - 7, 0) - 12
            crop_x2 = crop_x1 + config['box_width']
            if crop_x2 > img_w:
                crop_x2 = img_w
                crop_x1 = img_w - config['box_width']
            raw_boxes.append((crop_x1, 0, crop_x2, img_h))
        final_boxes = self.non_max_suppression(raw_boxes, iou_thresh=config['iou_threshold'], img_width=img_w)
        final_boxes = sorted(final_boxes, key=lambda b: b[0])
        new_boxes = []
        box_width = config.get('box_width', 40)
        for i in range(len(final_boxes) - 1):
            new_boxes.append(final_boxes[i])
            gap = final_boxes[i+1][0] - final_boxes[i][2]
            if gap >= box_width:
                new_x1 = final_boxes[i][2] + (gap - box_width) // 2
                new_x2 = new_x1 + box_width
                if new_x2 <= final_boxes[i+1][0]:
                    new_boxes.append((new_x1, 0, new_x2, img_h))
        if final_boxes:
            new_boxes.append(final_boxes[-1])
        return new_boxes[:14]

    def preprocess_and_crop(self, image):
        crops_processed = []
        top_contours, original = self.preprocess_image(image)
        final_boxes = self.highlight_final_boxes(top_contours, original)
        if len(final_boxes) != 14:
            return final_boxes, f"Expected 14 boxes, got {len(final_boxes)}"
        for (x1, y1, x2, y2) in final_boxes:
            x1 = max(x1, 0)
            x2 = min(x2, original.shape[1])
            crop = original[y1:y2, x1:x2]
            crop = cv2.resize(crop, (28, 28))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            norm = gray.astype("float32") / 255.0
            crops_processed.append(norm.reshape(28, 28, 1))
        return crops_processed

    def predict_image(self, image_path=None):
        image_path = image_path or self.image_path
        if not image_path or not os.path.exists(image_path):
            return "Invalid image path"
        image = cv2.imread(image_path)
        if not self.is_valid_image(image):
            return "Invalid image"
        crops = self.preprocess_and_crop(image)
        if not isinstance(crops, list) or len(crops) != 14:
            return f"Only found {len(crops)} digits" if isinstance(crops, list) else crops
        input_batch = np.array(crops).reshape((-1, 28, 28, 1))
        predictions = self.model.predict(input_batch)
        digits = [int(np.argmax(p)) for p in predictions]
        return digits