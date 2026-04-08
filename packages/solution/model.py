#!/usr/bin/env python3

import cv2
import numpy as np

from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from solution.config import CONF_THRESHOLD, STOP_DISTANCE, FORWARD_PWM
from solution.student_distance_estimator import DistanceEstimator

class MLModel:
    def __init__(self):
        print("Initializing MLModel")
        self.ground_projector = None
        self.min_area = 150
        self.max_detections = 10
        self.distance_estimator = DistanceEstimator()


    def _run_detector(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Fast color segmentation for yellow duckies.
        lower_yellow = np.array([18, 80, 80], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.4 or aspect_ratio > 1.8:
                continue

            fill_ratio = area / float(w * h)
            if fill_ratio < 0.25:
                continue

            score = min(1.0, max(fill_ratio, area / float(img_area * 0.01)))
            distance = self.distance_estimator.estimate_distance(w, h)
            detections.append([x, y, x + w, y + h, score, 0.0, distance])

        detections.sort(key=lambda det: det[4], reverse=True)

        if not detections:
            return np.empty((0, 7), dtype=np.float32)

        return np.array(detections[:self.max_detections], dtype=np.float32)


    def _should_stop(self, detections: np.ndarray): 
        if detections is None or len(detections) == 0:
            return False

        for x1, y1, x2, y2, score, _, distance in detections:
            if score < CONF_THRESHOLD:
                continue

            print(f"Detection: {x1:.0f}-{x2:.0f}, {y1:.0f}-{y2:.0f}, score={score:.2f}, distance={distance:.2f}m")

            if distance < STOP_DISTANCE:
                return True

        return False


    def set_ground_projector(self, gp):
        self.ground_projector = gp
        

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"Image-processing detector error: {e}")
            return [DifferentialPWM(left=0.0, right=0.0), None]
        if self._should_stop(detections):
            return [DifferentialPWM(left=0.0, right=0.0), detections]
        else:
            return [DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM), detections]
