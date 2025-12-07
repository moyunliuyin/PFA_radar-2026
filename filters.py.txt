import numpy as np
import cv2
import time

class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.last_measurement = None
        self.last_update_time = time.time()

    def update(self, measurement):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        prediction = self.kf.predict()
        measurement = np.array([[np.float32(measurement[0])],
                                [np.float32(measurement[1])]])
        self.kf.correct(measurement)
        self.last_measurement = measurement

    def get_estimate(self):
        if self.last_measurement is None:
            return None
        state = self.kf.statePost
        return (state[0, 0], state[1, 0])

class EnhancedKalmanFilter(KalmanFilter):
    def __init__(self, process_noise=1e-3, measurement_noise=0.5):
        super().__init__(process_noise, measurement_noise)
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)

    def update(self, measurement):
        if self.last_measurement is None:
            dx = dy = 0
        else:
            dx = measurement[0] - self.last_measurement[0]
            dy = measurement[1] - self.last_measurement[1]
            dx_val = dx.item() if isinstance(dx, np.ndarray) else dx
            dy_val = dy.item() if isinstance(dy, np.ndarray) else dy
            self.kf.processNoiseCov[2, 2] = min(0.5, 0.1 + abs(dx_val) / 10)
            self.kf.processNoiseCov[3, 3] = min(0.5, 0.1 + abs(dy_val) / 10)
        super().update(measurement)