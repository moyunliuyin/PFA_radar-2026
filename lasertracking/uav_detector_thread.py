import copy
import queue
import threading
import time

import cv2
import numpy as np


class UAVDetectorThread:
    """Independent UAV detection worker.

    This worker is isolated from the main recognition pipeline:
    - never changes camera hardware params
    - only consumes frame copies
    - only detects on the right half of each frame
    """

    def __init__(
        self,
        enabled=True,
        model_path="lasertracking/aerial.pt",
        img_size=(960, 960),
        conf_thres=0.15,
        iou_thres=0.45,
        max_det=6,
        process_interval_s=0.0,
        right_half_only=True,
        show_window=True,
        strict_yolo26=True,
        exposure_adapt=None,
        queue_size=2,
    ):
        self.enabled = bool(enabled)
        self.model_path = str(model_path)
        if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
            self.img_size = (int(img_size[0]), int(img_size[1]))
        else:
            self.img_size = (960, 960)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.process_interval_s = max(0.0, float(process_interval_s))
        self.right_half_only = bool(right_half_only)
        self.show_window = bool(show_window)
        self.strict_yolo26 = bool(strict_yolo26)

        exposure_adapt = exposure_adapt or {}
        self.exp_enabled = bool(exposure_adapt.get("enabled", True))
        self.exp_adaptive = bool(exposure_adapt.get("adaptive", True))
        self.exp_gamma = float(exposure_adapt.get("gamma", 1.35))
        self.exp_target_brightness = float(exposure_adapt.get("target_brightness", 108.0))
        self.exp_max_gamma = float(exposure_adapt.get("max_gamma", 2.2))
        self.exp_clahe_clip = float(exposure_adapt.get("clahe_clip", 2.2))
        self.exp_clahe_grid = max(2, int(exposure_adapt.get("clahe_grid", 8)))

        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = None
        self._detector = None
        self._backend = "none"
        self._gamma_lut_cache = {}

        self._latest_lock = threading.Lock()
        self._latest_result = {
            "ok": False,
            "timestamp": 0.0,
            "detections": [],
            "fps": 0.0,
            "brightness": 0.0,
            "right_half_only": self.right_half_only,
            "backend": self._backend,
        }

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_frame(self, frame):
        if not self.enabled:
            return
        if frame is None:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_latest_result(self):
        with self._latest_lock:
            return copy.deepcopy(self._latest_result)

    def get_visualization(self, frame):
        if not self.enabled or frame is None:
            return None

        vis = frame.copy()
        h, w = vis.shape[:2]
        split_x = w // 2 if self.right_half_only else 0

        if self.right_half_only:
            cv2.line(vis, (split_x, 0), (split_x, h - 1), (80, 80, 80), 2)
            cv2.putText(
                vis,
                "UAV detect area: RIGHT HALF",
                (max(10, split_x + 10), 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        latest = self.get_latest_result()
        for det in latest.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"{det['cls']} {det['conf']:.2f}",
                (int(x1), max(18, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            vis,
            f"UAV fps={latest.get('fps', 0.0):.1f} v={latest.get('brightness', 0.0):.1f}",
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        return vis

    def resize_for_display(self, frame, target_size):
        """Keep aspect ratio and avoid aggressive stretch blur on window display."""
        if frame is None:
            return None
        if not isinstance(target_size, (list, tuple)) or len(target_size) < 2:
            return frame

        target_w = max(1, int(target_size[0]))
        target_h = max(1, int(target_size[1]))
        h, w = frame.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _estimate_brightness(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))

    def _apply_gamma(self, frame, gamma):
        gamma = max(0.1, float(gamma))
        key = round(gamma, 3)
        table = self._gamma_lut_cache.get(key)
        if table is None:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            self._gamma_lut_cache[key] = table
        return cv2.LUT(frame, table)

    def _enhance(self, frame):
        if not self.exp_enabled:
            return frame

        enhanced = frame
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        v = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[:, :, 2]
        v_std = float(np.std(v))
        clip = self.exp_clahe_clip
        if v_std < 26.0:
            clip *= 1.2
        elif v_std > 55.0:
            clip *= 0.9
        clip = float(max(1.0, min(4.0, clip)))

        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(int(self.exp_clahe_grid), int(self.exp_clahe_grid)),
        )
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        gamma_use = float(self.exp_gamma)
        if self.exp_adaptive:
            bright = self._estimate_brightness(enhanced)
            if bright > 1.0:
                gamma_use = min(self.exp_max_gamma, max(1.0, self.exp_target_brightness / bright))

        return self._apply_gamma(enhanced, gamma_use)

    def _lazy_init_detector(self):
        if self._detector is not None:
            return True

        # 主程序无人机模型固定为yolo26，这里默认严格使用ultralytics后端。
        try:
            from ultralytics import YOLO

            self._detector = YOLO(self.model_path)
            self._backend = "yolo26"
            print(f"[UAV] backend={self._backend}, model={self.model_path}")
            return True
        except Exception as yolo26_exc:
            print(f"[UAV] yolo26 backend init failed: {yolo26_exc}")
            self._backend = "none"
            if self.strict_yolo26:
                print("[UAV] strict_yolo26=True, UAV thread disabled.")
                return False
            return False

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _predict(self, image):
        if self._backend == "yolo26":
            imgsz = max(int(self.img_size[0]), int(self.img_size[1]))
            results = self._detector.predict(
                source=image,
                conf=float(self.conf_thres),
                iou=float(self.iou_thres),
                imgsz=imgsz,
                max_det=int(self.max_det),
                verbose=False,
            )

            detections = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue

                xyxy = self._to_numpy(boxes.xyxy)
                confs = self._to_numpy(boxes.conf)
                clses = self._to_numpy(boxes.cls).astype(int)
                names = getattr(result, "names", {}) or {}

                for i in range(len(clses)):
                    x1, y1, x2, y2 = xyxy[i]
                    w = max(0.0, float(x2) - float(x1))
                    h = max(0.0, float(y2) - float(y1))
                    cls_id = int(clses[i])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    detections.append((str(cls_name), [int(x1), int(y1), int(w), int(h)], float(confs[i])))

            return detections

        return []

    def _worker_loop(self):
        if not self._lazy_init_detector():
            return

        last_run = 0.0
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.time()
            if self.process_interval_s > 0.0 and (now - last_run) < self.process_interval_s:
                continue
            last_run = now

            try:
                t0 = time.time()
                h, w = frame.shape[:2]
                split_x = w // 2 if self.right_half_only else 0
                roi = frame[:, split_x:]
                if roi.size == 0:
                    continue

                brightness = self._estimate_brightness(roi)
                roi_enhanced = self._enhance(roi)
                pred = self._predict(roi_enhanced)
                # 增强后未检出时，回退原图再检一次，避免增强过度导致漏检。
                if not pred:
                    pred = self._predict(roi)

                dets = []
                for cls_name, xywh, conf in pred:
                    x, y, bw, bh = xywh
                    x1 = int(x + split_x)
                    y1 = int(y)
                    x2 = int(x + bw + split_x)
                    y2 = int(y + bh)
                    dets.append(
                        {
                            "cls": str(cls_name),
                            "conf": float(conf),
                            "bbox": (x1, y1, x2, y2),
                        }
                    )

                dt = max(1e-6, time.time() - t0)
                result = {
                    "ok": True,
                    "timestamp": time.time(),
                    "detections": dets,
                    "fps": 1.0 / dt,
                    "brightness": float(brightness),
                    "right_half_only": self.right_half_only,
                    "backend": self._backend,
                }
                with self._latest_lock:
                    self._latest_result = result
            except Exception as exc:
                with self._latest_lock:
                    self._latest_result = {
                        "ok": False,
                        "timestamp": time.time(),
                        "detections": [],
                        "fps": 0.0,
                        "brightness": 0.0,
                        "right_half_only": self.right_half_only,
                        "backend": self._backend,
                        "error": str(exc),
                    }
