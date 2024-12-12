import os
import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from collections import deque
import subprocess
import furiosa.runtime.session
from furiosa.runtime.sync import create_runner
import psutil
from utils.parse_params import get_model_params_from_cfg
from utils.preprocess import YOLOPreProcessor, letterbox
import yaml
from utils.postprocess import ObjDetDecoder
import json
import threading
import typer

do = typer.Typer(pretty_exceptions_show_locals=False)

class DetectionPreProcessor:
    """YOLO 검출기 전처리"""
    def __init__(self, input_shape=(640, 640)):
        self.input_shape = input_shape
        self.preprocessor = YOLOPreProcessor()

    def __call__(self, img):
        input_tensor, preproc_params = self.preprocessor(
            img,
            new_shape=self.input_shape,
            tensor_type="uint8"
        )
        return input_tensor, preproc_params

class ReIDPreProcessor:
    """간소화된 FastReID 전처리"""
    def __init__(self, input_shape=(128, 256)):
        self.input_shape = input_shape

    def __call__(self, img):
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # uint8 타입 유지, BGR 채널 순서 유지
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

class NPUModel:
    """NPU 모델 기본 클래스"""
    def __init__(self, model_path):
        self.runner = create_runner(model_path, device="warboy(2)*1")

    def __del__(self):
        if hasattr(self, 'runner'):
            self.runner.close()

class YOLODetector(NPUModel):
    """YOLO 검출기"""
    def __init__(self, cfg_path):
        self.model_cfg, _, self.model_name, model_path, input_shape, self.class_names = (
            get_model_params_from_cfg(cfg_path, mode="inference")
        )
        super().__init__(model_path)
        self.preprocessor = DetectionPreProcessor(input_shape)
        self.decoder = ObjDetDecoder(self.model_name, **self.model_cfg)
        self.conf_thres = self.model_cfg.get('conf_thres', 0.5)

    def detect(self, img):
        input_tensor, preproc_params = self.preprocessor(img)
        outputs = self.runner.run([input_tensor])
        predictions = self.decoder(outputs, preproc_params, img.shape[:2])

        if len(predictions) > 0:
            dets = predictions[0]
            person_mask = dets[:, 5].astype(np.int32) == 0
            confidences = dets[person_mask, 4]
            conf_mask = confidences > self.conf_thres
            return dets[person_mask][conf_mask]

        return np.array([])

class Track:
    """트랙 관리"""
    def __init__(self, feature, bbox, track_id):
        self.id = track_id
        self.bbox = bbox
        self.feature_history = deque(maxlen=3)  # feature_history 선언 추가
        self.feature_history.append(feature)
        self.hits = 1
        self.time_since_update = 0
        self.state = 'tentative'
        self.last_feature = feature

    def update(self, feature, bbox):
        self.bbox = bbox
        if np.dot(feature, self.last_feature) < 0.95:
            self.feature_history.append(feature)
            self.last_feature = feature
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'

    def get_feature(self):
        return self.last_feature

    def __del__(self):
        self.feature_history.clear()

from collections import deque

class FastReIDExtractor(NPUModel):
    """개별 처리에 최적화된 FastReID 특징 추출기"""
    def __init__(self, model_path, cache_max_size=100):
        super().__init__(model_path)
        self.preprocessor = ReIDPreProcessor()
        self._feature_cache = deque(maxlen=cache_max_size)  # 고정 크기 캐시
        self.cache_max_size = cache_max_size

    def _extract_single_feature(self, patch):
        """단일 패치 처리 최적화"""
        try:
            # 전처리
            processed = self.preprocessor(patch)
            # (C,H,W) -> (1,C,H,W) 변환
            processed = np.expand_dims(processed, axis=0).astype(np.uint8)
            # 추론
            output = self.runner.run([processed])[0]
            # 정규화
            feature = output / (np.linalg.norm(output, axis=1, keepdims=True) + 1e-10)
            return feature[0]
        except Exception as e:
            print(f"Single feature extraction error: {str(e)}")
            return None

    def extract_features(self, img_patches):
        """여러 이미지 패치의 특징을 추출"""
        if not img_patches:
            return np.array([])

        features = []
        for patch in img_patches:
            # 캐시 확인
            cached_feature = self._get_cached_feature(patch)
            if cached_feature is not None:
                features.append(cached_feature)
                continue

            # 새로운 특징 추출
            feature = self._extract_single_feature(patch)
            if feature is not None:
                # 캐시에 저장
                self._feature_cache.append((patch, feature))
                features.append(feature)

        if not features:
            return np.array([])

        return np.stack(features)

    def _get_cached_feature(self, patch):
        """캐시에 동일한 패치가 있는지 확인"""
        for cached_patch, cached_feature in self._feature_cache:
            if np.array_equal(cached_patch, patch):  # 패치 비교
                return cached_feature
        return None

    def __del__(self):
        super().__del__()
        self._feature_cache.clear()

class CameraTracker:
    """단일 카메라 트래커"""
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.tracks = {}
        self.next_id = 1
        self.max_age = 20
        self.min_hits = 3
        self.last_cleanup = 0
        self.cleanup_interval = 100

    def cleanup(self, current_frame):
        if current_frame - self.last_cleanup >= self.cleanup_interval:
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                if track.time_since_update > self.max_age:
                    del self.tracks[track_id]
            self.last_cleanup = current_frame

class MultiCameraTracker:
    """다중 카메라 트래커"""
    def __init__(self, reid_model_path, yolo_cfg_path):
        self.detector = YOLODetector(yolo_cfg_path)
        self.reid = FastReIDExtractor(reid_model_path)
        self.camera_trackers = {}
        self.cos_threshold = 0.7
        self.min_box_area = 100
        self.frame_count = 0
        self.output_dir = "tracking_results"  # 결과 저장 디렉터리
        os.makedirs(self.output_dir, exist_ok=True)

    def process_frame(self, frame, camera_id):
        detections = self.detector.detect(frame)
        if len(detections) == 0:
            return np.array([]), [], []

        boxes = detections[:, :4]
        scores = detections[:, 4]

        patches = []
        valid_dets = []

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            if w * h < self.min_box_area:
                continue

            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patches.append(patch)
            valid_dets.append([x1, y1, x2, y2, score])

        if not patches:
            return np.array([]), [], []

        # 개별 처리된 특징 추출
        features = self.reid.extract_features(patches)
        return features, valid_dets, [det[4] for det in valid_dets]

    def update_tracks(self, camera_id, features, dets, confs):
        self.frame_count += 1

        if camera_id not in self.camera_trackers:
            self.camera_trackers[camera_id] = CameraTracker(camera_id)

        tracker = self.camera_trackers[camera_id]
        tracker.cleanup(self.frame_count)

        if not features.size or not tracker.tracks:
            for feat, det, conf in zip(features, dets, confs):
                track_id = tracker.next_id
                tracker.tracks[track_id] = Track(feat, det, track_id)
                tracker.next_id += 1
            return

        track_features = []
        track_ids = []

        for track_id, track in tracker.tracks.items():
            if track.state != 'deleted':
                track_features.append(track.get_feature())
                track_ids.append(track_id)

        if not track_features:
            return

        track_features = np.array(track_features)
        cost_matrix = 1 - np.dot(features, track_features.T)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= (1 - self.cos_threshold):
                track_id = track_ids[c]
                tracker.tracks[track_id].update(features[r], dets[r])
                matched.add(r)

        for i, (feat, det) in enumerate(zip(features, dets)):
            if i not in matched:
                track_id = tracker.next_id
                tracker.tracks[track_id] = Track(feat, det, track_id)
                tracker.next_id += 1

        for track_id in list(tracker.tracks.keys()):
            track = tracker.tracks[track_id]
            track.time_since_update += 1
            if track.time_since_update > tracker.max_age:
                del tracker.tracks[track_id]

    def save_tracking_results(self, frames):
        """각 카메라의 global_id에 해당하는 프레임 저장"""
        tracking_data = []
        for camera_id, frame in enumerate(frames):
            tracker = self.camera_trackers.get(camera_id)
            output_path = os.path.join(self.output_dir, f"{camera_id:04d}",f"fresh.bmp")
            cv2.imwrite(output_path, frame)
            if not tracker:
                continue
            for track_id, track in tracker.tracks.items():
                if track.state == 'confirmed':
                    # 바운딩 박스 그리기
                    frame_copy = frame.copy()
                    x1, y1, x2, y2 = map(int, track.bbox[:-1])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ID 표시
                    cv2.putText(frame_copy, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 프레임 저장
                    output_path = os.path.join(self.output_dir, f"{camera_id:04d}",f"{track_id:010d}.bmp")
                    cv2.imwrite(output_path, frame_copy)

                    # JSON 데이터 기록
                    tracking_data.append({
                        "camera_id": camera_id,
                        "global_id": track_id,
                        "bbox": track.bbox[:-1],
                        "frame_path": output_path
                    })

        # JSON 파일로 저장
        with open(os.path.join(self.output_dir, "tracking_data.json"), "w") as f:
            json.dump(tracking_data, f, indent=4)
    def open_video_stream(self, url, retries=3, delay=0.1):
        tcap = cv2.VideoCapture(url)
        attempt = 0
        while not tcap.isOpened() and attempt < retries:
            print(f"Retry {attempt + 1}/{retries}: Trying to open {url}")
            time.sleep(delay)
            tcap = cv2.VideoCapture(url)
            attempt += 1
        if not tcap.isOpened():
            print(f"Failed to open stream after {retries} attempts: {url}")
        return tcap
    
    def run(self, video_paths):
        for i in video_paths:
            cam_id = i.split("/")[-1]
            tmp = cam_id  # 결과 저장 디렉터리
            tmp_output_dir=os.path.join(self.output_dir, tmp)
            os.makedirs(tmp_output_dir, exist_ok=True)
        #caps = [cv2.VideoCapture(p) for p in video_paths]
        caps = [self.open_video_stream(p) for p in video_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Error: Could not open all video files")
            return

        frame_count = 0
        start_time = time.time()
        fps_history = deque(maxlen=30)  # FPS 히스토리 저장용

        try:
            while True:
                frame_start_time = time.time()

                # 각 카메라에서 프레임 및 타임스탬프 읽기
                rets_frames = [(cap.read(), time.time()) for cap in caps]
                frames, timestamps = [], []

                for (ret, frame), timestamp in rets_frames:
                    if not ret:
                        print("not_ret")
                        break
                    frames.append(frame)
                    timestamps.append(timestamp)

                # 동기화 기준 시간 계산
                base_time = min(timestamps)
                threshold = 0.05  # 50ms 이내로 동기화
                synced_frames = [
                    frame for frame, ts in zip(frames, timestamps) if abs(ts - base_time) < threshold
                ]
                if len(synced_frames) < len(caps):  # 동기화 실패
                    print("Frame dropped for synchronization")
                    continue

                # 각 카메라 프레임 처리
                for camera_id, frame in enumerate(synced_frames):
                    features, dets, confs = self.process_frame(frame, camera_id)
                    if len(features) > 0:
                        self.update_tracks(camera_id, features, dets, confs)
                # 결과 저장
                self.save_tracking_results(synced_frames)

                frame_count += 1
                frame_time = time.time() - frame_start_time
                fps = 1 / frame_time
                fps_history.append(fps)
                if frame_count % 30 == 0:
                    avg_fps = sum(fps_history) / len(fps_history)
                    print(f"Frame: {frame_count}, Average FPS: {avg_fps:.2f}")

        finally:
            elapsed_time = time.time() - start_time
            print("\nProcessing Summary:")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time elapsed: {elapsed_time:.2f} seconds")
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")

            for cap in caps:
                cap.release()
def shutdown_proc(proc):
        if proc is None:
            return

        pid = proc.pid
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        proc.terminate()

def load_video_paths(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)  # YAML 파일을 파싱
    return config.get('video_path', [])

@do.command()
def main(cfg):
    reid_model_path = "cfg/quantized_reid_uint8.onnx"
    yolo_cfg_path = "cfg/yolov8s.yaml"
    video_paths = load_video_paths(cfg)

    if not os.path.exists(reid_model_path):
        raise FileNotFoundError(f"ReID model not found: {reid_model_path}")

    if not os.path.exists(yolo_cfg_path):
        raise FileNotFoundError(f"YOLO config not found: {yolo_cfg_path}")

    print("\nInitializing Multi-Camera Tracking System...")
    print(f"YOLO Config: {yolo_cfg_path}")
    print(f"ReID Model: {reid_model_path}")
    print(f"Processing videos: {len(video_paths)} cameras")

    try:
        tracker = MultiCameraTracker(
            reid_model_path=reid_model_path,
            yolo_cfg_path=yolo_cfg_path
        )
        main_thread=threading.Thread(target=tracker.run, args=([video_paths]))
        print("\nStarting tracking pipeline...")
        main_thread.start()
        stream_process = None
        stream_process = subprocess.Popen(
            ["python", "tools/stream.py", cfg]
        )
        main_thread.join()
        time.sleep(10)
        shutdown_proc(stream_process)

    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    do()