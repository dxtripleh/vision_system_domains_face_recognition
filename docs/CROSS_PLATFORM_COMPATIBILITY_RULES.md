# 크로스 플랫폼 호환성 및 하드웨어 최적화 규칙

## 📋 개요

비전 시스템은 다음 환경에서 동일하게 작동해야 합니다:
- **개발 환경**: Windows (현재 노트북 - 사양 제한)
- **훈련 환경**: Ubuntu + RTX5090 (향후)
- **배포 환경**: NVIDIA Jetson (임베디드)

## 🖥️ 하드웨어 사양별 최적화

### 현재 노트북 환경 (사양 제한)
```python
# 저사양 노트북 최적화 설정
LOW_SPEC_CONFIG = {
    "model_size": "small",           # YOLOv8n, MobileNet 등 경량 모델
    "input_resolution": (640, 480),  # 낮은 해상도
    "batch_size": 1,                 # 배치 크기 최소화
    "precision": "fp32",             # CPU 호환 정밀도
    "device": "cpu",                 # CPU 전용
    "workers": 2,                    # 워커 수 제한
    "memory_limit": "2GB"            # 메모리 사용량 제한
}

def get_optimal_config():
    """현재 하드웨어에 최적화된 설정 반환"""
    import psutil
    import platform
    
    # 시스템 사양 확인
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    system = platform.system().lower()
    
    # GPU 사용 가능성 확인
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        else:
            gpu_memory = 0
    except:
        gpu_memory = 0
    
    # 사양별 설정 결정
    if gpu_available and gpu_memory >= 8:
        # 고사양 GPU (RTX5090 등)
        return {
            "model_size": "large",
            "input_resolution": (1280, 720),
            "batch_size": 8,
            "precision": "fp16",
            "device": "cuda",
            "workers": min(cpu_count, 8),
            "memory_limit": f"{gpu_memory-1}GB"
        }
    elif gpu_available and gpu_memory >= 4:
        # 중사양 GPU
        return {
            "model_size": "medium", 
            "input_resolution": (800, 600),
            "batch_size": 4,
            "precision": "fp16",
            "device": "cuda",
            "workers": min(cpu_count, 4),
            "memory_limit": f"{gpu_memory-1}GB"
        }
    else:
        # CPU 전용 (현재 노트북)
        return {
            "model_size": "small",
            "input_resolution": (640, 480),
            "batch_size": 1,
            "precision": "fp32", 
            "device": "cpu",
            "workers": min(cpu_count // 2, 2),
            "memory_limit": f"{memory_gb//2}GB"
        }
```

### RTX5090 환경 (향후 훈련용)
```python
# 고사양 GPU 최적화 설정
HIGH_SPEC_CONFIG = {
    "model_size": "large",            # YOLOv8x, ResNet152 등
    "input_resolution": (1280, 720),  # 고해상도
    "batch_size": 16,                 # 대용량 배치
    "precision": "fp16",              # 혼합 정밀도
    "device": "cuda",                 # GPU 가속
    "workers": 8,                     # 멀티프로세싱
    "memory_limit": "20GB",           # 대용량 메모리
    "tensorrt": True,                 # TensorRT 최적화
    "amp": True                       # Automatic Mixed Precision
}
```

### Jetson 환경 (배포용)
```python
# Jetson 최적화 설정
JETSON_CONFIG = {
    "model_size": "medium",           # 균형잡힌 크기
    "input_resolution": (640, 480),   # 실시간 처리 고려
    "batch_size": 1,                  # 실시간 단일 프레임
    "precision": "fp16",              # Jetson GPU 최적화
    "device": "cuda",                 # Jetson GPU
    "workers": 4,                     # Jetson 코어 수
    "memory_limit": "4GB",            # Jetson 메모리 제한
    "tensorrt": True,                 # TensorRT 가속 필수
    "power_mode": "MAX_N"             # 최대 성능 모드
}
```

## 🌐 크로스 플랫폼 경로 처리

### 필수 규칙
```python
import os
from pathlib import Path

# ❌ 절대 금지 - 플랫폼별 경로
model_path = "C:\\models\\model.onnx"      # Windows 전용
model_path = "/home/user/models/model.onnx" # Linux 전용
model_path = "models\\weights\\model.onnx"  # 백슬래시

# ✅ 필수 - 크로스 플랫폼 경로
model_path = os.path.join("models", "weights", "model.onnx")
config_path = Path("config") / "face_recognition.yaml"

# 프로젝트 루트 기준 절대 경로
project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "weights" / "model.onnx"
```

### 환경별 설정 파일
```yaml
# config/environments/development.yaml (Windows 노트북)
hardware:
  device: "cpu"
  batch_size: 1
  workers: 2
  memory_limit: "2GB"

models:
  face_detection: "face_detection_yolov8n_20250703.onnx"
  face_recognition: "face_recognition_mobilefacenet_20250703.onnx"

camera:
  backend: "dshow"  # Windows DirectShow
  resolution: [640, 480]
  fps: 15

# config/environments/training.yaml (Ubuntu RTX5090) 
hardware:
  device: "cuda"
  batch_size: 16
  workers: 8
  memory_limit: "20GB"

models:
  face_detection: "face_detection_yolov8x_20250703.onnx"
  face_recognition: "face_recognition_arcface_r100_20250703.onnx"

camera:
  backend: "v4l2"  # Linux V4L2
  resolution: [1280, 720]
  fps: 30

# config/environments/production.yaml (Jetson)
hardware:
  device: "cuda"
  batch_size: 1
  workers: 4
  memory_limit: "4GB"
  tensorrt: true

models:
  face_detection: "face_detection_yolov8s_jetson_20250703.onnx"
  face_recognition: "face_recognition_mobilefacenet_jetson_20250703.onnx"

camera:
  backend: "v4l2"
  resolution: [640, 480]
  fps: 30
  buffer_size: 1  # 지연 최소화
```

## 🤖 ONNX 런타임 플랫폼별 최적화

### 자동 프로바이더 선택
```python
import onnxruntime as ort
import platform

def create_optimized_session(model_path: str, config: dict):
    """플랫폼별 최적화된 ONNX 세션 생성"""
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    system = platform.system().lower()
    device = config.get("device", "cpu")
    
    # 프로바이더 우선순위 설정
    providers = []
    
    if system == "linux" and is_jetson():
        # Jetson: TensorRT 최우선
        if config.get("tensorrt", False):
            providers.append(('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './tensorrt_cache'
            }))
        providers.extend([
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider'
        ])
        
    elif device == "cuda" and torch.cuda.is_available():
        # Windows/Ubuntu GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        if gpu_memory > 8 * 1024**3:  # 8GB 이상 (RTX5090)
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': int(gpu_memory * 0.8),  # 80% 사용
                'cudnn_conv_algo_search': 'HEURISTIC'
            }))
        else:
            providers.append(('CUDAExecutionProvider', {'device_id': 0}))
            
        providers.append('CPUExecutionProvider')
        
    else:
        # CPU 전용 (현재 노트북)
        providers = ['CPUExecutionProvider']
        session_options.intra_op_num_threads = config.get("workers", 2)
        session_options.inter_op_num_threads = 1
        
        # 메모리 제한 설정
        session_options.enable_mem_pattern = False
        session_options.enable_cpu_mem_arena = False
    
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers
    )

def is_jetson():
    """Jetson 플랫폼 감지"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "jetson" in f.read().lower()
    except:
        return False
```

## 📷 카메라 백엔드 호환성

### 플랫폼별 카메라 설정
```python
def create_platform_camera(camera_id: int = 0, config: dict = None):
    """플랫폼별 최적화된 카메라 생성"""
    
    if config is None:
        config = get_optimal_config()
    
    system = platform.system().lower()
    
    # 플랫폼별 백엔드 선택
    if system == "windows":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
    elif system == "linux":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        
        if is_jetson():
            # Jetson 특화 최적화
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get("buffer_size", 1))
        
    else:
        cap = cv2.VideoCapture(camera_id)
    
    # 공통 설정
    resolution = config.get("resolution", [640, 480])
    fps = config.get("fps", 15)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera_id} 열기 실패")
    
    return cap
```

## 💾 메모리 및 성능 최적화

### 저사양 환경 최적화
```python
class LowSpecOptimizer:
    """저사양 환경 최적화 관리자"""
    
    def __init__(self, config: dict):
        self.config = config
        self.memory_usage = []
        
    def optimize_model_loading(self, model_path: str):
        """모델 로딩 최적화"""
        
        # 메모리 사용량 모니터링
        import psutil
        process = psutil.Process()
        
        before_memory = process.memory_info().rss / 1024**2  # MB
        
        # 경량 모델 선택
        if self.config["device"] == "cpu":
            # CPU용 최적화
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = False
            session_options.enable_cpu_mem_arena = False
            session_options.intra_op_num_threads = self.config["workers"]
            
        after_memory = process.memory_info().rss / 1024**2
        memory_used = after_memory - before_memory
        
        if memory_used > 1000:  # 1GB 이상
            print(f"경고: 모델이 {memory_used:.1f}MB 메모리 사용 중")
    
    def optimize_frame_processing(self, frame: np.ndarray):
        """프레임 처리 최적화"""
        
        # 해상도 다운스케일 (저사양 환경)
        if self.config["device"] == "cpu":
            target_resolution = self.config["input_resolution"]
            if frame.shape[:2] != target_resolution[::-1]:
                frame = cv2.resize(frame, target_resolution)
        
        return frame
    
    def cleanup_memory(self):
        """메모리 정리"""
        import gc
        gc.collect()
        
        if self.config["device"] == "cuda":
            import torch
            torch.cuda.empty_cache()
```

## 🔧 도메인 모듈 템플릿 (크로스 플랫폼)

### domains/factory/defect_detection/ 예시
```python
# domains/factory/defect_detection/model.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
공장 불량 검출 모델 (크로스 플랫폼 호환)

ONNX 기반 YOLOv8 불량 검출 모델을 로딩하고 추론을 수행합니다.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import cv2
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from common.config import load_config
from common.logging import get_logger

class DefectDetectionModel:
    """YOLOv8 기반 불량 검출 모델 (크로스 플랫폼)"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.logger = get_logger(__name__)
        
        # 설정 로드
        if config is None:
            config = load_config("factory_defect_detection")
        self.config = config
        
        # 플랫폼별 최적화 설정
        self.hardware_config = get_optimal_config()
        
        # 모델 경로 설정 (크로스 플랫폼)
        if model_path is None:
            model_path = self._get_default_model_path()
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # ONNX 세션 생성
        self.session = create_optimized_session(str(self.model_path), self.hardware_config)
        
        # 클래스 정보
        self.classes = config.get("classes", ["good", "defect"])
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        self.logger.info(f"모델 로드 완료: {self.model_path}")
        self.logger.info(f"하드웨어 설정: {self.hardware_config['device']}")
    
    def _get_default_model_path(self) -> Path:
        """기본 모델 경로 (크로스 플랫폼)"""
        model_size = self.hardware_config["model_size"]
        
        model_name = f"defect_detection_yolov8{model_size[0]}_factory_20250703.onnx"
        return project_root / "models" / "weights" / model_name
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """이미지에서 불량 검출"""
        
        # 전처리
        processed_image = self._preprocess(image)
        
        # 추론 실행
        outputs = self.session.run(None, {"images": processed_image})
        
        # 후처리
        detections = self._postprocess(outputs[0], image.shape[:2])
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (크로스 플랫폼 호환)"""
        
        # 입력 크기 조정 (하드웨어 사양에 따라)
        input_size = self.hardware_config["input_resolution"]
        
        # 비율 유지하며 리사이즈
        h, w = image.shape[:2]
        scale = min(input_size[0] / w, input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # 패딩 추가
        top = (input_size[1] - new_h) // 2
        bottom = input_size[1] - new_h - top
        left = (input_size[0] - new_w) // 2
        right = input_size[0] - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # 정규화 및 차원 변경
        normalized = padded.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # 배치 차원 추가
        
        return batched
    
    def _postprocess(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """출력 후처리"""
        
        detections = []
        
        # NMS 적용 및 좌표 변환
        # ... (YOLO 후처리 로직)
        
        return detections

# domains/factory/defect_detection/run.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
공장 불량 검출 실행 스크립트 (크로스 플랫폼)

USB 카메라 또는 이미지 파일을 입력으로 받아 실시간 불량 검출을 수행합니다.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from common.config import load_config
from common.logging import setup_logging
from .model import DefectDetectionModel

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="공장 불량 검출 시스템")
    parser.add_argument("--source", type=str, default="0", 
                       help="입력 소스 (카메라 ID, 파일 경로, RTSP URL)")
    parser.add_argument("--model", type=str, help="모델 파일 경로")
    parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--show", action="store_true", help="결과 화면 표시")
    parser.add_argument("--save", action="store_true", help="결과 저장")
    parser.add_argument("--headless", action="store_true", help="GUI 없이 실행")
    return parser.parse_args()

def verify_hardware_connection():
    """하드웨어 연결 상태 확인 (시뮬레이션 방지)"""
    import os
    
    # 시뮬레이션 모드 금지
    if os.environ.get("USE_SIMULATION", "False").lower() == "true":
        raise RuntimeError("시뮬레이션 모드는 금지되어 있습니다.")
    
    # 카메라 연결 확인
    cap = create_platform_camera(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise RuntimeError("카메라에서 프레임을 읽을 수 없습니다.")
    
    return True

def handle_keyboard_input():
    """표준 키보드 단축키 처리"""
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):      # q: 종료
        return 'quit'
    elif key == ord('s'):    # s: 현재 프레임 저장
        return 'save_frame'
    elif key == ord('p'):    # p: 일시 정지/재생
        return 'toggle_pause'
    elif key == ord('r'):    # r: 녹화 시작/중지
        return 'toggle_record'
    
    return None

def visualize_results(frame: np.ndarray, detections: list, config: dict) -> np.ndarray:
    """검출 결과 시각화 (크로스 플랫폼)"""
    
    result_frame = frame.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_name = detection["class"]
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if class_name == "good" else (0, 0, 255)
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 그리기
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(result_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_frame

def main():
    """메인 함수"""
    args = parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("공장 불량 검출 시스템 시작")
        
        # 하드웨어 연결 검증 (필수)
        verify_hardware_connection()
        
        # 설정 로드
        config = load_config("factory_defect_detection")
        hardware_config = get_optimal_config()
        
        logger.info(f"하드웨어 설정: {hardware_config}")
        
        # 모델 초기화
        model = DefectDetectionModel(model_path=args.model, config=config)
        
        # 입력 소스 설정
        if args.source.isdigit():
            cap = create_platform_camera(int(args.source), hardware_config)
        else:
            cap = cv2.VideoCapture(args.source)
        
        if not cap.isOpened():
            raise RuntimeError(f"입력 소스를 열 수 없습니다: {args.source}")
        
        # 성능 측정
        fps_counter = 0
        start_time = time.time()
        
        # 메인 처리 루프
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 불량 검출
                detections = model.predict(frame)
                
                # 결과 로깅
                defect_count = len([d for d in detections if d["class"] == "defect"])
                if defect_count > 0:
                    logger.warning(f"불량 검출: {defect_count}개")
                
                # 결과 시각화
                if args.show or not args.headless:
                    display_frame = visualize_results(frame, detections, config)
                    
                    # FPS 표시
                    fps_counter += 1
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps = fps_counter / elapsed
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Factory Defect Detection", display_frame)
                
                # 결과 저장
                if args.save:
                    timestamp = int(time.time())
                    output_dir = project_root / "data" / "runtime" / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"defect_result_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), display_frame)
            
            # 키보드 입력 처리
            if not args.headless:
                action = handle_keyboard_input()
                if action == 'quit':
                    break
                elif action == 'toggle_pause':
                    paused = not paused
                    logger.info(f"{'일시정지' if paused else '재생'}")
                elif action == 'save_frame':
                    timestamp = int(time.time())
                    output_dir = project_root / "data" / "runtime" / "output"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    logger.info(f"프레임 저장: {output_path}")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logger.info("시스템 종료 완료")

if __name__ == "__main__":
    main()

# domains/factory/defect_detection/test_model.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
불량 검출 모델 단위 테스트

예제 이미지를 사용하여 모델의 기본 기능을 테스트합니다.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .model import DefectDetectionModel

class TestDefectDetectionModel(unittest.TestCase):
    """불량 검출 모델 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.model = DefectDetectionModel()
        
        # 테스트용 이미지 생성 (실제로는 예제 이미지 사용)
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_model_loading(self):
        """모델 로딩 테스트"""
        self.assertIsNotNone(self.model.session)
        self.assertGreater(len(self.model.classes), 0)
    
    def test_prediction(self):
        """예측 기능 테스트"""
        detections = self.model.predict(self.test_image)
        self.assertIsInstance(detections, list)
    
    def test_cross_platform_path(self):
        """크로스 플랫폼 경로 테스트"""
        model_path = self.model.model_path
        self.assertTrue(model_path.exists())

if __name__ == "__main__":
    unittest.main()
```

## 📊 성능 벤치마크 가이드

### 환경별 예상 성능
```yaml
# 현재 노트북 (CPU)
performance_expectation:
  fps: 5-10
  latency: 200-500ms
  accuracy: 동일
  memory: 1-2GB

# RTX5090 (GPU)
performance_expectation:
  fps: 60-120
  latency: 8-15ms  
  accuracy: 동일
  memory: 4-8GB

# Jetson (임베디드)
performance_expectation:
  fps: 15-30
  latency: 30-60ms
  accuracy: 동일
  memory: 2-4GB
```

## ⚠️ 주의사항

1. **현재 노트북 개발 시**:
   - CPU 전용으로 개발하되, GPU 코드 경로도 유지
   - 경량 모델 우선 사용 (YOLOv8n, MobileNet)
   - 배치 크기는 1로 고정
   
2. **RTX5090 전환 시**:
   - 설정 파일만 변경하면 자동 GPU 활용
   - 대용량 모델 및 배치 처리 가능
   
3. **Jetson 배포 시**:
   - TensorRT 최적화 필수
   - 전력 관리 고려 필요 