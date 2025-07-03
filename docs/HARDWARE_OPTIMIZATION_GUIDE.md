# 하드웨어 사양별 최적화 가이드

## 📋 개발 환경별 설정

### 🖥️ 현재 노트북 (저사양 - CPU 전용)

#### 최적화 전략
```python
# config/environments/development_low_spec.yaml
hardware:
  device: "cpu"
  batch_size: 1
  workers: 2
  memory_limit: "2GB"
  optimization_level: "speed"  # 속도 우선

models:
  # 경량 모델 우선 사용
  face_detection: "face_detection_yolov8n_20250703.onnx"         # 가장 작은 YOLO
  face_recognition: "face_recognition_mobilefacenet_20250703.onnx" # 모바일 최적화
  
preprocessing:
  input_size: [640, 480]     # 낮은 해상도
  normalize: true
  resize_method: "bilinear"  # 빠른 리사이즈

inference:
  precision: "fp32"          # CPU 호환
  enable_optimizations: true
  thread_pool_size: 2
```

#### 성능 예상치
- **FPS**: 5-15 (모델 크기에 따라)
- **지연시간**: 100-300ms
- **메모리 사용량**: 1-2GB
- **정확도**: GPU 버전과 동일

#### 개발 시 주의사항
```python
# ✅ 저사양 환경 개발 팁
def optimize_for_low_spec():
    """저사양 환경 최적화"""
    
    # 1. 모델 크기 최소화
    model_config = {
        "yolo_version": "n",  # nano 버전
        "input_size": 640,    # 작은 입력 크기
        "nms_threshold": 0.5
    }
    
    # 2. 메모리 사용량 모니터링
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024**2
    
    # 모델 로딩
    # ... 
    
    memory_after = process.memory_info().rss / 1024**2
    if memory_after - memory_before > 1000:  # 1GB 초과
        logger.warning(f"높은 메모리 사용량: {memory_after - memory_before:.1f}MB")
    
    # 3. 프레임 스킵 활용
    frame_skip = 2  # 2프레임마다 1번 처리
    
    return model_config
```

### 🚀 RTX5090 환경 (고사양 - 훈련용)

#### 최적화 전략
```python
# config/environments/training_high_spec.yaml
hardware:
  device: "cuda"
  batch_size: 16
  workers: 8
  memory_limit: "20GB"
  optimization_level: "accuracy"  # 정확도 우선

models:
  # 고성능 모델 사용
  face_detection: "face_detection_yolov8x_20250703.onnx"         # 최대 크기 YOLO
  face_recognition: "face_recognition_arcface_r100_20250703.onnx" # 고정확도 모델

preprocessing:
  input_size: [1280, 720]    # 고해상도
  augmentation: true         # 데이터 증강
  normalize: true
  resize_method: "bicubic"   # 고품질 리사이즈

inference:
  precision: "fp16"          # 혼합 정밀도
  tensorrt: true            # TensorRT 최적화
  enable_optimizations: true
  amp: true                 # Automatic Mixed Precision
```

#### 성능 예상치
- **FPS**: 60-120
- **지연시간**: 8-15ms
- **메모리 사용량**: 8-16GB
- **정확도**: 최고 수준

#### 훈련 시 최적화
```python
# ✅ RTX5090 최적화 설정
def optimize_for_rtx5090():
    """RTX5090 환경 최적화"""
    
    import torch
    
    # 1. CUDA 설정 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # 2. 메모리 최적화
    torch.cuda.empty_cache()
    
    # 3. 멀티 GPU 지원 (필요시)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # 4. 혼합 정밀도 설정
    scaler = torch.cuda.amp.GradScaler()
    
    return {
        "device": "cuda",
        "scaler": scaler,
        "amp_enabled": True,
        "batch_size": 32,  # 대용량 배치
    }
```

### 🤖 Jetson 환경 (임베디드 - 배포용)

#### 최적화 전략
```python
# config/environments/production_jetson.yaml
hardware:
  device: "cuda"
  batch_size: 1
  workers: 4
  memory_limit: "4GB"
  optimization_level: "balanced"  # 균형

models:
  # Jetson 최적화 모델
  face_detection: "face_detection_yolov8s_jetson_20250703.onnx"  # 중간 크기
  face_recognition: "face_recognition_mobilefacenet_jetson_20250703.onnx"

preprocessing:
  input_size: [640, 480]     # 실시간 처리 고려
  normalize: true
  resize_method: "bilinear"

inference:
  precision: "fp16"          # Jetson GPU 최적화
  tensorrt: true            # TensorRT 필수
  enable_optimizations: true
  power_mode: "MAXN"        # 최대 성능 모드
```

#### Jetson 특화 최적화
```python
# ✅ Jetson 환경 최적화
def optimize_for_jetson():
    """Jetson 환경 최적화"""
    
    # 1. Jetson 감지
    def is_jetson():
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    if not is_jetson():
        return None
    
    # 2. 전력 모드 설정
    import subprocess
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", "0"], check=True)  # 최대 성능
        subprocess.run(["sudo", "jetson_clocks"], check=True)        # 클럭 최대화
    except:
        logger.warning("Jetson 성능 모드 설정 실패")
    
    # 3. TensorRT 설정
    tensorrt_config = {
        "workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
        "fp16_mode": True,
        "strict_type_constraints": True,
        "engine_cache": True,
        "cache_path": "./tensorrt_cache"
    }
    
    return tensorrt_config
```

## 🔄 환경별 자동 전환

### 환경 자동 감지
```python
# common/hardware_detector.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하드웨어 환경 자동 감지 및 최적화 설정
"""

import platform
import os
import psutil
from typing import Dict, Optional

class HardwareDetector:
    """하드웨어 환경 자동 감지"""
    
    @staticmethod
    def detect_environment() -> str:
        """현재 환경 감지"""
        
        # Jetson 감지
        if HardwareDetector.is_jetson():
            return "jetson"
        
        # GPU 사용 가능 여부 확인
        gpu_available, gpu_memory = HardwareDetector.detect_gpu()
        
        if gpu_available:
            if gpu_memory >= 16:  # 16GB 이상 (RTX5090 급)
                return "high_spec_gpu"
            elif gpu_memory >= 8:  # 8GB 이상
                return "medium_spec_gpu"
            else:
                return "low_spec_gpu"
        else:
            return "cpu_only"
    
    @staticmethod
    def is_jetson() -> bool:
        """Jetson 플랫폼 감지"""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "jetson" in f.read().lower()
        except:
            return False
    
    @staticmethod
    def detect_gpu() -> tuple[bool, float]:
        """GPU 감지 및 메모리 크기 반환"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, gpu_memory
        except ImportError:
            pass
        
        return False, 0.0
    
    @staticmethod
    def get_optimal_config() -> Dict:
        """환경별 최적 설정 반환"""
        
        env = HardwareDetector.detect_environment()
        system_memory = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        configs = {
            "cpu_only": {
                "device": "cpu",
                "batch_size": 1,
                "workers": min(cpu_count // 2, 2),
                "memory_limit": f"{int(system_memory * 0.5)}GB",
                "model_size": "small",
                "input_resolution": (640, 480),
                "precision": "fp32"
            },
            "low_spec_gpu": {
                "device": "cuda",
                "batch_size": 2,
                "workers": min(cpu_count // 2, 4),
                "memory_limit": "4GB",
                "model_size": "small",
                "input_resolution": (640, 480),
                "precision": "fp16"
            },
            "medium_spec_gpu": {
                "device": "cuda",
                "batch_size": 4,
                "workers": min(cpu_count, 6),
                "memory_limit": "6GB", 
                "model_size": "medium",
                "input_resolution": (800, 600),
                "precision": "fp16"
            },
            "high_spec_gpu": {
                "device": "cuda",
                "batch_size": 16,
                "workers": min(cpu_count, 8),
                "memory_limit": "16GB",
                "model_size": "large",
                "input_resolution": (1280, 720),
                "precision": "fp16",
                "tensorrt": True,
                "amp": True
            },
            "jetson": {
                "device": "cuda",
                "batch_size": 1,
                "workers": 4,
                "memory_limit": "4GB",
                "model_size": "medium",
                "input_resolution": (640, 480),
                "precision": "fp16",
                "tensorrt": True,
                "power_optimization": True
            }
        }
        
        config = configs.get(env, configs["cpu_only"])
        config["environment"] = env
        
        return config

# 전역 설정 로더
def load_hardware_config() -> Dict:
    """하드웨어별 최적 설정 로드"""
    return HardwareDetector.get_optimal_config()
```

### 설정 파일 자동 선택
```python
# common/config.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
환경별 설정 자동 로드
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

from .hardware_detector import load_hardware_config

def load_config(domain: str = "face_recognition") -> Dict:
    """환경에 맞는 설정 자동 로드"""
    
    # 하드웨어 설정 감지
    hardware_config = load_hardware_config()
    environment = hardware_config["environment"]
    
    # 환경별 설정 파일 경로
    config_dir = Path(__file__).parent.parent / "config"
    
    # 기본 설정 로드
    base_config_path = config_dir / f"{domain}.yaml"
    base_config = {}
    if base_config_path.exists():
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}
    
    # 환경별 설정 로드
    env_config_path = config_dir / "environments" / f"{environment}_{domain}.yaml"
    env_config = {}
    if env_config_path.exists():
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f) or {}
    
    # 설정 병합
    merged_config = {**base_config, **env_config}
    merged_config["hardware"] = hardware_config
    
    return merged_config
```

## 📊 성능 모니터링

### 환경별 성능 측정
```python
# common/performance_monitor.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
환경별 성능 모니터링
"""

import time
import psutil
from typing import Dict, List
from collections import deque

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.start_time = None
        
    def start_frame(self):
        """프레임 처리 시작"""
        self.start_time = time.time()
    
    def end_frame(self):
        """프레임 처리 종료 및 메트릭 수집"""
        if self.start_time is None:
            return
        
        # 지연시간 계산
        latency = (time.time() - self.start_time) * 1000  # ms
        self.latency_history.append(latency)
        
        # FPS 계산
        if len(self.latency_history) > 1:
            fps = 1000 / latency if latency > 0 else 0
            self.fps_history.append(fps)
        
        # 메모리 사용량
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        self.memory_history.append(memory_mb)
        
        self.start_time = None
    
    def get_metrics(self) -> Dict:
        """현재 성능 메트릭 반환"""
        import numpy as np
        
        if not self.fps_history:
            return {}
        
        return {
            "fps": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": np.mean(self.fps_history),
                "min": np.min(self.fps_history),
                "max": np.max(self.fps_history)
            },
            "latency_ms": {
                "current": self.latency_history[-1] if self.latency_history else 0,
                "average": np.mean(self.latency_history),
                "p95": np.percentile(self.latency_history, 95),
                "p99": np.percentile(self.latency_history, 99)
            },
            "memory_mb": {
                "current": self.memory_history[-1] if self.memory_history else 0,
                "average": np.mean(self.memory_history),
                "peak": np.max(self.memory_history)
            }
        }
    
    def check_performance_thresholds(self, config: Dict) -> List[str]:
        """성능 임계값 확인"""
        warnings = []
        metrics = self.get_metrics()
        
        # FPS 임계값 확인
        expected_fps = config.get("expected_fps", {})
        current_fps = metrics.get("fps", {}).get("current", 0)
        
        if current_fps < expected_fps.get("min", 5):
            warnings.append(f"낮은 FPS: {current_fps:.1f} < {expected_fps['min']}")
        
        # 지연시간 임계값 확인
        expected_latency = config.get("expected_latency_ms", {})
        current_latency = metrics.get("latency_ms", {}).get("current", 0)
        
        if current_latency > expected_latency.get("max", 1000):
            warnings.append(f"높은 지연시간: {current_latency:.1f}ms > {expected_latency['max']}ms")
        
        # 메모리 임계값 확인
        expected_memory = config.get("expected_memory_mb", {})
        current_memory = metrics.get("memory_mb", {}).get("current", 0)
        
        if current_memory > expected_memory.get("max", 4000):
            warnings.append(f"높은 메모리 사용량: {current_memory:.1f}MB > {expected_memory['max']}MB")
        
        return warnings

# 전역 성능 모니터
performance_monitor = PerformanceMonitor()
```

## 🎯 개발 가이드라인

### 현재 노트북에서 개발 시
1. **모델 선택**: 항상 경량 버전 (YOLOv8n, MobileNet)
2. **해상도**: 640x480 이하 권장
3. **배치 크기**: 1로 고정
4. **정밀도**: FP32 사용
5. **테스트**: CPU에서 정상 작동 확인 후 GPU 환경 테스트

### RTX5090 전환 시
1. **설정 변경**: `environment` 설정만 변경
2. **모델 업그레이드**: 고성능 모델로 자동 전환
3. **배치 처리**: 대용량 배치 활용
4. **TensorRT**: 자동 최적화 적용

### Jetson 배포 시
1. **TensorRT 변환**: 필수 과정
2. **전력 관리**: 성능 모드 설정
3. **실시간 처리**: 단일 프레임 처리 최적화
4. **메모리 관리**: 4GB 제한 고려 