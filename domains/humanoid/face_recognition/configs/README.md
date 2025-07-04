# Configs - 얼굴인식 설정 모듈

## 📋 개요

이 폴더는 얼굴인식 기능의 모든 설정 파일들을 포함합니다. 모델 설정, 서비스 설정, 성능 설정 등 다양한 환경과 요구사항에 맞는 설정을 관리합니다.

## 🏗️ 폴더 구조

```
configs/
├── __init__.py                    # 설정 패키지 초기화
├── README.md                      # 이 파일
├── face_recognition.yaml          # 기본 얼굴인식 설정
├── face_detection.yaml            # 얼굴 검출 설정
├── face_recognition_model.yaml    # 얼굴 인식 모델 설정
├── performance.yaml               # 성능 최적화 설정
├── development.yaml               # 개발 환경 설정
└── production.yaml                # 프로덕션 환경 설정
```

## 🔍 포함된 설정들

### 1. Face Recognition Config (기본 설정)
- **파일**: `face_recognition.yaml`
- **목적**: 얼굴인식 기능의 기본 설정
- **포함 내용**: 검출/인식 임계값, 데이터베이스 설정, 로깅 설정

### 2. Face Detection Config (검출 설정)
- **파일**: `face_detection.yaml`
- **목적**: 얼굴 검출 모델의 세부 설정
- **포함 내용**: 모델 경로, 입력 크기, 신뢰도 임계값

### 3. Face Recognition Model Config (인식 모델 설정)
- **파일**: `face_recognition_model.yaml`
- **목적**: 얼굴 인식 모델의 세부 설정
- **포함 내용**: 모델 경로, 특징 벡터 차원, 거리 측정 방법

### 4. Performance Config (성능 설정)
- **파일**: `performance.yaml`
- **목적**: 성능 최적화 관련 설정
- **포함 내용**: 배치 크기, GPU 설정, 메모리 최적화

### 5. Environment Configs (환경별 설정)
- **개발 환경**: `development.yaml`
- **프로덕션 환경**: `production.yaml`
- **목적**: 환경별 차별화된 설정

## 🚀 사용법

### 설정 로드
```python
from domains.humanoid.face_recognition.configs import load_config

# 기본 설정 로드
config = load_config('face_recognition.yaml')

# 환경별 설정 로드
dev_config = load_config('development.yaml')
prod_config = load_config('production.yaml')

# 설정 값 사용
confidence_threshold = config['detection']['confidence_threshold']
model_path = config['models']['face_detection']['path']
```

### 설정 검증
```python
from domains.humanoid.face_recognition.configs import validate_config

# 설정 유효성 검사
is_valid, errors = validate_config(config)
if not is_valid:
    print(f"설정 오류: {errors}")
```

### 설정 병합
```python
from domains.humanoid.face_recognition.configs import merge_configs

# 기본 설정과 환경 설정 병합
base_config = load_config('face_recognition.yaml')
env_config = load_config('development.yaml')
final_config = merge_configs(base_config, env_config)
```

## 🔧 설정 파일 구조

### 기본 얼굴인식 설정 (face_recognition.yaml)
```yaml
# 얼굴인식 기본 설정
version: "1.0.0"
description: "얼굴인식 기능 기본 설정"

# 검출 설정
detection:
  confidence_threshold: 0.5
  min_face_size: 80
  max_faces: 10
  nms_threshold: 0.4

# 인식 설정
recognition:
  confidence_threshold: 0.6
  distance_threshold: 0.6
  embedding_dim: 512
  distance_metric: "cosine"

# 모델 설정
models:
  face_detection:
    path: "models/weights/face_detection_retinaface.onnx"
    input_size: [640, 640]
    providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
  
  face_recognition:
    path: "models/weights/face_recognition_arcface.onnx"
    input_size: [112, 112]
    providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 데이터베이스 설정
database:
  path: "data/face_database.json"
  auto_save: true
  backup_interval: 3600
  max_faces: 10000

# 로깅 설정
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "data/logs/face_recognition.log"
  max_size: 10MB
  backup_count: 5

# 성능 설정
performance:
  target_fps: 30
  enable_monitoring: true
  memory_limit: 2GB
  batch_size: 1
```

### 얼굴 검출 설정 (face_detection.yaml)
```yaml
# 얼굴 검출 모델 설정
model:
  name: "RetinaFace"
  version: "1.0.0"
  path: "models/weights/face_detection_retinaface.onnx"
  input_size: [640, 640]
  output_names: ["output0", "output1", "output2"]

# 추론 설정
inference:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  min_face_size: 80
  max_faces: 10
  input_format: "BGR"
  output_format: "RGB"

# 하드웨어 설정
hardware:
  providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
  gpu_memory_fraction: 0.8
  enable_fp16: true
  optimization_level: "ORT_ENABLE_ALL"

# 후처리 설정
postprocessing:
  enable_landmarks: true
  enable_pose: false
  bbox_format: "xyxy"
  normalize_coordinates: true
```

### 성능 설정 (performance.yaml)
```yaml
# 성능 최적화 설정
optimization:
  target_fps: 30
  max_latency_ms: 100
  memory_limit_mb: 2048
  enable_caching: true
  cache_size: 1000

# 배치 처리 설정
batch_processing:
  enabled: true
  batch_size: 4
  max_queue_size: 100
  timeout_seconds: 5

# GPU 최적화
gpu_optimization:
  enable_tensorrt: false
  enable_fp16: true
  memory_pool_size: 1024
  enable_graph_optimization: true

# 메모리 관리
memory_management:
  enable_garbage_collection: true
  gc_threshold: 0.8
  max_image_size: [1920, 1080]
  enable_compression: false

# 모니터링 설정
monitoring:
  enable_performance_monitoring: true
  metrics_interval: 5
  enable_alerting: true
  alert_thresholds:
    fps_min: 15
    memory_max_mb: 1500
    latency_max_ms: 150
```

## 📊 환경별 설정

### 개발 환경 설정 (development.yaml)
```yaml
# 개발 환경 설정
environment: "development"

# 디버깅 설정
debug:
  enabled: true
  log_level: "DEBUG"
  enable_profiling: true
  save_intermediate_results: true

# 테스트 설정
testing:
  enable_mock_models: true
  test_data_path: "tests/data"
  enable_performance_tests: true

# 개발 도구 설정
development_tools:
  enable_hot_reload: true
  enable_auto_formatting: true
  enable_linting: true

# 로깅 설정 (개발용)
logging:
  level: "DEBUG"
  console_output: true
  file_output: true
  enable_traceback: true
```

### 프로덕션 환경 설정 (production.yaml)
```yaml
# 프로덕션 환경 설정
environment: "production"

# 보안 설정
security:
  enable_encryption: true
  enable_audit_logging: true
  data_retention_days: 30
  enable_gdpr_compliance: true

# 성능 설정 (프로덕션용)
performance:
  target_fps: 30
  enable_gpu: true
  memory_limit_mb: 4096
  enable_load_balancing: true

# 모니터링 설정 (프로덕션용)
monitoring:
  enable_metrics_collection: true
  enable_alerting: true
  enable_health_checks: true
  metrics_endpoint: "/metrics"

# 로깅 설정 (프로덕션용)
logging:
  level: "INFO"
  enable_structured_logging: true
  enable_log_rotation: true
  log_retention_days: 90
```

## 🔗 의존성

### 내부 의존성
- `common/config.py`: 공통 설정 관리
- `../models/`: 모델 설정 참조
- `../services/`: 서비스 설정 참조

### 외부 의존성
```python
# requirements.txt
pyyaml>=6.0
jsonschema>=4.0.0
```

## 🧪 설정 검증

### 설정 스키마 검증
```python
from jsonschema import validate

# 설정 스키마 정의
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["version", "detection", "recognition"],
    "properties": {
        "version": {"type": "string"},
        "detection": {
            "type": "object",
            "required": ["confidence_threshold"],
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            }
        }
    }
}

# 설정 검증
def validate_config(config):
    """설정 유효성 검사"""
    try:
        validate(instance=config, schema=CONFIG_SCHEMA)
        return True, None
    except Exception as e:
        return False, str(e)
```

### 설정 값 검증
```python
def validate_config_values(config):
    """설정 값 검증"""
    errors = []
    
    # 신뢰도 임계값 검증
    if not (0.0 <= config['detection']['confidence_threshold'] <= 1.0):
        errors.append("검출 신뢰도 임계값은 0.0과 1.0 사이여야 합니다")
    
    # 모델 파일 존재 여부 검증
    model_path = config['models']['face_detection']['path']
    if not os.path.exists(model_path):
        errors.append(f"모델 파일이 존재하지 않습니다: {model_path}")
    
    return errors
```

## 🔧 개발 가이드

### 새로운 설정 추가
1. **설정 파일 생성**: `new_config.yaml` 파일 생성
2. **스키마 정의**: 설정 스키마에 새로운 설정 추가
3. **검증 로직**: 설정 값 검증 로직 추가
4. **문서화**: 설정 파일별 문서화

### 설정 확장
```python
# 환경 변수 기반 설정
import os

def load_environment_config():
    """환경 변수 기반 설정 로드"""
    return {
        'database': {
            'path': os.getenv('FACE_DB_PATH', 'data/face_database.json'),
            'max_faces': int(os.getenv('FACE_DB_MAX_FACES', '10000'))
        },
        'performance': {
            'target_fps': int(os.getenv('TARGET_FPS', '30')),
            'memory_limit': os.getenv('MEMORY_LIMIT', '2GB')
        }
    }
```

## 🐛 문제 해결

### 일반적인 설정 문제들

#### 1. 설정 파일을 찾을 수 없음
```python
# 해결 방법
def safe_load_config(config_name):
    """안전한 설정 로드"""
    config_path = f"configs/{config_name}.yaml"
    if not os.path.exists(config_path):
        # 기본 설정 사용
        config_path = "configs/face_recognition.yaml"
    
    return load_config(config_path)
```

#### 2. 설정 값이 유효하지 않음
```python
# 해결 방법
def validate_and_fix_config(config):
    """설정 검증 및 수정"""
    # 기본값 정의
    defaults = {
        'detection': {'confidence_threshold': 0.5},
        'recognition': {'confidence_threshold': 0.6}
    }
    
    # 누락된 값에 기본값 적용
    for section, values in defaults.items():
        if section not in config:
            config[section] = {}
        for key, value in values.items():
            if key not in config[section]:
                config[section][key] = value
    
    return config
```

#### 3. 환경별 설정 충돌
```python
# 해결 방법
def resolve_config_conflicts(base_config, env_config):
    """설정 충돌 해결"""
    resolved_config = base_config.copy()
    
    # 환경 설정이 기본 설정을 오버라이드
    for key, value in env_config.items():
        if isinstance(value, dict) and key in resolved_config:
            resolved_config[key].update(value)
        else:
            resolved_config[key] = value
    
    return resolved_config
```

## 📈 설정 모니터링

### 설정 변경 감지
```python
import hashlib
import time

class ConfigMonitor:
    """설정 변경 모니터링"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.last_hash = self._get_config_hash()
        self.last_modified = os.path.getmtime(config_path)
    
    def _get_config_hash(self):
        """설정 파일 해시 계산"""
        with open(self.config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def has_changed(self):
        """설정 변경 여부 확인"""
        current_hash = self._get_config_hash()
        current_modified = os.path.getmtime(self.config_path)
        
        changed = (current_hash != self.last_hash or 
                  current_modified != self.last_modified)
        
        if changed:
            self.last_hash = current_hash
            self.last_modified = current_modified
        
        return changed
```

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [모델 문서](../models/README.md)
- [서비스 문서](../services/README.md)
- [Humanoid 도메인 문서](../../README.md)
- [프로젝트 전체 문서](../../../../README.md)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일 확인
2. 상위 폴더의 README.md 확인
3. 설정 파일 예시 참조
4. 개발팀에 문의

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 