# Config - 설정 파일 관리

## 📋 개요

`config/` 폴더는 프로젝트 전체의 설정 파일들을 관리합니다. 각 도메인별 설정, 환경별 설정, 모델 설정 등을 YAML 형식으로 관리하여 유지보수성과 확장성을 높입니다.

## 🏗️ 구조

```
config/
├── __init__.py
├── README.md
├── face_recognition.yaml      # 얼굴인식 설정
├── defect_detection.yaml      # 불량검출 설정
├── development.yaml           # 개발 환경 설정
├── production.yaml            # 운영 환경 설정
└── schemas/                   # 설정 스키마 (향후)
    ├── base_config.yaml
    ├── face_recognition.yaml
    └── defect_detection.yaml
```

## 📁 설정 파일별 설명

### 도메인별 설정 파일

#### `face_recognition.yaml` - 얼굴인식 설정
얼굴인식 기능의 모든 설정을 포함합니다.
```yaml
model:
  path: "models/weights/face_recognition.onnx"
  confidence_threshold: 0.5
  max_faces: 10

camera:
  device_id: 0
  resolution: [640, 480]
  fps: 30

face_database:
  path: "data/domains/humanoid/face_recognition/faces/"
  update_interval: 3600

output:
  save_results: true
  output_dir: "data/domains/humanoid/face_recognition/"
  save_images: true
  save_annotations: true
```

#### `defect_detection.yaml` - 불량검출 설정
불량검출 기능의 모든 설정을 포함합니다.
```yaml
model:
  path: "models/weights/defect_detection.onnx"
  confidence_threshold: 0.5
  max_defects: 50

camera:
  device_id: 0
  resolution: [1920, 1080]
  fps: 30

defect_types:
  - scratch
  - stain
  - crack
  - dent
  - color_variation
  - missing_part

output:
  save_results: true
  output_dir: "data/domains/factory/defect_detection/"
  save_images: true
  save_annotations: true

alerts:
  email_notification: false
  sound_alert: true
  defect_count_threshold: 10
```

### 환경별 설정 파일

#### `development.yaml` - 개발 환경 설정
개발 환경에서 사용하는 설정입니다.
```yaml
# 개발 환경 공통 설정
logging:
  level: "DEBUG"
  file: "data/logs/development.log"

model:
  confidence_threshold: 0.3  # 개발 환경에서는 낮은 임계값

camera:
  resolution: [640, 480]     # 개발 환경에서는 낮은 해상도

debug:
  enabled: true
  save_intermediate_results: true
  verbose_output: true
```

#### `production.yaml` - 운영 환경 설정
운영 환경에서 사용하는 설정입니다.
```yaml
# 운영 환경 공통 설정
logging:
  level: "WARNING"
  file: "data/logs/production.log"

model:
  confidence_threshold: 0.7  # 운영 환경에서는 높은 임계값

camera:
  resolution: [1920, 1080]   # 운영 환경에서는 높은 해상도

debug:
  enabled: false
  save_intermediate_results: false
  verbose_output: false

security:
  enable_encryption: true
  enable_audit_logging: true
```

## 🔧 설정 파일 사용법

### 기본 사용법

```python
from common.config import load_config

# 도메인별 설정 로딩
face_config = load_config('config/face_recognition.yaml')
defect_config = load_config('config/defect_detection.yaml')

# 환경별 설정 로딩
if os.environ.get('ENVIRONMENT') == 'production':
    env_config = load_config('config/production.yaml')
else:
    env_config = load_config('config/development.yaml')
```

### 설정값 접근

```python
from common.config import get_config

# 특정 설정값 가져오기
model_path = get_config('model.path', default='models/default.onnx')
confidence = get_config('model.confidence_threshold', default=0.5)

# 중첩된 설정값 접근
camera_resolution = get_config('camera.resolution', default=[640, 480])
```

### 설정 병합

```python
from common.config import merge_configs

# 기본 설정
base_config = load_config('config/face_recognition.yaml')

# 환경별 설정
env_config = load_config('config/development.yaml')

# 설정 병합 (환경 설정이 기본 설정을 오버라이드)
merged_config = merge_configs(base_config, env_config)
```

## 📋 설정 파일 작성 규칙

### 1. 구조화된 설정
```yaml
# ✅ 좋은 예: 논리적으로 그룹화
model:
  path: "models/weights/model.onnx"
  confidence_threshold: 0.5
  max_detections: 10

camera:
  device_id: 0
  resolution: [640, 480]
  fps: 30

output:
  save_results: true
  output_dir: "data/output/"
```

### 2. 기본값 제공
```yaml
# ✅ 좋은 예: 기본값 포함
model:
  confidence_threshold: 0.5  # 기본값
  max_detections: 10         # 기본값

camera:
  device_id: 0               # 기본 카메라
  resolution: [640, 480]     # 기본 해상도
```

### 3. 주석 사용
```yaml
# 모델 설정
model:
  path: "models/weights/model.onnx"  # 모델 파일 경로
  confidence_threshold: 0.5          # 신뢰도 임계값 (0.0 ~ 1.0)
  max_detections: 10                 # 최대 검출 개수

# 카메라 설정
camera:
  device_id: 0                       # 카메라 ID (0: 기본 카메라)
  resolution: [640, 480]             # 해상도 [width, height]
  fps: 30                           # 프레임레이트
```

## 🔍 설정 검증

### 스키마 기반 검증 (향후)
```yaml
# schemas/face_recognition.yaml
type: object
required: [model, camera, output]
properties:
  model:
    type: object
    required: [path, confidence_threshold]
    properties:
      path:
        type: string
        pattern: "^.*\\.onnx$"
      confidence_threshold:
        type: number
        minimum: 0.0
        maximum: 1.0
```

### 프로그래밍적 검증
```python
def validate_face_recognition_config(config):
    """얼굴인식 설정 검증"""
    required_keys = ['model', 'camera', 'output']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"필수 설정 키 누락: {key}")
    
    # 모델 경로 검증
    model_path = config['model']['path']
    if not model_path.endswith('.onnx'):
        raise ValueError("모델 파일은 .onnx 형식이어야 합니다")
    
    # 신뢰도 임계값 검증
    confidence = config['model']['confidence_threshold']
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("신뢰도 임계값은 0.0 ~ 1.0 범위여야 합니다")
```

## 🚀 환경별 설정 관리

### 환경 변수 사용
```bash
# 환경 변수 설정
export ENVIRONMENT=production
export CAMERA_ID=1
export MODEL_PATH=models/custom_model.onnx

# Python에서 환경 변수 사용
import os
camera_id = int(os.environ.get('CAMERA_ID', 0))
model_path = os.environ.get('MODEL_PATH', 'models/default.onnx')
```

### 동적 설정 생성
```python
def create_dynamic_config():
    """환경에 따른 동적 설정 생성"""
    base_config = load_config('config/face_recognition.yaml')
    
    # 환경 변수로 설정 오버라이드
    if os.environ.get('CAMERA_ID'):
        base_config['camera']['device_id'] = int(os.environ['CAMERA_ID'])
    
    if os.environ.get('MODEL_PATH'):
        base_config['model']['path'] = os.environ['MODEL_PATH']
    
    return base_config
```

## 🔧 설정 파일 템플릿

### 새로운 도메인 설정 파일 생성
```yaml
# config/new_domain.yaml
# 새 도메인 설정 파일 템플릿

# 모델 설정
model:
  path: "models/weights/new_domain.onnx"  # 모델 파일 경로
  confidence_threshold: 0.5               # 신뢰도 임계값
  max_detections: 10                      # 최대 검출 개수

# 카메라 설정
camera:
  device_id: 0                            # 카메라 ID
  resolution: [640, 480]                  # 해상도
  fps: 30                                 # 프레임레이트

# 출력 설정
output:
  save_results: true                      # 결과 저장 여부
  output_dir: "data/domains/new_domain/"  # 출력 디렉토리
  save_images: true                       # 이미지 저장 여부
  save_annotations: true                  # 주석 저장 여부

# 로깅 설정
logging:
  level: "INFO"                           # 로그 레벨
  file: "data/logs/new_domain.log"        # 로그 파일
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 도메인별 특수 설정
domain_specific:
  # 도메인에 특화된 설정들
  feature1: true
  feature2: "value"
  feature3: [1, 2, 3]
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 설정 파일을 찾을 수 없음
```python
# 기본 설정 제공
def load_config_with_default(config_path, default_config=None):
    try:
        return load_config(config_path)
    except FileNotFoundError:
        if default_config:
            logger.warning(f"설정 파일을 찾을 수 없음: {config_path}, 기본값 사용")
            return default_config
        else:
            raise
```

#### 2. 설정값이 잘못된 형식
```python
# 타입 검증
def validate_config_types(config):
    """설정값 타입 검증"""
    if not isinstance(config['model']['confidence_threshold'], (int, float)):
        raise ValueError("confidence_threshold는 숫자여야 합니다")
    
    if not isinstance(config['camera']['resolution'], list):
        raise ValueError("resolution은 리스트여야 합니다")
```

#### 3. 환경별 설정 충돌
```python
# 설정 우선순위 관리
def load_environment_config():
    """환경별 설정 로딩 (우선순위 고려)"""
    # 1. 기본 설정
    config = load_config('config/face_recognition.yaml')
    
    # 2. 환경별 설정 (기본 설정 오버라이드)
    env = os.environ.get('ENVIRONMENT', 'development')
    env_config = load_config(f'config/{env}.yaml')
    
    # 3. 환경 변수 (환경별 설정 오버라이드)
    if os.environ.get('MODEL_PATH'):
        config['model']['path'] = os.environ['MODEL_PATH']
    
    return merge_configs(config, env_config)
```

## 📊 설정 모니터링

### 설정 변경 감지
```python
import time
import hashlib

class ConfigMonitor:
    """설정 파일 변경 감지"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.last_hash = self._get_file_hash()
    
    def _get_file_hash(self):
        """파일 해시 계산"""
        with open(self.config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def has_changed(self):
        """설정 파일 변경 여부 확인"""
        current_hash = self._get_file_hash()
        if current_hash != self.last_hash:
            self.last_hash = current_hash
            return True
        return False
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일을 먼저 확인
2. 설정 파일 문법 확인
3. 환경 변수 설정 확인
4. 프로젝트 루트의 README.md 확인

## 📄 라이선스

이 모듈의 코드는 프로젝트 전체 라이선스를 따릅니다. 