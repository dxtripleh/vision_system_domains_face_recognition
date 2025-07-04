# Common - 범용 유틸리티

## 📋 개요

`common/` 폴더는 프로젝트 전체에서 공통으로 사용되는 유틸리티 함수들과 기본 기능들을 포함합니다. 로깅, 설정 관리, 파일 처리, 하드웨어 감지 등 모든 모듈에서 필요한 기본 기능들을 제공합니다.

## 🏗️ 구조

```
common/
├── __init__.py
├── README.md
├── config.py      # 설정 관리
├── logging.py     # 로깅 시스템
└── utils.py       # 범용 유틸리티
```

## 📁 모듈별 설명

### `config.py` - 설정 관리
- **목적**: 프로젝트 전체의 설정 파일 관리
- **기능**:
  - YAML 설정 파일 로딩
  - 환경별 설정 관리
  - 설정값 검증
  - 기본값 제공

### `logging.py` - 로깅 시스템
- **목적**: 통일된 로깅 시스템 제공
- **기능**:
  - 구조화된 로깅
  - 로그 레벨 관리
  - 파일 및 콘솔 출력
  - 로그 포맷팅

### `utils.py` - 범용 유틸리티
- **목적**: 자주 사용되는 유틸리티 함수들
- **기능**:
  - 하드웨어 감지
  - 경로 처리
  - 시간 관련 함수
  - 데이터 변환

## 🔧 사용법

### 설정 관리 사용법

```python
from common.config import load_config, get_config

# 설정 파일 로딩
config = load_config('config/face_recognition.yaml')

# 특정 설정값 가져오기
model_path = get_config('model.path', default='models/default.onnx')
confidence = get_config('model.confidence_threshold', default=0.5)

# 환경별 설정
if os.environ.get('ENVIRONMENT') == 'production':
    config = load_config('config/production.yaml')
else:
    config = load_config('config/development.yaml')
```

### 로깅 시스템 사용법

```python
from common.logging import setup_logging, get_logger

# 로깅 시스템 초기화
setup_logging(level='INFO', log_file='data/logs/app.log')

# 로거 가져오기
logger = get_logger(__name__)

# 로깅 사용
logger.info("애플리케이션 시작")
logger.debug("디버그 정보")
logger.warning("경고 메시지")
logger.error("오류 발생", exc_info=True)
```

### 유틸리티 함수 사용법

```python
from common.utils import detect_hardware, get_project_root, create_temp_file

# 하드웨어 감지
hardware_info = detect_hardware()
print(f"GPU 사용 가능: {hardware_info['gpu_available']}")
print(f"CPU 코어 수: {hardware_info['cpu_count']}")

# 프로젝트 루트 경로
project_root = get_project_root()
model_path = project_root / "models" / "weights" / "model.onnx"

# 임시 파일 생성
temp_file = create_temp_file(prefix="temp_", suffix=".jpg")
```

## 📋 주요 함수들

### 설정 관리 함수

#### `load_config(config_path: str) -> Dict`
설정 파일을 로딩합니다.
```python
config = load_config('config/face_recognition.yaml')
```

#### `get_config(key: str, default: Any = None) -> Any`
설정값을 가져옵니다.
```python
model_path = get_config('model.path', default='models/default.onnx')
```

#### `save_config(config: Dict, config_path: str)`
설정을 파일에 저장합니다.
```python
save_config(config, 'config/updated_config.yaml')
```

### 로깅 함수

#### `setup_logging(level: str = 'INFO', log_file: str = None)`
로깅 시스템을 초기화합니다.
```python
setup_logging(level='DEBUG', log_file='data/logs/debug.log')
```

#### `get_logger(name: str) -> logging.Logger`
로거를 가져옵니다.
```python
logger = get_logger(__name__)
```

### 유틸리티 함수

#### `detect_hardware() -> Dict`
하드웨어 정보를 감지합니다.
```python
hardware = detect_hardware()
# Returns: {
#     'gpu_available': True,
#     'gpu_name': 'RTX 3080',
#     'cpu_count': 8,
#     'memory_gb': 16,
#     'platform': 'Windows'
# }
```

#### `get_project_root() -> Path`
프로젝트 루트 경로를 반환합니다.
```python
root = get_project_root()
# Returns: Path('/path/to/vision_system')
```

#### `create_temp_file(prefix: str = 'temp_', suffix: str = '.tmp') -> Path`
임시 파일을 생성합니다.
```python
temp_file = create_temp_file(prefix='image_', suffix='.jpg')
```

#### `ensure_directory(path: Union[str, Path])`
디렉토리가 존재하지 않으면 생성합니다.
```python
ensure_directory('data/output/results')
```

## 🔧 설정 파일 예시

### 기본 설정 파일 (config/face_recognition.yaml)
```yaml
model:
  path: "models/weights/face_recognition.onnx"
  confidence_threshold: 0.5
  max_faces: 10

camera:
  device_id: 0
  resolution: [640, 480]
  fps: 30

output:
  save_results: true
  output_dir: "data/output/"
  save_images: true

logging:
  level: "INFO"
  file: "data/logs/face_recognition.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 환경별 설정 파일
```yaml
# config/development.yaml
model:
  confidence_threshold: 0.3  # 개발 환경에서는 낮은 임계값

logging:
  level: "DEBUG"  # 개발 환경에서는 상세한 로깅

# config/production.yaml
model:
  confidence_threshold: 0.7  # 운영 환경에서는 높은 임계값

logging:
  level: "WARNING"  # 운영 환경에서는 중요한 로그만
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 설정 파일을 찾을 수 없음
```python
# 기본값 사용
config = load_config('config/missing.yaml', default_config={
    'model': {'confidence_threshold': 0.5}
})
```

#### 2. 로그 파일이 생성되지 않음
```python
# 디렉토리 확인 및 생성
from common.utils import ensure_directory
ensure_directory('data/logs/')
setup_logging(log_file='data/logs/app.log')
```

#### 3. 하드웨어 감지 실패
```python
# 안전한 하드웨어 감지
try:
    hardware = detect_hardware()
except Exception as e:
    logger.warning(f"하드웨어 감지 실패: {e}")
    hardware = {'gpu_available': False, 'cpu_count': 1}
```

## 📊 성능 최적화

### 설정 캐싱
```python
# 설정 캐싱을 통한 성능 향상
_config_cache = {}

def get_cached_config(config_path: str) -> Dict:
    if config_path not in _config_cache:
        _config_cache[config_path] = load_config(config_path)
    return _config_cache[config_path]
```

### 로깅 최적화
```python
# 조건부 로깅을 통한 성능 향상
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"복잡한 계산 결과: {expensive_calculation()}")
```

## 🔗 의존성

### 필수 라이브러리
- **PyYAML**: YAML 파일 처리
- **pathlib**: 경로 처리 (Python 3.4+)
- **logging**: 로깅 시스템 (표준 라이브러리)

### 선택적 라이브러리
- **psutil**: 시스템 정보 수집
- **GPUtil**: GPU 정보 수집
- **numpy**: 수치 계산

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일을 먼저 확인
2. 각 모듈의 docstring 확인
3. 로그 파일 확인 (`data/logs/`)
4. 프로젝트 루트의 README.md 확인

## 📄 라이선스

이 모듈의 코드는 프로젝트 전체 라이선스를 따릅니다. 