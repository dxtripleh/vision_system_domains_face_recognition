# Face Recognition - 얼굴인식 기능

## 📋 개요

이 모듈은 실시간 얼굴 검출 및 인식 기능을 제공합니다. 웹캠이나 이미지 파일을 입력으로 받아 얼굴을 검출하고, 미리 등록된 얼굴 데이터베이스와 비교하여 신원을 식별합니다.

## 🚀 빠른 시작

### 기본 실행
```bash
# 웹캠을 사용한 실시간 얼굴인식
python run_face_recognition.py

# 이미지 파일로 테스트
python run_face_recognition.py --source path/to/image.jpg

# 결과 화면 표시
python run_face_recognition.py --show
```

### 모델 테스트
```bash
# 모델 로딩 및 추론 테스트
python -m pytest tests/test_service.py -v
```

## 📁 파일 구조

```
face_recognition/
├── __init__.py                    # 패키지 초기화
├── run_face_recognition.py        # 실시간 실행 스크립트
├── README.md                      # 이 파일
├── models/                        # 모델 클래스들
│   ├── __init__.py
│   ├── face_detection_model.py    # 얼굴 검출 모델
│   └── face_recognition_model.py  # 얼굴 인식 모델
├── services/                      # 서비스 클래스들
│   ├── __init__.py
│   ├── service.py                 # 기본 서비스
│   └── face_recognition_service.py # 통합 서비스
├── utils/                         # 유틸리티 함수들
│   ├── __init__.py
│   └── demo.py                    # 데모 스크립트
├── tests/                         # 테스트 파일들
│   ├── __init__.py
│   └── test_service.py            # 서비스 테스트
├── configs/                       # 설정 파일들
│   └── __init__.py
└── pipeline/                      # 9단계 파이프라인
    └── __init__.py
```

## 🔧 주요 기능

### 1. 얼굴 검출 (Face Detection)
- 이미지에서 얼굴 영역을 자동으로 찾아냄
- 여러 얼굴을 동시에 검출 가능
- 얼굴의 바운딩 박스 좌표 반환

### 2. 얼굴 인식 (Face Recognition)
- 검출된 얼굴의 신원을 식별
- 미리 등록된 얼굴 데이터베이스와 비교
- 신뢰도 점수 제공

### 3. 실시간 처리
- 웹캠 스트림을 실시간으로 처리
- 프레임별 얼굴 검출 및 인식
- 결과를 실시간으로 화면에 표시

### 4. 결과 저장
- 검출 결과를 JSON 형식으로 저장
- 처리된 이미지 저장
- 로그 파일 생성

## 📊 입력/출력

### 입력
- **웹캠**: 실시간 카메라 스트림
- **이미지 파일**: JPG, PNG, BMP 등
- **비디오 파일**: MP4, AVI 등

### 출력
```json
{
  "faces": [
    {
      "bbox": [x, y, width, height],
      "confidence": 0.95,
      "identity": "홍길동",
      "landmarks": [[x1, y1], [x2, y2], ...],
      "timestamp": "2024-01-01 12:00:00"
    }
  ],
  "processing_time": 0.023,
  "total_faces": 1
}
```

## ⚙️ 설정

### 명령줄 옵션
- `--source`: 입력 소스 (기본값: 0, 웹캠)
- `--model`: 모델 파일 경로
- `--config`: 설정 파일 경로
- `--conf`: 신뢰도 임계값 (기본값: 0.5)
- `--show`: 결과 화면 표시
- `--save`: 결과 저장
- `--output`: 출력 디렉토리

### 설정 파일 (config/face_recognition.yaml)
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

## 🔧 사용법

### 1. 기본 실행
```bash
# 웹캠 사용 (카메라 ID 0)
python run_face_recognition.py

# 다른 카메라 사용 (카메라 ID 1)
python run_face_recognition.py --source 1

# 이미지 파일 사용
python run_face_recognition.py --source path/to/image.jpg
```

### 2. 설정 변경
```bash
# 신뢰도 임계값 변경
python run_face_recognition.py --conf 0.8

# 설정 파일 지정
python run_face_recognition.py --config config/face_recognition.yaml

# 모델 파일 지정
python run_face_recognition.py --model models/weights/custom_face_model.onnx
```

### 3. 결과 저장
```bash
# 결과 저장 활성화
python run_face_recognition.py --save

# 출력 디렉토리 지정
python run_face_recognition.py --save --output data/output/faces/
```

### 4. 화면 표시
```bash
# 결과 화면 표시
python run_face_recognition.py --show

# 화면 표시 없이 백그라운드 실행
python run_face_recognition.py
```

## 🎯 키보드 단축키

실행 중 사용할 수 있는 키보드 단축키:
- `q`: 프로그램 종료
- `s`: 현재 프레임 저장
- `r`: 녹화 시작/중지
- `p`: 일시 정지/재생
- `h`: 도움말 표시
- `+`: 신뢰도 임계값 증가
- `-`: 신뢰도 임계값 감소

## 🔍 모델 정보

### 사용 모델
- **검출 모델**: RetinaFace 또는 MTCNN
- **인식 모델**: ArcFace 또는 FaceNet
- **형식**: ONNX (크로스 플랫폼 호환)

### 성능 지표
- **검출 정확도**: 95% 이상
- **인식 정확도**: 90% 이상
- **처리 속도**: 30 FPS (GPU 사용 시)
- **최소 얼굴 크기**: 80x80 픽셀

## 🧪 테스트

### 테스트 실행
```bash
# 도메인 테스트 실행
python -m pytest tests/ -v

# 특정 테스트 실행
python -m pytest tests/test_service.py::test_face_detection -v

# 전체 프로젝트 통합 테스트
python -m pytest ../../tests/ -v
```

### 테스트 구조
- `tests/`: 도메인 개발 중 빠른 테스트
- `../../tests/`: 전체 프로젝트 통합 테스트
- `pipeline/tests/`: 파이프라인 단계별 테스트 (향후)

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 얼굴이 검출되지 않음
```bash
# 신뢰도 임계값 낮추기
python run_face_recognition.py --conf 0.3

# 조명 상태 확인
# 카메라 각도 조정
# 얼굴이 화면 중앙에 오도록 조정
```

#### 2. 인식 정확도가 낮음
```bash
# 더 많은 얼굴 데이터 등록
# 조명 조건 개선
# 얼굴 각도 조정
# 모델 재학습 고려
```

#### 3. 처리 속도가 느림
```bash
# GPU 사용 확인
# 모델 크기 축소
# 해상도 낮추기
# 배치 처리 사용
```

## 📚 개발 가이드

### 모델 추가
1. `models/` 폴더에 새 모델 클래스 생성
2. `__init__.py`에 import 추가
3. 서비스에서 모델 사용

### 서비스 확장
1. `services/` 폴더에 새 서비스 클래스 생성
2. `__init__.py`에 import 추가
3. `run_face_recognition.py`에서 서비스 사용

### 테스트 작성
1. `tests/` 폴더에 테스트 파일 생성
2. pytest 규칙 준수
3. 테스트 실행 및 검증

## 🔗 관련 문서

- [프로젝트 전체 README](../../../README.md)
- [데이터 파이프라인 가이드](../../../PIPELINE_IMPLEMENTATION_SUMMARY.md)
- [공통 모듈 문서](../../../common/README.md)
- [설정 파일 문서](../../../config/README.md)

# 새로운 도메인 생성
python scripts/create_domain.py [도메인] [기능]

# 파이프라인 실행
python scripts/run_pipeline.py [도메인] [기능]

# 파이프라인 검증
python scripts/validate_pipeline.py [도메인] [기능]

# 파일명 검증
python scripts/validate_filenames.py [도메인] [기능] --suggestions

# 추적성 관리
python scripts/manage_traceability.py [도메인] [기능] scan
python scripts/manage_traceability.py [도메인] [기능] verify 