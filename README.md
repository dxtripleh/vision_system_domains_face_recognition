# Vision System - 비전 시스템 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 **크로스 플랫폼 호환성**을 우선으로 하는 비전 시스템입니다. Windows, Linux, Jetson 등 다양한 환경에서 동작하며, 얼굴인식, 불량검출 등 다양한 비전 기능을 제공합니다.

## 🏗️ 프로젝트 구조

```
vision_system/
├── domains/                   # 도메인별 독립 개발
│   ├── humanoid/              # 인간형 도메인
│   │   └── face_recognition/  # 얼굴인식 기능
│   └── factory/               # 공장 도메인
│       └── defect_detection/  # 불량 검출 기능
├── shared/                    # 공통 모듈
│   └── vision_core/           # 비전 알고리즘 공통 기능
├── common/                    # 범용 유틸리티
├── config/                    # 설정 파일
├── models/                    # 모델 저장소
│   └── weights/               # 모델 가중치
├── datasets/                  # 학습 데이터
├── data/                      # 런타임 데이터
│   ├── temp/                  # 임시 파일
│   ├── logs/                  # 로그 파일
│   ├── output/                # 결과물 저장
│   └── domains/               # 도메인별 데이터
├── scripts/                   # 개발 도구 스크립트
└── tests/                     # 테스트 코드
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Python 3.10+ 설치 필요
python --version

# 의존성 설치
pip install -r requirements.txt
```

### 2. 얼굴인식 실행
```bash
# 기본 실행 (웹캠 사용)
python domains/humanoid/face_recognition/run_face_recognition.py

# 이미지 파일로 테스트
python domains/humanoid/face_recognition/run_face_recognition.py --source path/to/image.jpg

# 결과 화면 표시
python domains/humanoid/face_recognition/run_face_recognition.py --show
```

### 3. 불량검출 실행
```bash
# 기본 실행
python domains/factory/defect_detection/run_defect_detection.py

# 설정 파일 지정
python domains/factory/defect_detection/run_defect_detection.py --config config/defect_detection.yaml
```

## 📁 폴더별 설명

### `domains/` - 도메인별 기능
각 도메인은 독립적으로 개발되며, 특정 비즈니스 영역의 기능을 담당합니다.

#### `domains/humanoid/` - 인간형 도메인
- **face_recognition/**: 얼굴인식 기능
  - `face_recognition_model.py`: ONNX 모델 클래스
  - `run_face_recognition.py`: 실시간 추론 실행
  - `test_face_recognition_model.py`: 모델 테스트

#### `domains/factory/` - 공장 도메인
- **defect_detection/**: 불량검출 기능
  - `defect_detection_model.py`: ONNX 모델 클래스
  - `run_defect_detection.py`: 실시간 추론 실행
  - `test_defect_detection_model.py`: 모델 테스트

### `shared/` - 공통 모듈
여러 도메인에서 공통으로 사용하는 기능들입니다.

#### `shared/vision_core/` - 비전 알고리즘 공통 기능
- **detection/**: 객체 검출 공통 기능
- **recognition/**: 객체 인식 공통 기능
- **preprocessing/**: 이미지 전처리 공통 기능

### `common/` - 범용 유틸리티
로깅, 설정 관리, 유틸리티 함수 등 프로젝트 전체에서 사용하는 공통 기능들입니다.

### `config/` - 설정 파일
각 도메인별 설정 파일들이 저장됩니다.
- `face_recognition.yaml`: 얼굴인식 설정
- `defect_detection.yaml`: 불량검출 설정

### `models/` - 모델 저장소
학습된 AI 모델들이 저장됩니다.

#### `models/weights/` - 모델 가중치
- `.onnx` 형식의 모델 파일들만 저장
- 예: `face_recognition_arcface_glint360k_20250628.onnx`

### `datasets/` - 학습 데이터
학습용 데이터셋들이 도메인별로 저장됩니다.
- `datasets/humanoid/`: 인간형 도메인 학습 데이터
- `datasets/factory/`: 공장 도메인 학습 데이터

### `data/` - 런타임 데이터
실행 중 생성되는 데이터들이 저장됩니다.

#### `data/temp/` - 임시 파일
실행 중 생성되는 임시 파일들 (자동 정리됨)

#### `data/logs/` - 로그 파일
시스템 실행 로그들이 저장됩니다.

#### `data/output/` - 결과물 저장
추론 결과, 처리된 이미지 등이 저장됩니다.

#### `data/domains/` - 도메인별 데이터
각 도메인별 런타임 데이터가 저장됩니다.
- `data/domains/humanoid/`: 인간형 도메인 데이터
- `data/domains/factory/`: 공장 도메인 데이터

### `scripts/` - 개발 도구 스크립트
개발 및 유지보수를 위한 스크립트들이 저장됩니다.

### `tests/` - 테스트 코드
전체 프로젝트의 공식 테스트 코드들이 저장됩니다.

## 🔧 개발 가이드

### 새로운 도메인 추가
1. `domains/{domain_category}/` 폴더 생성
2. `{feature_name}/` 폴더 생성
3. 필수 파일 생성:
   - `{feature_name}_model.py`
   - `run_{feature_name}.py`
   - `test_{feature_name}_model.py`
   - `README.md`

### 모델 추가
1. 학습된 모델을 `.onnx` 형식으로 변환
2. `models/weights/` 폴더에 저장
3. 파일명 규칙: `{task}_{architecture}_{dataset}_{date}.onnx`

## 📋 규칙 준수

### CROSS_PLATFORM_COMPATIBILITY (최우선 규칙)
- 모든 경로는 `pathlib.Path` 사용
- 하드코딩된 경로 금지
- ONNX 모델 사용 (추론 시)
- 하드웨어 환경 자동 감지

### 파일 네이밍 규칙
- 모델 파일: `{기능명}_model.py`
- 실행 파일: `run_{기능명}.py`
- 테스트 파일: `test_{기능명}_model.py`

## 🐛 문제 해결

### 일반적인 문제들
1. **모델 파일을 찾을 수 없음**
   - `models/weights/` 폴더에 `.onnx` 파일이 있는지 확인
   - 파일명이 올바른 형식인지 확인

2. **카메라가 작동하지 않음**
   - 카메라가 다른 프로그램에서 사용 중인지 확인
   - 카메라 ID를 변경해보기 (--source 1, 2 등)

3. **메모리 부족 오류**
   - GPU 메모리 확인
   - 배치 크기 줄이기

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일을 먼저 확인
2. 각 도메인 폴더의 README 파일 확인
3. 로그 파일 확인 (`data/logs/`)

## 📄 라이선스

이 프로젝트는 [라이선스 정보]에 따라 배포됩니다. 