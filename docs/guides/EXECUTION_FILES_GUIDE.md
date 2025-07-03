# 실행 파일 및 테스트 파일 가이드

## 📂 실행 파일 구조 및 정의

### 1. 메인 진입점
- **파일**: `main.py`
- **목적**: 전체 시스템의 통합 진입점
- **사용법**: `python main.py --mode [api|realtime|detection] [옵션]`
- **기능**: 모드에 따라 적절한 하위 시스템 실행

### 2. 도메인별 실행 파일
- **파일**: `domains/face_recognition/run_face_recognition.py`
- **목적**: 얼굴인식 도메인의 직접 실행
- **사용법**: `python domains/face_recognition/run_face_recognition.py [옵션]`
- **기능**: 도메인 내부 서비스 직접 실행

### 3. 스크립트 실행 파일

#### 3.1 API 서버
- **파일**: `scripts/domains/face_recognition/face_recognition_api_server.py`
- **목적**: REST API 서버 실행
- **사용법**: `python scripts/domains/face_recognition/face_recognition_api_server.py`
- **포트**: 8000 (기본)

#### 3.2 실시간 웹캠 처리
- **파일**: `scripts/domains/face_recognition/run_realtime_face_recognition.py`
- **목적**: 웹캠 실시간 얼굴인식
- **사용법**: `python scripts/domains/face_recognition/run_realtime_face_recognition.py --camera 0`
- **키보드 조작**:
  - `q`: 종료
  - `s`: 현재 프레임 저장
  - `r`: 녹화 시작/중지
  - `p`: 일시정지/재생

#### 3.3 단순 얼굴 검출
- **파일**: `scripts/domains/face_recognition/run_face_detection.py`
- **목적**: 이미지/비디오에서 얼굴 검출만 수행
- **사용법**: `python scripts/domains/face_recognition/run_face_detection.py --input [이미지/비디오 경로]`

## 🧪 테스트 파일 구조 및 정의

### 1. 단위 테스트 (Unit Tests)
#### 1.1 엔티티 테스트
- **파일**: `domains/face_recognition/tests/test_entities.py`
- **검증 대상**:
  - `Face` 엔티티: 바운딩 박스 유효성, 랜드마크 처리
  - `Person` 엔티티: UUID 생성, 임베딩 관리
  - `FaceEmbedding`: 정규화, 유사도 계산
  - `BoundingBox`: 좌표 검증, IoU 계산
  - `ConfidenceScore`: 범위 검증, 레벨 분류

#### 1.2 서비스 테스트
- **파일**: `domains/face_recognition/tests/test_services.py`
- **검증 대상**:
  - `FaceRecognitionService`: 비즈니스 로직, 배치 처리
  - `FaceDetectionService`: 검출 서비스
  - `FaceMatchingService`: 매칭 알고리즘

### 2. 통합 테스트 (Integration Tests)
#### 2.1 카메라 통합 테스트
- **파일**: `scripts/core/testing/test_camera_integration.py`
- **검증 대상**:
  - 하드웨어 연결 상태
  - 카메라 기본 기능 (해상도, FPS)
  - 얼굴 검출/인식 파이프라인
  - 성능 벤치마크

#### 2.2 API 통합 테스트
- **파일**: `tests/integration/test_api_integration.py` (생성 예정)
- **검증 대상**:
  - REST API 엔드포인트
  - 요청/응답 검증
  - 에러 처리

### 3. E2E 테스트 (End-to-End Tests)
- **파일**: `tests/e2e/test_face_recognition_workflow.py` (생성 예정)
- **검증 대상**:
  - 전체 워크플로우
  - 실제 사용자 시나리오

## 🚀 실행 순서 및 권장사항

### 개발 환경 설정
1. 품질 도구 설정: `python scripts/development/setup/setup_quality_tools.py`
2. 환경 설정: `python scripts/development/setup/setup_environment_configs.py`
3. 하드웨어 검증: `python scripts/core/validation/validate_hardware_connection.py`

### 테스트 실행
1. 단위 테스트: `pytest domains/face_recognition/tests/`
2. 통합 테스트: `python scripts/core/testing/test_camera_integration.py`
3. 전체 테스트: `pytest --cov=domains/face_recognition --cov-report=html`

### 실제 구동
1. **API 서버**: `python main.py --mode api`
2. **실시간 웹캠**: `python main.py --mode realtime --camera 0`
3. **이미지 검출**: `python main.py --mode detection --input image.jpg`

## ⚠️ 주의사항

### 하드웨어 연결 필수
- 모든 실행 파일은 실제 카메라 연결을 검증합니다
- 시뮬레이션 모드는 금지됩니다
- `USE_SIMULATION=True` 환경변수 설정 시 오류 발생

### 파일 저장 위치
- 로그: `data/logs/`
- 임시 파일: `data/temp/`
- 결과물: `data/output/`
- 모델: `models/weights/`

### 성능 모니터링
- 실행 중 `scripts/core/monitoring/performance_monitor.py`가 자동 실행
- 메트릭은 `data/logs/performance.log`에 저장

## 📋 파일명 규칙

### 실행 파일
- 메인: `main.py`
- 도메인: `run_{domain_name}.py`
- 스크립트: `run_{기능명}.py`

### 테스트 파일
- 단위: `test_{모듈명}.py`
- 통합: `test_{기능명}_integration.py`
- E2E: `test_{워크플로우명}_e2e.py`

### 검증 내용
- **Unit**: 개별 클래스/함수 로직
- **Integration**: 컴포넌트 간 연동
- **E2E**: 전체 사용자 워크플로우 