# Scripts 폴더 네이밍 규칙

## 📋 개요

Scripts 폴더 내 모든 파일과 폴더의 네이밍 규칙을 정의합니다. 일관성 있는 네이밍을 통해 가독성과 유지보수성을 향상시킵니다.

## 📁 폴더 네이밍 규칙

### 최상위 폴더
```
scripts/
├── core/                 # 핵심 시스템 (snake_case)
├── domains/              # 도메인별 스크립트 (복수형)
├── interfaces/           # 사용자 인터페이스 (복수형)
├── development/          # 개발 도구 (단수형)
├── deployment/           # 배포 관련 (단수형)
└── utilities/            # 유틸리티 (복수형)
```

### 하위 폴더 네이밍
- **기능별**: `run/`, `test/`, `validation/`, `monitoring/`
- **도구별**: `setup/`, `model_management/`, `data_processing/`
- **인터페이스별**: `web/`, `cli/`, `api/`

## 📄 파일 네이밍 규칙

### 🚀 실행 스크립트 (`scripts/core/run/`)
```
패턴: run_{기능명}.py
예시:
- run_face_recognition.py
- run_realtime_demo.py
- run_realtime_face_recognition.py
- run_batch_processing.py
- run_api_server.py
```

### 🧪 테스트 스크립트 (`scripts/core/test/`)
```
패턴: test_{범위}_{기능}.py
예시:
- test_system_health.py
- test_basic_system.py
- test_complete_system.py
- test_integrated_system.py
- test_models.py
- test_performance.py
```

### ✅ 검증 스크립트 (`scripts/core/validation/`)
```
패턴: validate_{대상}.py
예시:
- validate_root_protection.py
- validate_project_structure.py
- validate_code_quality.py
- validate_dependencies.py
- validate_configuration.py
```

### 📊 모니터링 스크립트 (`scripts/core/monitoring/`)
```
패턴: monitor_{대상}.py 또는 {기능}_monitor.py
예시:
- performance_monitor.py
- resource_monitor.py
- monitor_system_health.py
- monitor_training_progress.py
```

### 🛠️ 개발 도구 (`scripts/development/`)

#### 환경 설정
```
패턴: setup_{대상}.py 또는 install_{대상}.py
예시:
- setup_environment.py
- setup_development_tools.py
- install_requirements.py
- install_cuda.py
```

#### 모델 관리
```
패턴: {동작}_{대상}.py
예시:
- download_models.py
- convert_models.py
- optimize_models.py
- validate_models.py
- benchmark_models.py
```

#### 데이터 처리
```
패턴: {동작}_{대상}.py
예시:
- preprocess_dataset.py
- augment_data.py
- split_dataset.py
- normalize_images.py
- align_faces.py
```

#### 학습 관련
```
패턴: train_{모델타입}.py 또는 {기능}_training.py
예시:
- train_face_detection.py
- train_face_recognition.py
- monitor_training.py
- evaluate_training.py
```

### 🌐 인터페이스 (`scripts/interfaces/`)

#### 웹 인터페이스
```
패턴: app.py (메인), {기능}_handler.py
예시:
- app.py
- upload_handler.py
- api_handler.py
```

#### CLI 도구
```
패턴: run_{도메인}_cli.py
예시:
- run_face_recognition_cli.py
- run_system_cli.py
- run_model_management_cli.py
```

#### API 서버
```
패턴: {도메인}_api.py
예시:
- face_recognition_api.py
- system_management_api.py
- model_serving_api.py
```

### 🔧 유틸리티 (`scripts/utilities/`)

#### 평가 도구
```
패턴: evaluate_{대상}.py
예시:
- evaluate_model.py
- evaluate_performance.py
- evaluate_accuracy.py
```

#### 벤치마크 도구
```
패턴: benchmark_{대상}.py
예시:
- benchmark_inference.py
- benchmark_preprocessing.py
- benchmark_system.py
```

#### 유지보수 도구
```
패턴: {동작}_{대상}.py
예시:
- cleanup_temp_files.py
- backup_models.py
- optimize_storage.py
```

## 🏷️ 변수 네이밍 규칙

### 함수명
```python
# 동사 + 명사 형태
def run_face_recognition():
def validate_model_integrity():
def monitor_system_performance():
def setup_development_environment():
```

### 클래스명
```python
# PascalCase, 명사 형태
class FaceRecognitionRunner:
class ModelValidator:
class PerformanceMonitor:
class DatasetProcessor:
```

### 파일 내 변수
```python
# snake_case
model_path = "path/to/model"
detection_results = []
performance_metrics = {}
```

## 📊 도메인별 네이밍 규칙

### 얼굴인식 도메인
```
접두사: face_
예시:
- run_face_detection.py
- train_face_recognition.py
- evaluate_face_model.py
- benchmark_face_inference.py
```

### 공장 불량 검출 도메인 (향후)
```
접두사: defect_ 또는 factory_
예시:
- run_defect_detection.py
- train_factory_model.py
- evaluate_defect_classifier.py
```

### 전선 검사 도메인 (향후)
```
접두사: powerline_ 또는 line_
예시:
- run_powerline_inspection.py
- train_line_detector.py
- evaluate_powerline_model.py
```

## 🚫 금지된 네이밍 패턴

### 금지 사항
```
❌ 모호한 이름
- script.py
- tool.py
- utils.py
- main.py (루트가 아닌 경우)

❌ 의미없는 접두사/접미사
- new_script.py
- old_validator.py
- temp_runner.py
- script_v2.py

❌ 대문자 시작 (클래스 제외)
- Script.py
- Runner.py
- Test.py

❌ 특수문자 (하이픈, 공백)
- run-script.py
- test script.py
- model@validator.py
```

### 허용되지 않는 위치
```
❌ 루트에 스크립트 파일
/run_anything.py → scripts/core/run/run_anything.py

❌ 잘못된 폴더의 스크립트
scripts/run_test.py → scripts/core/run/run_test.py
scripts/validate.py → scripts/core/validation/validate_something.py
```

## ✅ 네이밍 검증 규칙

### 자동 검증 항목
1. **파일 위치**: 올바른 폴더에 배치되었는가?
2. **네이밍 패턴**: 규칙에 따른 명명인가?
3. **중복 방지**: 동일한 이름의 파일이 없는가?
4. **일관성**: 도메인별 접두사가 일관된가?

### 검증 스크립트
```bash
# 네이밍 규칙 검증
python scripts/core/validation/validate_naming_rules.py

# 자동 리네이밍
python scripts/core/validation/validate_naming_rules.py --auto-fix

# 네이밍 제안
python scripts/core/validation/validate_naming_rules.py --suggest
```

## 🔄 네이밍 가이드라인

### 새 스크립트 추가 시
1. **목적 파악**: 스크립트의 주요 기능 확인
2. **위치 결정**: 적절한 하위 폴더 선택
3. **네이밍 결정**: 규칙에 따른 파일명 생성
4. **검증 실행**: 네이밍 규칙 검증 스크립트 실행
5. **문서 업데이트**: README.md 및 STRUCTURE.md 업데이트

### 기존 스크립트 리네이밍 시
1. **의존성 확인**: 다른 스크립트에서 참조하는지 확인
2. **Git 이력 보존**: `git mv` 명령 사용
3. **Import 경로 업데이트**: 관련 import 문 수정
4. **문서 동기화**: 모든 관련 문서 업데이트

이 네이밍 규칙을 준수하여 Scripts 폴더의 일관성과 가독성을 유지해주세요. 