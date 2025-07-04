# ADVANCED 규칙 (향후 적용)

## 🎯 적용 시점
- **조건**: 새로운 도메인 (factory_defect, powerline_inspection) 추가 시
- **현재 상태**: 🔒 아직 적용 안됨
- **준비 상태**: ✅ 규칙 정의 완료, 적용 대기 중

## 📋 포함될 규칙들

### 🔄 data-pipeline-automation.mdc (NEW!)
- **목적**: 9단계 데이터 생명주기 자동화
- **기술**: 자동 폴더 생성, 파이프라인 실행, 진행 상황 추적
- **현재 대비**: 수동 데이터 처리 → 자동화된 9단계 파이프라인
- **핵심 기능**:
  - 도메인 자동 생성 (`scripts/create_domain.py`)
  - 파이프라인 자동 실행 (`scripts/run_pipeline.py`)
  - 구조 검증 (`scripts/validate_pipeline.py`)
  - 진행 상황 추적 및 롤백

### 🔍 monitoring-system.mdc
- **목적**: 성능 모니터링 시스템 도입
- **기술**: Prometheus, Grafana, 기본 메트릭 수집
- **현재 대비**: 기본 FPS 측정 → 종합 성능 대시보드

### 🔐 security-enhanced.mdc  
- **목적**: 보안 강화 (기본 → 향상)
- **기술**: 데이터 암호화, GDPR 준수, 접근 제어
- **현재 대비**: 기본 하드웨어 검증 → 종합 보안 시스템

### 🤖 automation-tools.mdc
- **목적**: CI/CD 및 자동화 도구 도입
- **기술**: GitHub Actions, 자동 테스트, 코드 품질 자동화
- **현재 대비**: 수동 개발 → 자동화된 파이프라인

### 🏗️ multi-domain.mdc
- **목적**: 다중 도메인 관리 전략
- **기술**: 도메인 간 통신, 공유 모듈 확장, 의존성 관리
- **현재 대비**: 단일 도메인 → 다중 도메인 아키텍처

## 🔄 업그레이드 절차 (예정)

### 1. 준비 단계
```bash
# 1. 현재 BASIC 규칙 백업
cp .cursorrules .cursorrules.basic.backup

# 2. 프로젝트 상태 점검
python scripts/validate_basic_structure.py

# 3. 데이터 파이프라인 자동화 준비
python scripts/create_domain.py --check-requirements
```

### 2. 규칙 활성화
```bash
# 4. 데이터 파이프라인 자동화 규칙 먼저 적용
cat basic-rules/*.mdc advanced-rules/data-pipeline-automation.mdc > .cursorrules.pipeline

# 5. 전체 ADVANCED 규칙 병합 (단계적)
cat basic-rules/*.mdc advanced-rules/*.mdc > .cursorrules.advanced

# 6. 새 규칙 적용
mv .cursorrules.pipeline .cursorrules  # 또는 .cursorrules.advanced
```

### 3. 도구 설치 (예정)
```bash
# 7. 데이터 파이프라인 자동화 도구
python scripts/setup_pipeline_automation.py

# 8. 모니터링 시스템 설치
pip install prometheus-client grafana-api

# 9. 보안 도구 설치  
pip install cryptography python-jose

# 10. 자동화 도구 설정
# GitHub Actions workflow 설정
# 코드 품질 도구 설정
```

## 📊 예상 변화

### Before (BASIC)
- 🟢 1개 도메인 (face_recognition)
- 🟢 기본 로깅
- 🟢 수동 테스트
- 🟢 로컬 개발만
- 🟢 수동 데이터 처리

### After (ADVANCED)  
- 🟡 다중 도메인 (face_recognition + factory_defect + α)
- 🟡 **9단계 자동화 파이프라인** ⭐
- 🟡 성능 대시보드
- 🟡 자동화된 테스트
- 🟡 CI/CD 파이프라인
- 🟡 보안 강화
- 🟡 **진행 상황 추적 및 롤백** ⭐

## 🚨 주의사항

1. **점진적 적용**: 한 번에 모든 규칙을 적용하지 말고 단계별로
2. **기존 코드 호환성**: BASIC 규칙으로 작성된 코드가 계속 동작해야 함
3. **성능 고려**: 모니터링 시스템이 개발 성능에 영향주지 않도록
4. **팀 규모 고려**: 1인 개발에서 다인 개발로 확장 시 적용
5. **데이터 파이프라인 우선**: 다른 고급 기능보다 데이터 자동화를 우선 적용

## 📅 예상 적용 일정

```
현재 (2025-07): BASIC 규칙 적용 중
└── Phase 1: face_recognition 도메인 완성

2025-08 (예정): ADVANCED 규칙 적용 시작  
├── Step 1: data-pipeline-automation.mdc 적용 ⭐ (최우선)
├── Step 2: factory_defect 도메인 추가 (자동화 스크립트 사용)
├── Step 3: monitoring-system.mdc 적용
├── Step 4: multi-domain.mdc 적용
└── Step 5: automation-tools.mdc 적용

2025-09 (예정): ADVANCED 규칙 안정화
└── 성능 최적화 및 버그 수정
```

## 🎯 데이터 파이프라인 자동화 우선 적용

### 필수 구현 스크립트 (즉시 필요)
```bash
# 1. 도메인 자동 생성
scripts/create_domain.py

# 2. 파이프라인 자동 실행  
scripts/run_pipeline.py

# 3. 구조 검증
scripts/validate_pipeline.py

# 4. 진행 상황 추적
scripts/track_pipeline_progress.py
```

### 적용 효과
- ✅ 새로운 도메인 추가 시 자동으로 9단계 구조 생성
- ✅ 데이터 처리 파이프라인 자동 실행
- ✅ 단계별 진행 상황 추적
- ✅ 에러 발생 시 자동 롤백
- ✅ Git 훅을 통한 구조 검증

---

**현재 할 일**: BASIC 규칙으로 face_recognition 도메인 완성에 집중
**즉시 다음**: 데이터 파이프라인 자동화 규칙 적용 (data-pipeline-automation.mdc)
**향후 계획**: factory_defect 도메인 추가 시 전체 ADVANCED 규칙 적용 