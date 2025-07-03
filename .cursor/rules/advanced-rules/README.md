# ADVANCED 규칙 (향후 적용)

## 🎯 적용 시점
- **조건**: 새로운 도메인 (factory_defect, powerline_inspection) 추가 시
- **현재 상태**: 🔒 아직 적용 안됨
- **준비 상태**: ✅ 규칙 정의 완료, 적용 대기 중

## 📋 포함될 규칙들

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
```

### 2. 규칙 활성화
```bash
# 3. ADVANCED 규칙 병합
cat basic-rules/*.mdc advanced-rules/*.mdc > .cursorrules.new

# 4. 새 규칙 적용
mv .cursorrules.new .cursorrules
```

### 3. 도구 설치 (예정)
```bash
# 5. 모니터링 시스템 설치
pip install prometheus-client grafana-api

# 6. 보안 도구 설치  
pip install cryptography python-jose

# 7. 자동화 도구 설정
# GitHub Actions workflow 설정
# 코드 품질 도구 설정
```

## 📊 예상 변화

### Before (BASIC)
- 🟢 1개 도메인 (face_recognition)
- 🟢 기본 로깅
- 🟢 수동 테스트
- 🟢 로컬 개발만

### After (ADVANCED)  
- 🟡 다중 도메인 (face_recognition + factory_defect + α)
- 🟡 성능 대시보드
- 🟡 자동화된 테스트
- 🟡 CI/CD 파이프라인
- 🟡 보안 강화

## 🚨 주의사항

1. **점진적 적용**: 한 번에 모든 규칙을 적용하지 말고 단계별로
2. **기존 코드 호환성**: BASIC 규칙으로 작성된 코드가 계속 동작해야 함
3. **성능 고려**: 모니터링 시스템이 개발 성능에 영향주지 않도록
4. **팀 규모 고려**: 1인 개발에서 다인 개발로 확장 시 적용

## 📅 예상 적용 일정

```
현재 (2025-07): BASIC 규칙 적용 중
└── Phase 1: face_recognition 도메인 완성

2025-08 (예정): ADVANCED 규칙 적용 시작  
├── Step 1: monitoring-system.mdc 적용
├── Step 2: factory_defect 도메인 추가
├── Step 3: multi-domain.mdc 적용
└── Step 4: automation-tools.mdc 적용

2025-09 (예정): ADVANCED 규칙 안정화
└── 성능 최적화 및 버그 수정
```

---

**현재 할 일**: BASIC 규칙으로 face_recognition 도메인 완성에 집중
**다음 할 일**: factory_defect 도메인 추가 시 ADVANCED 규칙 적용 