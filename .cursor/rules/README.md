# 비전 시스템 개발 규칙 (단계별 적용)

## 📋 개요

이 문서는 비전 시스템 프로젝트의 개발 규칙을 **단계별로 적용할 수 있도록** 재구성한 것입니다.
복잡한 엔터프라이즈급 규칙들을 현재 개발 단계에 맞게 점진적으로 적용할 수 있습니다.

## 🏗️ 새로운 구조

### **🟢 BASIC (현재 단계) - 1인 개발, 기본 기능 구현**
```
basic-rules/
├── python-standards.mdc      # ✅ 기본 Python 개발 표준
├── project-structure.mdc     # ✅ 현재 프로젝트 구조 규칙
├── vision-system-core.mdc    # ✅ 비전 시스템 핵심 규칙
└── data-management.mdc       # ✅ 기본 데이터 관리 규칙
```

### **🟡 ADVANCED (확장 단계) - 다중 도메인, 성능 최적화**
```
advanced-rules/
├── README.md                # ✅ 로드맵 및 적용 가이드
├── monitoring-system.mdc    # 🔄 성능 모니터링 (향후 작성)
├── security-enhanced.mdc    # 🔄 보안 강화 (향후 작성)
├── automation-tools.mdc     # 🔄 CI/CD, 자동화 도구 (향후 작성)
└── multi-domain.mdc         # 🔄 다중 도메인 관리 (향후 작성)
```

### **🔴 ENTERPRISE (운영 단계) - 대규모 배포, 엔터프라이즈**
```
enterprise-rules/
├── README.md                # ✅ 로드맵 및 적용 가이드
├── deployment-scale.mdc     # 🔮 대규모 배포 전략 (장기 계획)
├── monitoring-enterprise.mdc# 🔮 엔터프라이즈 모니터링 (장기 계획)
├── security-enterprise.mdc  # 🔮 엔터프라이즈 보안 (장기 계획)
└── data-lifecycle.mdc       # 🔮 데이터 생명주기 관리 (장기 계획)
```

## 🎯 현재 적용 규칙

**현재 프로젝트 단계**: Phase 1 (얼굴인식 도메인 개발)
**적용 규칙**: `basic-rules/` 만 적용
**다음 단계**: factory_defect 도메인 추가 시 → `advanced-rules/` 적용

## 📖 규칙 적용 가이드

### 1. 현재 단계 (BASIC) ✅ 완료
```bash
# 현재 .cursorrules 파일은 basic-rules만 포함
✅ python-standards.mdc      # 적용됨
✅ project-structure.mdc     # 적용됨
✅ vision-system-core.mdc    # 적용됨
✅ data-management.mdc       # 적용됨
```

### 2. 확장 단계 (ADVANCED)
```bash
# 새로운 도메인 추가 시
- basic-rules 유지 + advanced-rules 추가
- 모니터링 시스템 도입
- 보안 강화 적용
```

### 3. 운영 단계 (ENTERPRISE)
```bash
# 대규모 운영 시
- 모든 규칙 적용
- 엔터프라이즈급 도구 도입
- 컴플라이언스 준수
```

## 🗂️ 기존 규칙 백업

기존의 복잡한 규칙들은 `archive/` 폴더에 안전하게 백업되어 있습니다:
- `archive/01-universal/` - 범용 규칙들
- `archive/02-vision-specific/` - 비전 시스템 특화 규칙들
- `archive/03-project-analysis/` - 프로젝트 분석 규칙들
- `archive/04-roadmap/` - 로드맵 규칙들

## 🔄 규칙 업그레이드 방법

### 현재 → 확장 단계
```bash
# 1. advanced-rules 활성화
cp advanced-rules/* .
# 2. .cursorrules 업데이트
# 3. 필요한 도구 설치 (Prometheus, etc.)
```

### 확장 → 운영 단계
```bash
# 1. enterprise-rules 활성화
cp enterprise-rules/* .
# 2. 엔터프라이즈 도구 설정
# 3. 컴플라이언스 체크
```

## 💡 규칙 적용 원칙

1. **점진적 적용**: 현재 단계에 맞는 규칙만 적용
2. **실용성 우선**: 복잡한 규칙은 필요할 때만
3. **유연성 유지**: 프로젝트 상황에 맞게 조정 가능
4. **백업 보장**: 기존 규칙들은 항상 백업 유지

## 📞 문의 및 수정

규칙 수정이나 새로운 요구사항이 있을 때는 해당 단계의 규칙 파일을 수정하거나
새로운 규칙을 추가하여 점진적으로 개선할 수 있습니다.

---

**최종 업데이트**: 2025-07-02
**현재 적용 단계**: BASIC
**다음 단계 예정**: ADVANCED (factory_defect 도메인 추가 시) 