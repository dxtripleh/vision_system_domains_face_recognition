# Vision System Rules 변경 이력

## [2025-06-28] - 프로젝트 구조 정리 및 파일 재배치

### 🧹 주요 변경사항
- **Archive 폴더 제거 완료**
  - 혼란 방지를 위해 archive 폴더 완전 삭제
  - 관련 문서들 업데이트 완료
  - 깔끔한 새 프로젝트 시작 환경 구축

### 📁 파일 재배치 작업
- **01-universal 폴더로 이동**
  - `common-folder-management-strategy.mdc` (04-roadmap → 01-universal)
  - `clean-slate-architecture-strategy.mdc` (04-roadmap → 01-universal)  
  - `pre-development-checklist.mdc` (04-roadmap → 01-universal)

### 📂 폴더 구조 최적화
- **04-roadmap**: 실제 로드맵 문서만 보관
  - `vision-system-complete-expansion-roadmap.mdc` ✅
  - `face-recognition-detailed-implementation.mdc` ✅
- **01-universal**: 범용 개발 가이드 통합
  - 프로젝트 구조, 아키텍처, 체크리스트 등

### 📋 도메인 구조 명확화
- **독립적 도메인 구조 확립**
  - 얼굴인식 도메인: Face Detection + Face Recognition
  - 공장불량인식 도메인: Defect Detection + Defect Classification
  - 전선불량인식 도메인: Component Detection + Defect Detection + Defect Classification
- **개발 순서**: 복잡도 기준 (쉬운 것부터 어려운 것 순)

### ✅ 업데이트된 문서들
- `README.md` - 새로운 파일 위치 반영
- `CHANGELOG.md` - 변경 이력 업데이트
- `project-structure-rules.mdc` - archive 폴더 참조 제거
- `common-development-rules.mdc` - archive 폴더 참조 제거
- `vision-system-complete-expansion-roadmap.mdc` - 도메인 독립성 명확화

### 🎯 현재 프로젝트 상태
- **완전히 새로운 시작**: archive 의존성 완전 제거
- **명확한 구조**: 혼란 요소 제거로 개발 집중도 향상
- **확장 준비 완료**: 얼굴인식 → 공장불량인식 → 활선전선불량인식 순차 개발 준비
- **체계적인 문서화**: 파일 분류 최적화로 접근성 향상
- **도메인 독립성**: 각 도메인이 완전히 독립적으로 개발 가능

---

## [2024-12-23] - 초기 규칙 체계 구축

### 🎯 규칙 체계 확립
- **01-universal**: 범용 개발 규칙
- **02-security**: 보안 관련 규칙  
- **03-project-analysis**: 프로젝트 분석 규칙
- **04-roadmap**: 개발 로드맵 및 전략

### 📋 핵심 문서 작성
- 비전 시스템 특화 규칙
- 하드웨어 요구사항 규칙
- 보안 및 데이터 보호 규칙
- 공통 개발 표준

### 🏗️ 아키텍처 전략
- Clean Slate 아키텍처 전략 수립
- 모듈화 및 확장성 고려
- Interface First 설계 원칙

---

*이 변경 이력은 Vision System 개발팀의 규칙 진화 과정을 기록합니다.* 

# Changelog

All notable changes to the Vision System Development Rules will be documented in this file.

## [2.0.0] - 2025-06-28

### 🎯 **Major Restructuring - Rule Type System Implementation**

#### Added
- **Rule Type Classification System**: 모든 규칙 파일에 rule_type, priority, phase, applies_to 메타데이터 추가
- **Priority-based Rule Application**: HIGH/MEDIUM/LOW 우선순위 기반 규칙 적용 체계
- **Phase-based Development Guide**: 현재 개발 단계(AI 모델 통합)에 맞는 규칙 분류
- **Rule Application Dashboard**: 현재 적용 상태를 시각적으로 보여주는 대시보드

#### Changed
- **README.md**: 완전히 새로운 구조로 재작성, Rule Type별 분류 및 현재 개발 상황 반영
- **repo_specific_rule.mdc**: CORE_MANDATORY 타입으로 설정, 현재 개발 단계에 맞게 내용 정리
- **Rule Priority System**: 
  - HIGH: 6개 규칙 (현재 필수 적용)
  - MEDIUM: 5개 규칙 (단계별 적용)
  - LOW: 5개 규칙 (향후 적용)

#### Removed
- **vision-system-project.mdc**: repo_specific_rule.mdc와 중복으로 삭제

#### Fixed
- **Rule Application Confusion**: 어떤 규칙을 언제 적용해야 하는지 명확히 정의
- **Development Phase Mismatch**: 현재 개발 단계와 규칙 적용의 불일치 해결
- **Priority Ambiguity**: 규칙 우선순위가 불명확했던 문제 해결

### 📊 **Rule Type Breakdown**
```
HIGH PRIORITY (현재 필수):
├── CORE_MANDATORY (1개)
├── DOMAIN_SPECIFIC (1개)  
├── IMPLEMENTATION_GUIDE (1개)
├── CODING_STANDARDS (1개)
├── ARCHITECTURE_RULES (1개)
└── SECURITY_COMPLIANCE (1개)

MEDIUM PRIORITY (단계별):
├── ARCHITECTURE_STRATEGY (1개)
├── DOCUMENTATION_STANDARDS (1개)
├── QUALITY_ASSURANCE (2개)
└── MONITORING_SYSTEMS (1개)

LOW PRIORITY (향후):
├── DEPLOYMENT_AUTOMATION (1개)
├── HARDWARE_OPTIMIZATION (1개)
├── DEVELOPMENT_TOOLS (1개)
├── LEGACY_ANALYSIS (1개)
└── EXPANSION_ROADMAP (1개)
```

### 🎯 **Current Development Focus**
- **Phase**: Week 3-4 AI 모델 통합
- **Active Rules**: 6개 HIGH PRIORITY 규칙 + 1개 IMPLEMENTATION_GUIDE
- **Next Phase**: Week 5-6 API 개발 (DOCUMENTATION_STANDARDS 추가 예정)

---

## [1.5.0] - 2025-06-28

### Added
- 자동 문서화 규칙을 repo_specific_rule.mdc에 추가
- 문서 템플릿 시스템 및 품질 보증 규칙
- 실행 시점 정의 및 트리거 시스템

### Changed
- 모든 .cursor/rules 파일들을 현재 domains/ 기반 구조에 맞게 업데이트
- project-structure-rules.mdc: features/modules → domains/shared 구조로 변경
- vision-system-specific-rules.mdc: common/vision/ → shared/vision_core/ 구조로 변경

### Fixed
- 규칙 파일들과 실제 프로젝트 구조 간의 불일치 해결
- DDD 패턴 반영 및 도메인 독립성 규칙 강화

---

## [1.0.0] - 2025-06-27

### Added
- 초기 규칙 체계 구축
- 24개월 확장 로드맵 완성
- DDD 기반 얼굴인식 도메인 구조 완성
- 상세한 README 문서들 생성

### Changed
- Archive 폴더 삭제 및 관련 문서 정리
- 파일 재배치: 04-roadmap → 01-universal 이동
- 모든 문서 날짜를 2025-06-28로 수정

### Fixed
- 데이터 폴더 구분 원칙 명확화 (datasets/ vs data/)
- 도메인 독립성 규칙 정립

---

## [0.1.0] - 2025-06-27

### Added
- 프로젝트 초기 설정
- 기본 폴더 구조 생성
- 얼굴인식 도메인 엔티티 정의

### Notes
- 프로젝트 시작점
- 확장 가능한 비전 시스템 기반 구조 설계 