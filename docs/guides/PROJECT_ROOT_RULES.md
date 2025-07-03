# 📋 프로젝트 최상위 루트 관리 규칙

## 🎯 **최상위 루트 관리 원칙**

### **✅ 허용되는 파일들 (필수 최소한)**
```
vision_system/
├── README.md              # 프로젝트 개요 (필수)
├── launcher.py            # 통합 런처 (필수)
├── requirements.txt       # 의존성 정의 (필수)
├── .gitignore            # Git 제외 파일 (필수)
└── pytest.ini           # 테스트 설정 (필수)
```

### **✅ 허용되는 폴더들 (도메인 및 시스템)**
```
vision_system/
├── domains/              # 도메인별 비즈니스 로직
├── shared/               # 공유 모듈 (비전 코어, 보안)
├── common/               # 범용 유틸리티
├── config/               # 전역 설정
├── models/               # AI 모델 저장소
├── datasets/             # 학습 데이터셋
├── data/                 # 런타임 데이터
├── scripts/              # 개발 스크립트
├── requirements/         # 환경별 의존성
├── docs/                 # 📝 NEW: 문서 저장소
└── tools/                # 📝 NEW: 시스템 도구
```

### **❌ 금지되는 파일들**
- `run_*.py` (도메인 내부로 이동)
- `main.py` (tools/legacy로 이동)
- `download_*.py` (tools/setup으로 이동)
- `test_*.py` (tests/ 폴더로 이동)
- `*_STATUS.md` (docs/status로 이동)
- `*_GUIDE.md` (docs/guides로 이동)
- 임시 파일들 (`temp_*`, `quick_*`, etc.)

## 🔧 **자동 검증 시스템**

### 검증 스크립트 규칙
```python
ALLOWED_ROOT_FILES = {
    "README.md": "프로젝트 개요",
    "launcher.py": "통합 런처", 
    "requirements.txt": "의존성 정의",
    ".gitignore": "Git 제외 파일",
    "pytest.ini": "테스트 설정"
}

ALLOWED_ROOT_DIRS = {
    "domains": "도메인별 비즈니스 로직",
    "shared": "공유 모듈",
    "common": "범용 유틸리티", 
    "config": "전역 설정",
    "models": "AI 모델 저장소",
    "datasets": "학습 데이터셋",
    "data": "런타임 데이터",
    "scripts": "개발 스크립트",
    "requirements": "환경별 의존성",
    "docs": "문서 저장소",
    "tools": "시스템 도구"
}
```

## 📝 **폴더별 필수 문서 규칙**

### 모든 최상위 폴더에 필수
- `README.md` - 폴더 목적 및 사용법
- `STRUCTURE.md` - 폴더 내부 구조 설명

### 하위 폴더 문서화 규칙
- 3개 이상의 파일이 있는 폴더: `README.md` 필수
- 5개 이상의 하위 폴더가 있는 폴더: `STRUCTURE.md` 필수

## 🚨 **위반 시 조치**

1. **자동 감지**: 매일 검증 스크립트 실행
2. **자동 이동**: 규칙 위반 파일 자동 이동
3. **알림**: 개발자에게 규칙 위반 알림
4. **문서 생성**: 누락된 README/STRUCTURE 자동 생성

## 📊 **현재 상태 vs 목표 상태**

### 현재 (정리 후)
- ✅ 핵심 파일 5개만 최상위 유지
- ✅ 체계적 폴더 구조
- ❌ 새 폴더(docs, tools) 문서 누락
- ❌ 자동 검증 시스템 부재

### 목표
- ✅ 엄격한 최상위 관리
- ✅ 모든 폴더 완전 문서화
- ✅ 자동 검증 및 정리 시스템
- ✅ 초보자 친화적 가이드 