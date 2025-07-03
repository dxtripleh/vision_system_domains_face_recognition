# 📁 프로젝트 구조 정리 상태

## 🎯 **구조 정리 목표**

사용자 지적사항:
1. ❌ **최상위 루트에 run 파일들이 너무 많음**
2. ❌ **data/storage 위치가 부적절함** - 도메인별로 분리 필요

## 📋 **현재 상태 (정리 전)**

### 문제점
```
vision_system/
├── run_simple_demo.py              # ❌ 최상위에 위치
├── run_face_recognition_demo.py    # ❌ 최상위에 위치  
├── run_face_registration.py        # ❌ 최상위에 위치
├── download_models.py              # ❌ 최상위에 위치
├── main.py                         # ❌ 최상위에 위치
├── run_face_system.py              # ❌ 최상위에 위치
└── data/storage/                   # ❌ 도메인별 분리 필요
    ├── faces/
    └── persons/
```

## 🎯 **목표 구조 (정리 후)**

### ✅ 올바른 DDD 구조
```
vision_system/
├── domains/face_recognition/
│   ├── data/storage/              # ✅ 도메인별 데이터 저장소
│   │   ├── faces/                 # 얼굴 데이터
│   │   └── persons/               # 인물 데이터
│   ├── runners/
│   │   ├── demos/                 # ✅ 데모 실행 파일들
│   │   │   ├── run_simple_demo.py
│   │   │   └── run_face_recognition_demo.py
│   │   └── data_collection/       # ✅ 데이터 수집 도구들
│   │       └── run_face_registration.py
│   └── config/
│       └── storage_config.py      # 도메인 저장소 설정
├── tools/setup/                   # ✅ 시스템 도구들
│   └── download_models.py
├── data/                          # ✅ 시스템 공통 데이터만
│   ├── logs/
│   ├── temp/
│   └── output/
└── launcher.py                    # ✅ 통합 런처 (하나만)
```

## 📝 **생성된 파일들**

### 1. 도메인 저장소 설정
- `domains/face_recognition/config/storage_config.py` ✅ 생성됨
  - 도메인별 데이터 저장 경로 관리
  - Repository 클래스에서 사용할 설정

### 2. 통합 런처
- `launcher.py` ✅ 생성됨
  - 최상위에 하나만 존재
  - 모든 실행 가능한 명령 안내

### 3. 구조 정리 가이드
- `MANUAL_REORGANIZATION_GUIDE.md` ✅ 생성됨
  - 수동 구조 정리 단계별 가이드
  - Windows PowerShell 명령 포함

## 🛠️ **수동 정리 필요 작업**

현재 런처가 터미널을 점유하고 있어 자동화 스크립트 실행이 불가능합니다.
다음 작업을 수동으로 진행해야 합니다:

### 1단계: 런처 종료
- 현재 실행 중인 런처 프로세스 종료

### 2단계: 디렉토리 생성
```bash
mkdir domains\face_recognition\data\storage\faces
mkdir domains\face_recognition\data\storage\persons
mkdir domains\face_recognition\runners\demos
mkdir tools\setup
```

### 3단계: 파일 이동
```bash
# 데모 파일들
move run_simple_demo.py domains\face_recognition\runners\demos\
move run_face_recognition_demo.py domains\face_recognition\runners\demos\
move run_face_registration.py domains\face_recognition\runners\data_collection\

# 도구들
move download_models.py tools\setup\
```

### 4단계: 데이터 저장소 이동
```bash
xcopy data\storage\faces domains\face_recognition\data\storage\faces /E /I
xcopy data\storage\persons domains\face_recognition\data\storage\persons /E /I
```

### 5단계: Import 경로 수정
이동된 파일들의 `project_root` 경로를 수정:
```python
# 변경 전
project_root = Path(__file__).parent

# 변경 후
project_root = Path(__file__).parent.parent.parent.parent.parent
```

## ✅ **정리 완료 후 예상 효과**

1. **깔끔한 최상위 구조**
   - 문서 파일들만 최상위에 유지
   - 실행 파일들은 적절한 도메인/도구 폴더에 배치

2. **도메인별 데이터 분리**
   - 각 도메인이 독립적인 데이터 저장소 보유
   - GDPR 준수 및 데이터 관리 용이성 향상

3. **명확한 실행 경로**
   - `python launcher.py`로 사용 가능한 모든 명령 확인
   - 도메인별 기능 구분 명확화

4. **DDD 원칙 준수**
   - 도메인 독립성 보장
   - 계층별 의존성 규칙 준수

## 🚀 **정리 완료 후 사용법**

```bash
# 통합 런처로 명령 확인
python launcher.py

# 개별 기능 실행
python domains/face_recognition/runners/demos/run_simple_demo.py
python domains/face_recognition/runners/demos/run_face_recognition_demo.py
python domains/face_recognition/runners/data_collection/run_face_registration.py
python tools/setup/download_models.py
```

## 📊 **정리 진행률**

- [x] 목표 구조 설계
- [x] 도메인 저장소 설정 파일 생성
- [x] 통합 런처 생성  
- [x] 수동 정리 가이드 작성
- [ ] 실제 파일 이동 (수동 작업 필요)
- [ ] Import 경로 수정 (수동 작업 필요)
- [ ] Repository 클래스 수정 (수동 작업 필요)
- [ ] 테스트 및 검증

**진행률: 50% (설계 및 준비 완료, 실제 이동 작업 대기 중)** 