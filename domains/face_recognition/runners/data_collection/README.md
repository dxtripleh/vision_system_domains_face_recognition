# 얼굴인식 데이터 수집 워크플로우

## 📋 핵심 4단계 워크플로우

### 🎯 **1단계: 얼굴 수집 (Face Capture)**
```bash
# 기본 얼굴 수집 (웹캠/이미지에서 자동 수집)
python domains/face_recognition/runners/data_collection/run_enhanced_face_capture.py

# 개선된 얼굴 수집 (더 나은 품질 필터링)
python domains/face_recognition/runners/data_collection/run_enhanced_face_capture_v2.py
```

**목적**: 웹캠, 이미지 파일, 업로드된 파일에서 얼굴을 자동으로 검출하고 수집

### 🔄 **2단계: 배치 처리 (Batch Processing)**
```bash
# 대량 이미지 처리 및 얼굴 추출
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py
```

**목적**: 수집된 이미지들을 일괄 처리하여 얼굴을 추출하고 품질 검증

### 🤖 **3단계: AI 그룹핑 (AI Grouping)**
```bash
# AI 기반 얼굴 그룹핑 및 분류
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py
```

**목적**: 유사한 얼굴들을 AI가 자동으로 그룹핑하여 인물별로 분류

### 🗃️ **4단계: 저장소 관리 (Storage Management)**
```bash
# 저장소 정리 및 관리
python domains/face_recognition/runners/management/run_storage_manager.py
```

**목적**: 수집된 데이터를 체계적으로 저장하고 관리

## 📁 파일 정리 상태

### ✅ **유지할 핵심 파일들**
- `run_enhanced_face_capture.py` - 1단계 메인
- `run_enhanced_face_capture_v2.py` - 1단계 개선 버전
- `run_unified_batch_processor.py` - 2단계 메인
- `run_unified_ai_grouping_processor.py` - 3단계 메인
- `run_storage_manager.py` - 4단계 메인

### ❌ **삭제 예정 파일들 (중복/임시)**
- `run_manual_face_classifier.py` - 3단계에서 대체됨
- `run_semi_auto_classifier.py` - 3단계에서 대체됨
- `debug_face_similarity.py` - 디버그용 (개발 완료 후 삭제)
- `emergency_regroup_faces.py` - 긴급용 (정상화 후 삭제)

## 🔄 **실제 워크플로우 예시**

### 1단계: 얼굴 수집
```bash
# 웹캠에서 실시간 얼굴 수집
python run_enhanced_face_capture_v2.py --source camera --output data/domains/face_recognition/raw_input/captured

# 이미지 폴더에서 얼굴 수집
python run_enhanced_face_capture_v2.py --source folder --input images/ --output data/domains/face_recognition/raw_input/uploads
```

### 2단계: 배치 처리
```bash
# 수집된 이미지들을 일괄 처리
python run_unified_batch_processor.py --input data/domains/face_recognition/raw_input --output data/domains/face_recognition/detected_faces
```

### 3단계: AI 그룹핑
```bash
# 얼굴들을 AI가 자동으로 그룹핑
python run_unified_ai_grouping_processor.py --input data/domains/face_recognition/detected_faces --output data/domains/face_recognition/staging
```

### 4단계: 저장소 관리
```bash
# 그룹핑된 얼굴들을 저장소에 정리
python run_storage_manager.py --input data/domains/face_recognition/staging --register-persons
```

## 📊 **데이터 흐름**

```
📷 입력 소스
├── 웹캠 (실시간)
├── 이미지 파일
└── 업로드된 파일

↓ 1단계: 얼굴 수집

🔍 검출된 얼굴
├── auto_collected/     # 자동 수집
├── from_captured/      # 캡처에서
└── from_uploads/       # 업로드에서

↓ 2단계: 배치 처리

✅ 품질 검증된 얼굴
├── 고품질 얼굴
├── 중간 품질 얼굴
└── 저품질 얼굴 (거부)

↓ 3단계: AI 그룹핑

👥 그룹핑된 얼굴
├── grouped/            # AI 그룹핑 결과
├── named/              # 이름 지정된 그룹
└── rejected/           # 거부된 얼굴

↓ 4단계: 저장소 관리

🗃️ 최종 저장소
├── faces/              # 얼굴 데이터
├── persons/            # 인물 데이터
└── embeddings/         # 임베딩 데이터
```

## ⚠️ **주의사항**

1. **순서 준수**: 반드시 1→2→3→4 순서로 실행
2. **데이터 백업**: 각 단계 전에 데이터 백업 권장
3. **품질 검증**: 3단계에서 수동 검증 필요할 수 있음
4. **저장소 정리**: 4단계 후 불필요한 임시 파일 정리

## 🚀 **빠른 시작**

```bash
# 전체 워크플로우 한 번에 실행 (개발 중)
python run_enhanced_face_capture_v2.py --auto-pipeline
```

이 명령어는 1-4단계를 자동으로 순차 실행합니다 (개발 예정). 