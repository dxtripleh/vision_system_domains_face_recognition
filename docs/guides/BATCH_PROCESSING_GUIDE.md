# 배치 얼굴 처리 가이드

## 📋 개요

얼굴인식 시스템에서 모든 소스의 얼굴을 처리하는 통합 시스템입니다.

## 🎯 실행 파일 비교

| 실행 파일 | 기능 | 사용자 개입 | 적합한 상황 |
|-----------|------|-------------|-------------|
| `run_unified_batch_processor.py` | **통합 배치 처리**<br/>• captured + uploads 폴더 처리<br/>• 이미지/동영상 자동 처리<br/>• 자동 저장 옵션 | 최소 (자동화) | 대량 파일 처리, 자동화 필요 |
| `run_unified_ai_grouping_processor.py` | **통합 AI 그룹핑**<br/>• detected_faces의 모든 얼굴 처리<br/>• from_manual 우선 자동 이름 전파<br/>• 유사도 기반 그룹핑<br/>• 오류 처리 및 수정 기능 | 중간 (그룹 확인) | 얼굴 분류 및 이름 지정 |
| `run_batch_face_processor.py` | **대화형 배치 처리**<br/>• 사용자 선택 기반<br/>• 얼굴별 이름 지정<br/>• 품질 평가 포함 | 높음 (대화형) | 정확한 얼굴 분류 필요 |
| `run_upload_file_processor.py` | **AI 그룹핑 처리**<br/>• uploads 폴더만 처리<br/>• AI 자동 그룹핑<br/>• 그룹별 이름 지정 | 중간 (그룹 확인) | uploads 폴더만 처리 |

## 📁 폴더 구조

```
data/domains/face_recognition/
├── raw_input/
│   ├── captured/          # S키로 저장된 프레임들
│   └── uploads/           # 업로드된 파일들
├── detected_faces/
│   ├── auto_collected/    # 카메라 자동 수집된 얼굴
│   ├── from_captured/     # captured에서 추출된 얼굴
│   ├── from_uploads/      # uploads에서 추출된 얼굴
│   └── from_manual/       # C키로 캡처된 얼굴
└── staging/
    ├── grouped/           # AI 그룹핑된 얼굴들
    ├── named/             # 이름이 지정된 얼굴들
    └── rejected/          # 거부된 얼굴들
```

## 🚀 사용법

### 1. 통합 배치 처리기 (1단계)

```bash
# 모든 소스 처리 (기본값)
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py

# 특정 소스만 처리
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py --source captured
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py --source uploads
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py --source manual

# 자동 저장 비활성화
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py --no-auto-save
```

**특징:**
- ✅ **가장 간단하고 빠름**
- ✅ **captured + uploads + manual 모두 처리**
- ✅ **자동 저장 옵션**
- ✅ **상세한 통계 제공**

### 2. 통합 AI 그룹핑 처리기 (2단계)

```bash
# 모든 detected_faces 처리 (기본값)
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py

# 특정 소스만 처리
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py --source auto_collected
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py --source from_captured
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py --source from_uploads
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py --source from_manual
```

**특징:**
- ✅ **detected_faces의 모든 얼굴 처리**
- ✅ **from_manual 우선 자동 이름 전파**
- ✅ **유사도 기반 그룹핑**
- ✅ **오류 처리 및 수정 기능**
- ✅ **그룹별 시각화**

### 3. 대화형 배치 처리기

```bash
python domains/face_recognition/runners/data_collection/run_batch_face_processor.py
```

**특징:**
- ✅ **사용자가 얼굴별로 선택 가능**
- ✅ **이름 직접 지정**
- ✅ **품질 평가 포함**
- ❌ **수동 작업 필요**

### 4. AI 그룹핑 처리기 (uploads 전용)

```bash
python domains/face_recognition/runners/data_collection/run_upload_file_processor.py
```

**특징:**
- ✅ **AI가 유사 얼굴 자동 그룹핑**
- ✅ **그룹별 일괄 이름 지정**
- ✅ **시각적 그룹 확인**
- ❌ **uploads 폴더만 처리**

## 🔄 실행 흐름

### 1단계: 데이터 수집
```bash
# 1. 실시간 캡처 (C키로 얼굴 캡처)
python run_enhanced_face_capture.py
# → C키: 얼굴 → detected_faces/from_manual/ → 이름 입력 → staging/named/
# → S키: 프레임 → raw_input/captured/

# 2. 파일 업로드
# → 파일을 raw_input/uploads/에 복사
```

### 2단계: 얼굴 추출
```bash
# 3. 배치 얼굴 추출
python run_unified_batch_processor.py
# → raw_input/captured/ → detected_faces/from_captured/
# → raw_input/uploads/ → detected_faces/from_uploads/
```

### 3단계: AI 그룹핑 및 이름 지정
```bash
# 4. AI 그룹핑 처리 (새로운 기능!)
python run_unified_ai_grouping_processor.py
# → from_manual 우선 처리 → 자동 이름 전파
# → detected_faces/의 모든 얼굴 → staging/grouped/ → staging/named/
```

### 4단계: 최종 처리
```bash
# 5. 최종 처리 및 저장
python run_storage_manager.py
# → staging/named/ → processed/final/ → storage/
```

## 📊 처리 흐름

### 완전 자동화 흐름 (권장)
```
1. run_enhanced_face_capture.py (C키) → detected_faces/from_manual/ + staging/named/
2. run_unified_batch_processor.py → detected_faces/from_[source]/
3. run_unified_ai_grouping_processor.py → staging/grouped/ + staging/named/
```

### 통합 배치 처리기 흐름
```
1. captured + uploads + manual 폴더 스캔
2. 이미지/동영상 파일 감지
3. 얼굴 검출 (RetinaFace)
4. 신뢰도 필터링 (기본: 0.5)
5. 얼굴 이미지 추출
6. detected_faces/from_[source]에 저장
7. 통계 리포트 출력
```

### 통합 AI 그룹핑 처리기 흐름
```
1. detected_faces의 모든 폴더 스캔
2. 얼굴 크기별 그룹핑
3. 그룹별 시각화
4. 사용자 그룹 확인
5. 그룹별 이름 지정
6. staging/grouped/ 또는 staging/named/에 저장
```

## ⚙️ 설정 옵션

### 신뢰도 임계값
```yaml
# config/face_recognition_api.yaml
face_detection:
  min_confidence: 0.5  # 기본값
```

### 그룹핑 설정
```python
# 그룹핑 임계값
similarity_threshold = 0.6
min_group_size = 2
```

### 지원 파일 형식
- **이미지**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- **동영상**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`

## 🎯 권장 사용 시나리오

### 시나리오 1: 완전 자동화 (권장)
```bash
# 1단계: 모든 소스에서 얼굴 추출
python run_unified_batch_processor.py

# 2단계: AI 그룹핑 및 이름 지정
python run_unified_ai_grouping_processor.py
```

### 시나리오 2: 빠른 대량 처리
```bash
# 모든 파일을 빠르게 처리하고 싶을 때
python run_unified_batch_processor.py
```

### 시나리오 3: 정확한 얼굴 분류
```bash
# 각 얼굴을 정확히 분류하고 싶을 때
python run_batch_face_processor.py
```

### 시나리오 4: 실시간 캡처 + 즉시 이름 지정
```bash
# run_enhanced_face_capture.py에서 C키 사용
# → 즉시 detected_faces/from_manual/ + staging/named/
```

## 📈 성능 최적화

### 동영상 처리 최적화
- **프레임 간격**: 10프레임마다 처리
- **최대 프레임**: 1000프레임으로 제한
- **신뢰도 필터**: 0.7 이상만 추출

### 메모리 관리
- **이미지 크기**: 자동 리사이즈
- **배치 처리**: 파일별 순차 처리
- **리소스 해제**: 자동 정리

## 🔧 문제 해결

### 일반적인 문제들

1. **파일을 찾을 수 없음**
   - 폴더 경로 확인: `data/domains/face_recognition/raw_input/`
   - 파일 확장자 확인

2. **얼굴이 검출되지 않음**
   - 신뢰도 임계값 낮추기
   - 이미지 품질 확인

3. **처리 속도가 느림**
   - 동영상 프레임 간격 조정
   - GPU 가속 사용

### 로그 확인
```bash
# 로그 파일 위치
data/logs/face_recognition/
```

## 📝 예시 출력

### 통합 배치 처리기 출력
```
통합 배치 처리 시작 - 소스: captured, uploads, manual
소스 'captured' 처리 시작
소스 'captured'에서 5개 파일 발견
처리 중: frame_001.jpg
2개 얼굴 검출됨: frame_001.jpg
얼굴 저장됨: face_frame_001_0_20250629_143022_conf0.85.jpg
얼굴 저장됨: face_frame_001_1_20250629_143022_conf0.78.jpg
...
==================================================
통합 배치 처리 완료
==================================================
총 파일 수: 15
처리된 파일 수: 15
총 얼굴 수: 23
저장된 얼굴 수: 23
오류 수: 0
==================================================
```

### 통합 AI 그룹핑 처리기 출력
```
통합 AI 그룹핑 시작 - 소스: auto_collected, from_captured, from_uploads, from_manual
소스 'auto_collected'에서 10개 얼굴 수집
소스 'from_captured'에서 5개 얼굴 수집
소스 'from_uploads'에서 8개 얼굴 수집
소스 'from_manual'에서 3개 얼굴 수집
총 26개 얼굴 수집 완료
크기별 그룹핑 완료: 3개 그룹
그룹 'medium' 처리 중: 15개 얼굴
👤 그룹 'medium' (15개 얼굴)의 인물 이름을 입력하세요: 김철수
✅ 김철수: 15개 얼굴을 staging/grouped/kim_20250629_143022_group_medium/으로 이동
...
==================================================
통합 AI 그룹핑 완료
==================================================
총 얼굴 수: 26
그룹핑된 얼굴 수: 20
개별 이름 지정된 얼굴 수: 4
거부된 얼굴 수: 2
총 그룹 수: 3
==================================================
```

## 🆕 새로운 AI 그룹핑 기능

### 🔄 자동 이름 전파 시스템
- **from_manual 우선 처리**: C키로 캡처된 얼굴들의 이름을 기준으로 그룹핑
- **유사도 기반 자동 할당**: from_manual과 유사한 얼굴들에 자동으로 같은 이름 할당
- **사용자 확인**: 자동 할당 결과를 사용자가 확인하고 수정 가능

### 🛠️ 오류 처리 및 수정 기능
1. **그룹 승인**: 자동 할당된 이름 그대로 사용
2. **그룹 거부**: 전체 그룹을 rejected로 이동
3. **그룹 수정**: 개별 얼굴 제거/유지 선택
4. **건너뛰기**: 나중에 다시 처리

### 📊 처리 옵션
```
🎯 그룹 처리 옵션:
   1. ✅ 그룹 승인 (이름 지정)
   2. ❌ 그룹 거부 (rejected로 이동)
   3. 🔧 그룹 수정 (개별 얼굴 제거)
   4. ⏭️  건너뛰기 (나중에 처리)
``` 