# 데이터 파이프라인 자동화 사용법 가이드

## 🎯 개요

9단계 데이터 생명주기를 자동화하여 일관된 개발 프로세스를 제공합니다.

## 🔄 9단계 데이터 생명주기

```
1. Raw Data Collection    → 2. Feature Extraction    → 3. Similarity Clustering
                                        ↓
6. Realtime Recognition  ← 5. Embedding Generation  ← 4. Group Labeling
        ↓
7. Database Management   → 8. Performance Monitoring → 9. Continuous Learning
```

## 🚀 빠른 시작

### 1. 새로운 도메인/기능 생성
```bash
# 새로운 도메인 자동 생성 (폴더 구조 + 설정파일)
python scripts/create_domain.py humanoid emotion_detection

# 또는 공장 불량검출
python scripts/create_domain.py factory surface_defect

# 자동으로 9단계 구조 생성
python scripts/create_domain.py factory defect_detection
python scripts/create_domain.py powerline_inspection inspection
```

### 2. 데이터 파이프라인 실행
```bash
# 전체 파이프라인 실행
python scripts/run_pipeline.py humanoid emotion_detection

# 특정 단계만 실행
python scripts/run_pipeline.py humanoid emotion_detection --stage 2_extract_features
```

### 3. 구조 검증
```bash
# 파이프라인 구조 검증
python scripts/validate_pipeline.py humanoid emotion_detection
```

## 📁 자동 생성되는 폴더 구조

```
data/domains/{domain}/{feature}/
├── 1_raw/                  # 원본 데이터
│   ├── uploads/            # 사용자 업로드
│   ├── captures/           # 카메라 캡처
│   └── imports/            # 외부 연동
├── 2_extracted/            # 특이점 추출
├── 3_clustered/            # 자동 그룹핑
├── 4_labeled/              # 라벨링
├── 5_embeddings/           # 임베딩
├── cache/                  # 캐시
└── models/                 # 모델
```

## 🎨 도메인별 사용 예시

### 얼굴인식 도메인
```bash
# 1. 도메인 생성
python scripts/create_domain.py humanoid face_recognition --upgrade-existing

# 2. 데이터 수집 (1단계)
# - data/domains/humanoid/face_recognition/1_raw/uploads/ 에 이미지 업로드
# - 또는 카메라로 자동 캡처

# 3. 파이프라인 실행
python scripts/run_pipeline.py humanoid face_recognition

# 4. 실시간 인식 시작
python domains/humanoid/face_recognition/run_face_recognition.py
```

### 불량검출 도메인
```bash
# 1. 도메인 생성
python scripts/create_domain.py factory defect_detection

# 2. 제품 이미지 수집
# - data/domains/factory/defect_detection/1_raw/captures/ 에 제품 이미지

# 3. 파이프라인 실행
python scripts/run_pipeline.py factory defect_detection

# 4. 실시간 검출 시작
python domains/factory/defect_detection/run_defect_detection.py
```

## 🔧 고급 사용법

### 단계별 실행
```bash
# 1단계: Raw 데이터 수집
python scripts/run_pipeline.py humanoid face_recognition --stage 1_collect_raw_data

# 2단계: 특이점 추출
python scripts/run_pipeline.py humanoid face_recognition --stage 2_extract_features

# 3단계: 유사도 그룹핑
python scripts/run_pipeline.py humanoid face_recognition --stage 3_cluster_by_similarity

# 4단계: 라벨링 (사용자 개입 필요)
python scripts/run_pipeline.py humanoid face_recognition --stage 4_label_groups

# 5단계: 임베딩 생성
python scripts/run_pipeline.py humanoid face_recognition --stage 5_generate_embeddings
```

### 진행 상황 확인
```bash
# 현재 진행 상황 확인
python scripts/check_pipeline_status.py humanoid face_recognition

# 단계별 성능 확인
python scripts/check_performance.py humanoid face_recognition
```

### 에러 복구
```bash
# 특정 단계 롤백
python scripts/rollback_stage.py humanoid face_recognition --stage 3_clustered

# 전체 파이프라인 초기화
python scripts/reset_pipeline.py humanoid face_recognition
```

## 📊 진행 상황 추적

각 도메인의 `pipeline_progress.json` 파일에서 진행 상황을 확인할 수 있습니다:

```json
{
  "1_collect_raw_data": {
    "status": "completed",
    "timestamp": "2025-07-28T10:30:00",
    "duration": 120.5,
    "output_files": ["uploads/face_001.jpg", "uploads/face_002.jpg"]
  },
  "2_extract_features": {
    "status": "in_progress",
    "timestamp": "2025-07-28T10:32:00",
    "duration": null,
    "output_files": []
  }
}
```

## ⚠️ 주의사항

### 필수 준수 사항
1. **순차 실행**: 단계를 건너뛰지 말고 순서대로 실행
2. **데이터 검증**: 각 단계 전에 이전 단계 완료 확인
3. **수동 라벨링**: 4단계는 사용자 개입 필요
4. **백업**: 중요한 단계 전에 백업 수행

### 권장 사항
1. **소규모 테스트**: 처음에는 적은 데이터로 테스트
2. **성능 모니터링**: 각 단계별 성능 확인
3. **정기 검증**: 주기적으로 구조 검증 실행
4. **문서화**: 도메인별 특이사항 문서화

## 🔍 문제해결

### 자주 발생하는 문제
1. **폴더 권한 오류**: `chmod 755 data/domains/` 실행
2. **모델 파일 없음**: 해당 단계의 모델 다운로드 확인
3. **메모리 부족**: 배치 크기 조정 또는 단계별 실행
4. **의존성 오류**: `pip install -r requirements.txt` 재실행

### 로그 확인
```bash
# 전체 로그 확인
tail -f data/logs/pipeline.log

# 도메인별 로그 확인
tail -f data/logs/humanoid/face_recognition.log
```

## 📞 도움말

더 자세한 정보는 다음 파일들을 참조하세요:
- `.cursor/rules/advanced-rules/data-pipeline-automation.mdc`
- `data/domains/{domain}/{feature}/README.md`
- `scripts/README.md`
