# Processed 폴더

## 📁 개요

`processed/` 폴더는 얼굴인식 시스템에서 최종 처리된 데이터들을 저장하는 곳입니다.
얼굴 검출부터 임베딩 추출, 최종 등록까지의 모든 처리 단계가 완료된 데이터들이 이 폴더에 저장됩니다.

## 🎯 목적

- **최종 데이터 저장**: 완전히 처리된 얼굴 데이터 보관
- **임베딩 관리**: 얼굴 특징 벡터 저장 및 관리
- **품질 보장**: 품질 검증을 통과한 데이터만 저장
- **시스템 연동**: 얼굴인식 시스템과의 연동 준비

## 📂 폴더 구조

```
processed/
├── embeddings/            # 얼굴 특징 벡터
├── final/                 # 최종 처리 완료
└── registered/            # 시스템 등록 완료
```

## 📊 하위 폴더 상세

### embeddings/ - 얼굴 특징 벡터

**목적**: 얼굴 인식에 사용되는 특징 벡터(임베딩) 저장

- **파일명 패턴**: `embedding_{face_id}.json`
- **내용**: 512차원 벡터 + 메타데이터
- **용도**: 얼굴 인식 및 비교에 사용

#### 임베딩 파일 구조
```json
{
  "face_id": "uuid-generated-id",
  "embedding": [0.1, 0.2, 0.3, ...],  // 512차원 벡터
  "model_version": "arcface_v1.0",
  "extraction_time": "2025-06-29T21:50:30",
  "quality_score": 0.92,
  "source_face": "face_john_20250629_215030_0.87.jpg",
  "metadata": {
    "face_size": [224, 224],
    "normalization": "standard",
    "preprocessing": "face_alignment"
  }
}
```

### final/ - 최종 처리 완료

**목적**: 모든 처리 단계가 완료된 최종 데이터

- **파일명 패턴**: `{face_id}_final.json`
- **내용**: 완전한 얼굴 정보 + 임베딩 + 품질 점수
- **특징**: 시스템 등록 준비 완료 상태

#### 최종 파일 구조
```json
{
  "face_id": "uuid-generated-id",
  "person_name": "john",
  "embedding": [0.1, 0.2, 0.3, ...],
  "image_path": "face_john_20250629_215030_0.87.jpg",
  "quality_score": 0.92,
  "detection_confidence": 0.87,
  "processing_steps": [
    "face_detection",
    "face_alignment", 
    "quality_assessment",
    "embedding_extraction",
    "final_validation"
  ],
  "metadata": {
    "source_method": "from_manual",
    "detection_time": "2025-06-29T21:50:30",
    "processing_time": "2025-06-29T21:50:35",
    "model_versions": {
      "detector": "retinaface_v1.0",
      "recognizer": "arcface_v1.0",
      "quality_assessor": "face_quality_v1.0"
    }
  }
}
```

### registered/ - 시스템 등록 완료

**목적**: 얼굴인식 시스템에 최종 등록된 데이터

- **파일명 패턴**: `{face_id}_registered.json`
- **내용**: 등록 완료된 얼굴 정보
- **특징**: 실제 인식 시스템에서 사용 가능

#### 등록 파일 구조
```json
{
  "face_id": "uuid-generated-id",
  "person_id": "person-uuid",
  "person_name": "john",
  "embedding": [0.1, 0.2, 0.3, ...],
  "registration_time": "2025-06-29T21:50:40",
  "status": "active",
  "access_count": 0,
  "last_accessed": null,
  "metadata": {
    "registration_method": "manual",
    "quality_verified": true,
    "duplicate_checked": true,
    "system_compatible": true
  }
}
```

## 🔄 처리 과정

### 1. 임베딩 추출
```
detected_faces/ → 얼굴 정렬 → 임베딩 추출 → embeddings/
```

### 2. 최종 처리
```
embeddings/ → 품질 검증 → 중복 검사 → final/
```

### 3. 시스템 등록
```
final/ → 시스템 호환성 검사 → 등록 → registered/
```

## ⚙️ 품질 기준

### 임베딩 품질
- **벡터 차원**: 512차원 (고정)
- **정규화**: L2 정규화 적용
- **추출 성공률**: 95% 이상

### 최종 품질
- **품질 점수**: 0.7 이상
- **신뢰도**: 0.5 이상
- **얼굴 크기**: 80x80 픽셀 이상

### 등록 품질
- **중복 검사**: 기존 임베딩과 유사도 0.8 이하
- **시스템 호환성**: 모든 시스템 요구사항 충족
- **데이터 무결성**: 모든 필수 필드 완성

## 🛠️ 관리 도구

### 품질 검증
```bash
# 임베딩 품질 검증
python scripts/utilities/validate_embeddings.py --folder processed/embeddings/

# 최종 데이터 품질 검증
python scripts/utilities/validate_final_data.py --folder processed/final/
```

### 중복 검사
```bash
# 중복 얼굴 검사
python scripts/utilities/check_duplicates.py --folder processed/final/
```

### 통계 확인
```bash
# 처리 통계 확인
python scripts/utilities/analyze_processed_data.py --folder processed/
```

## ⚠️ 주의사항

### 데이터 보호
1. **암호화**: 민감한 임베딩 데이터 암호화
2. **접근 제어**: 권한이 있는 사용자만 접근
3. **백업**: 정기적인 데이터 백업

### 성능 최적화
1. **벡터 인덱싱**: 빠른 검색을 위한 인덱스 구축
2. **캐싱**: 자주 사용되는 임베딩 캐싱
3. **압축**: 저장 공간 절약을 위한 압축

### 시스템 연동
1. **호환성**: 얼굴인식 시스템과의 호환성 확인
2. **버전 관리**: 모델 버전 변경 시 마이그레이션
3. **성능 모니터링**: 처리 성능 지속적 모니터링

## 🔗 다음 단계

처리된 데이터들은 다음 단계로 진행됩니다:

1. **시스템 등록**: 얼굴인식 시스템에 최종 등록
2. **인덱스 업데이트**: 검색 인덱스 업데이트
3. **성능 테스트**: 등록된 얼굴로 인식 성능 테스트
4. **모니터링**: 시스템 성능 지속적 모니터링

---

**마지막 업데이트**: 2025-06-29  
**버전**: 1.0 