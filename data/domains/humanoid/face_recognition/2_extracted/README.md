# 2단계: Feature Extraction (humanoid/face_recognition)

## 파일명 패턴
- 패턴: `{raw_id}_f{feature_idx}.{ext}`
- 예시: `20250728_143022_001_f01.jpg`

## 폴더 구조
- `features/`: 검출된 특이점 파일들
- `metadata/`: 검출 과정의 메타데이터

## 추적성
- 원본 파일과 추출된 특이점 간의 매핑이 `traceability/trace.json`에 저장됩니다.
