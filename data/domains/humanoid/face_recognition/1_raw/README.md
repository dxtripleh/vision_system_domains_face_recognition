# 1단계: Raw Data Collection (humanoid/face_recognition)

## 파일명 패턴
- 패턴: `{timestamp}_{idx}.{ext}`
- 예시: `20250728_143022_001.jpg`

## 폴더 구조
- `uploads/`: 사용자가 직접 업로드한 파일
- `captures/`: 카메라/센서에서 자동 캡처한 파일
- `imports/`: 외부 시스템에서 가져온 파일

## 사용법
```bash
# 파일 업로드 후 다음 단계 실행
python scripts/run_pipeline.py humanoid face_recognition --stage 2
```
