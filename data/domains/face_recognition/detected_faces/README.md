# Detected Faces

이 폴더는 얼굴 검출 결과를 저장합니다.

## 폴더 구조

- `YYYYMMDD_HHMM/` - 날짜별로 정리된 검출된 얼굴 이미지 및 메타데이터
- `unknown/` - 날짜를 추출할 수 없는 파일들
- `backup_old_structure/` - 기존 구조 백업 (삭제 가능)

## 파일 형식

- `*.jpg`, `*.png` - 검출된 얼굴 이미지
- `*.json` - 얼굴 검출 메타데이터 (바운딩 박스, 신뢰도 등)

## 메타데이터 구조

```json
{
    "bbox": [x, y, width, height],
    "confidence": 0.95,
    "landmarks": [[x1, y1], [x2, y2], ...],
    "timestamp": "20250630_110859",
    "source_file": "original_image.jpg"
}
```

## 정리 규칙

- 파일명에서 날짜(YYYYMMDD_HHMM)를 추출하여 해당 폴더로 분류
- 중복 파일명은 자동으로 번호를 추가하여 구분
- 날짜를 추출할 수 없는 파일은 'unknown' 폴더로 이동
