# 6단계: 실시간 인식 (factory/defect_detection)

## 📋 개요

이 폴더는 불량 검출 도메인의 6단계 실시간 인식 데이터를 저장합니다.
실시간 카메라 스트림에서 제품 불량을 검출하는 과정에서 생성되는 데이터들이 저장됩니다.

## 🏗️ 폴더 구조

```
6_realtime/
├── streams/              # 실시간 스트림 데이터
│   ├── camera_0/         # 카메라 0 스트림
│   ├── camera_1/         # 카메라 1 스트림
│   └── processed/        # 처리된 스트림
├── detection_logs/       # 검출 로그
│   ├── defects/          # 불량 검출 로그
│   ├── quality/          # 품질 검사 로그
│   └── errors/           # 오류 로그
└── performance/          # 성능 데이터
    ├── fps_logs/         # FPS 로그
    ├── latency_logs/     # 지연시간 로그
    └── accuracy_logs/    # 정확도 로그
```

## 📊 데이터 형식

### 스트림 데이터
- **파일명 패턴**: `{camera_id}_{timestamp}_{frame_id}.{ext}`
- **예시**: `camera_0_20250704_133022_001.jpg`
- **형식**: JPG, PNG, MP4

### 검출 로그
- **파일명 패턴**: `{date}_{camera_id}_detection.json`
- **예시**: `20250704_camera_0_detection.json`
- **형식**: JSON

### 성능 데이터
- **파일명 패턴**: `{metric}_{date}.csv`
- **예시**: `fps_20250704.csv`
- **형식**: CSV

## 🔧 사용법

### 실시간 검출 실행
```bash
# 6단계 실시간 검출 실행
python domains/factory/defect_detection/run_stage_6_recognition.py

# 특정 카메라로 실행
python domains/factory/defect_detection/run_stage_6_recognition.py --camera 0

# 성능 모니터링 포함
python domains/factory/defect_detection/run_stage_6_recognition.py --monitor
```

### 검출 로그 확인
```bash
# 불량 검출 로그 확인
cat data/domains/factory/defect_detection/6_realtime/detection_logs/defects/20250704_camera_0_detection.json

# 성능 로그 확인
cat data/domains/factory/defect_detection/6_realtime/performance/fps_20250704.csv
```

## 📈 성능 지표

### 실시간 성능 목표
- **FPS**: 25 FPS 이상
- **지연시간**: 200ms 이하
- **정확도**: 98% 이상
- **메모리 사용량**: 4GB 이하

### 모니터링 지표
- **CPU 사용률**: 85% 이하
- **GPU 사용률**: 95% 이하
- **네트워크 대역폭**: 20Mbps 이하

## 🔄 데이터 흐름

1. **카메라 스트림** → `streams/`
2. **불량 검출** → `detection_logs/defects/`
3. **품질 검사** → `detection_logs/quality/`
4. **성능 측정** → `performance/`

## ⚠️ 주의사항

- 실시간 데이터는 자동으로 정리됩니다 (24시간 보관)
- 중요한 로그는 백업을 권장합니다
- 성능 데이터는 장기 보관됩니다 