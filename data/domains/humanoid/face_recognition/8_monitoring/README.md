# 8단계: 성능 모니터링 (humanoid/face_recognition)

## 📋 개요

이 폴더는 얼굴인식 도메인의 8단계 성능 모니터링 데이터를 저장합니다.
실시간 성능 지표, 알림, 성능 리포트 등의 모니터링 관련 데이터들이 저장됩니다.

## 🏗️ 폴더 구조

```
8_monitoring/
├── metrics/              # 성능 지표
│   ├── realtime/         # 실시간 지표
│   ├── historical/       # 히스토리 지표
│   └── aggregated/       # 집계 지표
├── alerts/               # 알림 데이터
│   ├── active/           # 활성 알림
│   ├── resolved/         # 해결된 알림
│   └── rules/            # 알림 규칙
└── reports/              # 성능 리포트
    ├── daily/            # 일일 리포트
    ├── weekly/           # 주간 리포트
    └── monthly/          # 월간 리포트
```

## 📊 데이터 형식

### 성능 지표
- **파일명 패턴**: `{metric}_{timestamp}.json`
- **예시**: `fps_20250704_133022.json`
- **형식**: JSON

### 알림 데이터
- **파일명 패턴**: `alert_{level}_{timestamp}.json`
- **예시**: `alert_critical_20250704_133022.json`
- **형식**: JSON

### 성능 리포트
- **파일명 패턴**: `report_{type}_{date}.{ext}`
- **예시**: `report_daily_20250704.pdf`
- **형식**: PDF, HTML, CSV

## 🔧 사용법

### 성능 모니터링 실행
```bash
# 8단계 성능 모니터링 실행
python domains/humanoid/face_recognition/run_stage_8_monitoring.py

# 실시간 모니터링 시작
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --realtime

# 대시보드 실행
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --dashboard

# 알림 설정
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --alerts
```

### 성능 지표 확인
```bash
# 실시간 성능 지표 확인
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --metrics

# 히스토리 성능 지표 확인
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --history --days 7

# 성능 리포트 생성
python domains/humanoid/face_recognition/run_stage_8_monitoring.py --report --type daily
```

## 📈 모니터링 지표

### 실시간 성능 지표
- **FPS (Frames Per Second)**: 30 FPS 이상
- **Latency (지연시간)**: 100ms 이하
- **Accuracy (정확도)**: 95% 이상
- **Memory Usage (메모리 사용량)**: 2GB 이하

### 시스템 리소스 지표
- **CPU Usage**: 80% 이하
- **GPU Usage**: 90% 이하
- **Network Bandwidth**: 10Mbps 이하
- **Disk I/O**: 100MB/s 이하

### 비즈니스 지표
- **Recognition Rate**: 98% 이상
- **False Positive Rate**: 2% 이하
- **False Negative Rate**: 1% 이하
- **Throughput**: 1000 faces/min 이상

## 🚨 알림 시스템

### 알림 레벨
- **INFO**: 정보성 알림
- **WARNING**: 경고 알림
- **CRITICAL**: 심각한 알림
- **EMERGENCY**: 긴급 알림

### 알림 채널
- **Email**: 이메일 알림
- **Slack**: 슬랙 알림
- **SMS**: SMS 알림
- **Webhook**: 웹훅 알림

### 알림 규칙
```json
{
  "fps_threshold": 25,
  "latency_threshold": 150,
  "accuracy_threshold": 90,
  "memory_threshold": 2048,
  "cpu_threshold": 85,
  "gpu_threshold": 95
}
```

## 📊 대시보드

### Grafana 대시보드
- **실시간 성능 대시보드**
- **히스토리 성능 대시보드**
- **시스템 리소스 대시보드**
- **비즈니스 지표 대시보드**

### 대시보드 패널
- **FPS 그래프**: 실시간 FPS 추이
- **Latency 히스토그램**: 지연시간 분포
- **Accuracy 차트**: 정확도 변화
- **Resource Usage**: 리소스 사용량

## 🔄 데이터 흐름

1. **성능 측정** → `metrics/realtime/`
2. **알림 생성** → `alerts/active/`
3. **데이터 집계** → `metrics/aggregated/`
4. **리포트 생성** → `reports/`

## ⚠️ 주의사항

- 성능 지표는 실시간으로 수집됩니다
- 알림은 즉시 처리해야 합니다
- 리포트는 정기적으로 생성됩니다
- 모니터링 데이터는 90일간 보관됩니다 