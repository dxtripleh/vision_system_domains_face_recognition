# 범용 비전 시스템 네이밍 체계 가이드

## 개요

이 문서는 비전 시스템의 모든 도메인에서 일관된 파일명을 생성하기 위한 범용 네이밍 시스템을 설명합니다.

## 핵심 원칙

### 1. 일관된 식별자 체계
- **시간 기반 식별자**: `YYYYMMDD_HHMMSS` (14자리)
- **도메인 식별자**: 2-3자리 영문 약어
- **객체 식별자**: 도메인별 특화
- **순번**: 2자리 숫자

### 2. 추적 가능성
- 각 단계에서 원본 파일 정보를 메타데이터에 저장
- 파일명만으로도 처리 흐름을 추적 가능

## 도메인별 네이밍 패턴

### 얼굴인식 도메인 (FR - Face Recognition)

#### 1단계: 카메라 캡처
```
입력: 카메라 스트림
출력: cap_20250630_110815.jpg
```

#### 2단계: 얼굴 검출
```
입력: cap_20250630_110815.jpg
출력: fr_face_20250630_110815_01_conf0.80.jpg
```

#### 3단계: 그룹핑
```
입력: fr_face_20250630_110815_01_conf0.80.jpg
출력: fr_group_001_face_01.jpg
```

### 공장 불량 검출 도메인 (FD - Factory Defect)

#### 1단계: 카메라 캡처
```
입력: 카메라 스트림
출력: cap_20250630_110815.jpg
```

#### 2단계: 불량 검출
```
입력: cap_20250630_110815.jpg
출력: fd_defect_20250630_110815_01_conf0.85.jpg
```

#### 3단계: 그룹핑
```
입력: fd_defect_20250630_110815_01_conf0.85.jpg
출력: fd_group_001_defect_01.jpg
```

### 전선 검사 도메인 (PI - Powerline Inspection)

#### 1단계: 카메라 캡처
```
입력: 카메라 스트림
출력: cap_20250630_110815.jpg
```

#### 2단계: 문제 검출
```
입력: cap_20250630_110815.jpg
출력: pi_issue_20250630_110815_01_conf0.90.jpg
```

#### 3단계: 그룹핑
```
입력: pi_issue_20250630_110815_01_conf0.90.jpg
출력: pi_group_001_issue_01.jpg
```

## 파일명 구조 분석

### 캡처 파일명
```
cap_YYYYMMDD_HHMMSS.jpg
│   │
│   └── 시간 식별자 (14자리)
└── 캡처 접두사
```

### 검출 파일명
```
fr_face_YYYYMMDD_HHMMSS_01_conf0.80.jpg
│  │     │                    │  │
│  │     │                    │  └── 신뢰도
│  │     │                    └── 순번
│  │     └── 시간 식별자
│  └── 객체 타입
└── 도메인 접두사
```

### 그룹 파일명
```
fr_group_001_face_01.jpg
│  │      │   │
│  │      │   └── 그룹 내 순번
│  │      └── 객체 타입
│  └── 그룹 ID
└── 도메인 접두사
```

## 메타데이터 구조

### 기본 메타데이터
```json
{
  "domain": "face_recognition",
  "timestamp": "20250630_110815_180",
  "source_file": "cap_20250630_110815.jpg",
  "created_at": "20250703_005706_211",
  "version": "1.0"
}
```

### 검출 메타데이터
```json
{
  "domain": "face_recognition",
  "object_id": "20250630_110815_01",
  "detection_confidence": 0.80,
  "bbox": [x, y, w, h],
  "object_type": "face",
  "original_capture": "cap_20250630_110815.jpg"
}
```

### 그룹 메타데이터
```json
{
  "group_id": 1,
  "face_count": 2,
  "similarity_threshold": 0.6,
  "created_at": "20250703_005706_211",
  "faces": [
    {
      "index": 1,
      "original_file": "fr_face_20250630_110815_01_conf0.80.jpg",
      "saved_as": "fr_group_001_face_01.jpg"
    }
  ]
}
```

## 사용법

### 1. 캡처 파일명 생성
```python
from shared.vision_core.naming import UniversalNamingSystem

timestamp = "20250630_110815_180"
filename = UniversalNamingSystem.create_capture_filename(timestamp)
# 결과: "cap_20250630_110815.jpg"
```

### 2. 검출 파일명 생성
```python
filename = UniversalNamingSystem.create_detection_filename(
    domain='face_recognition',
    timestamp="20250630_110815_180",
    sequence=1,
    confidence=0.80
)
# 결과: "fr_face_20250630_110815_01_conf0.80.jpg"
```

### 3. 그룹 파일명 생성
```python
filename = UniversalNamingSystem.create_group_filename(
    domain='face_recognition',
    group_id=1,
    sequence=1
)
# 결과: "fr_group_001_face_01.jpg"
```

## 장점

### 1. 일관성
- 모든 도메인에서 동일한 패턴 사용
- 파일명만으로도 처리 단계 파악 가능

### 2. 확장성
- 새로운 도메인 추가 시 간단한 설정만으로 적용
- 기존 코드 수정 없이 확장 가능

### 3. 추적성
- 각 단계에서 원본 정보 보존
- 파일 간 관계 추적 용이

### 4. 가독성
- 의미 있는 접두사와 구조
- 사람이 읽기 쉬운 형태

## 향후 확장 계획

### 1. 추가 도메인 지원
- **AD**: 자동차 검사 (Automotive Detection)
- **MD**: 의료 영상 (Medical Detection)
- **SD**: 보안 감시 (Security Detection)

### 2. 고급 기능
- 파일명 압축 (긴 파일명 단축)
- 해시 기반 식별자
- 버전 관리 지원

### 3. 자동화
- 파일명 검증 자동화
- 메타데이터 일관성 검사
- 네이밍 규칙 위반 감지 