# 데이터 구조 재설계 가이드

## 🎯 목표
향후 불량검출, 전선검사 등 다양한 도메인 확장을 고려한 체계적인 데이터 구조 설계

## 📁 새로운 데이터 구조

### 현재 문제점
```
data/temp/
├── captured_frames/     # 얼굴인식용
├── face_staging/        # 얼굴인식용
├── auto_collected/      # 얼굴인식용
├── uploads/             # 얼굴인식용
└── ...                  # 모두 얼굴인식 관련
```

### 개선된 구조
```
data/
├── runtime/             # 런타임 데이터 (모든 도메인 공통)
│   ├── temp/           # 임시 파일 (24시간 자동 정리)
│   ├── logs/           # 로그 파일
│   └── output/         # 최종 결과물
├── domains/            # 도메인별 데이터
│   ├── face_recognition/
│   │   ├── raw_input/         # 원본 입력 (captured_frames, uploads)
│   │   ├── detected_faces/    # 얼굴 검출 결과 (auto_collected)
│   │   ├── staging/           # 이름 지정 대기 (face_staging)
│   │   └── processed/         # 최종 처리 완료
│   ├── factory_defect/        # 향후 공장 불량 검출
│   │   ├── raw_input/
│   │   ├── detected_defects/
│   │   ├── staging/
│   │   └── processed/
│   └── powerline_inspection/  # 향후 전선 검사
│       ├── raw_input/
│       ├── detected_issues/
│       ├── staging/
│       └── processed/
└── shared/             # 도메인 간 공유 데이터
    ├── models/         # 공유 모델
    └── cache/          # 공유 캐시
```

## 🔄 데이터 플로우 정의

### 얼굴인식 도메인 플로우
```
1. 입력 단계:
   - 카메라 캡처 → raw_input/captured/
   - 파일 업로드 → raw_input/uploads/
   - 자동 수집 → detected_faces/auto_collected/

2. 검출 단계:
   - raw_input/* → AI 얼굴 검출 → detected_faces/

3. 그룹핑 단계:
   - detected_faces/* → AI 그룹핑 → staging/grouped/

4. 이름 지정 단계:
   - staging/grouped/* → 사용자 이름 입력 → staging/named/

5. 최종 처리 단계:
   - staging/named/* → 품질 검증 → processed/final/
```

## 📋 폴더별 역할 정의

### Face Recognition 도메인

#### raw_input/ (원본 입력)
- `captured/` - s키로 저장된 카메라 캡처 프레임
- `uploads/` - 사용자가 직접 업로드한 파일
- `manual/` - c키로 수동 캡처한 프레임

#### detected_faces/ (얼굴 검출 결과)
- `auto_collected/` - 자동 모드에서 검출된 얼굴들
- `from_captured/` - captured에서 검출된 얼굴들  
- `from_uploads/` - uploads에서 검출된 얼굴들

#### staging/ (처리 대기)
- `grouped/` - AI 그룹핑 완료, 이름 입력 대기
- `named/` - 이름 지정 완료, 품질 검증 대기
- `rejected/` - 품질 검증 실패

#### processed/ (최종 처리 완료)
- `final/` - 최종 처리 완료된 얼굴 데이터
- `embeddings/` - 임베딩 추출 완료
- `registered/` - 시스템 등록 완료

## 🛠️ 마이그레이션 계획

### 1단계: 새 구조 생성
```python
# 새로운 폴더 구조 생성
data/domains/face_recognition/
├── raw_input/
│   ├── captured/
│   ├── uploads/
│   └── manual/
├── detected_faces/
│   ├── auto_collected/
│   ├── from_captured/
│   └── from_uploads/
├── staging/
│   ├── grouped/
│   ├── named/
│   └── rejected/
└── processed/
    ├── final/
    ├── embeddings/
    └── registered/
```

### 2단계: 기존 데이터 이동
```
data/temp/captured_frames/ → data/domains/face_recognition/raw_input/captured/
data/temp/uploads/ → data/domains/face_recognition/raw_input/uploads/
data/temp/auto_collected/ → data/domains/face_recognition/detected_faces/auto_collected/
data/temp/face_staging/ → data/domains/face_recognition/staging/named/
```

### 3단계: 코드 업데이트
- 모든 경로 참조 업데이트
- 도메인별 설정 파일 생성
- 공통 유틸리티 함수 개발

## 💡 확장 예시

### 공장 불량 검출 확장 시
```
data/domains/factory_defect/
├── raw_input/
│   ├── camera_feed/     # 실시간 카메라
│   ├── batch_images/    # 배치 이미지
│   └── manual_check/    # 수동 검사
├── detected_defects/
│   ├── scratch/         # 스크래치 검출
│   ├── dent/           # 찌그러짐 검출
│   └── crack/          # 균열 검출
├── staging/
│   ├── confirmed/      # 검증 완료
│   ├── pending/        # 검증 대기
│   └── false_positive/ # 오탐지
└── processed/
    ├── reports/        # 검사 보고서
    └── statistics/     # 통계 데이터
```

## 🔧 구현 우선순위

1. **즉시 구현**: 얼굴인식 도메인 구조 정리
2. **단기 계획**: 공통 유틸리티 개발
3. **중기 계획**: 다른 도메인 구조 준비
4. **장기 계획**: 통합 관리 시스템

이 구조를 통해 각 도메인이 독립적으로 관리되면서도 
공통 기능은 효율적으로 공유할 수 있습니다. 