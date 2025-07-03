# 얼굴인식 도메인 데이터 관리 가이드

## 📁 개요

`data/domains/face_recognition/` 폴더는 얼굴인식 시스템의 모든 런타임 데이터를 저장하는 전용 공간입니다. 
이 폴더는 얼굴 검출부터 인식, 등록까지의 전체 과정에서 생성되는 모든 데이터를 체계적으로 관리합니다.

## 🎯 폴더의 목적

- **얼굴 데이터 관리**: 검출된 얼굴 이미지들의 체계적 저장
- **처리 과정 추적**: 각 단계별 처리 결과의 명확한 구분
- **품질 관리**: 품질 검증 및 거부된 데이터의 관리
- **데이터 플로우**: 원본부터 최종 등록까지의 완전한 추적

## 📂 폴더 구조

```
face_recognition/
├── detected_faces/            # 감지된 얼굴 이미지들
│   ├── auto_collected/        # AI 자동 수집
│   ├── from_captured/         # 카메라 캡처에서 추출
│   ├── from_uploads/          # 업로드 파일에서 추출
│   └── from_manual/           # 수동 선택
├── processed/                 # 처리된 데이터
│   ├── embeddings/            # 얼굴 특징 벡터
│   ├── final/                 # 최종 처리 완료
│   └── registered/            # 시스템 등록 완료
├── raw_input/                 # 원본 입력 데이터
│   ├── captured/              # 카메라 캡처
│   └── uploads/               # 사용자 업로드
└── staging/                   # 임시 처리 데이터
    ├── grouped/               # AI 그룹핑 완료
    ├── named/                 # 이름 지정 완료
    └── rejected/              # 품질 검증 실패
```

## 🔄 데이터 생명주기

### 1단계: 데이터 입력
- **카메라 캡처**: 실시간 카메라로 얼굴 촬영
- **파일 업로드**: 사용자가 이미지 파일 업로드
- **수동 선택**: 사용자가 화면에서 얼굴 직접 선택

### 2단계: 얼굴 검출
- **자동 검출**: AI가 자동으로 얼굴 영역 감지
- **수동 검출**: 사용자가 수동으로 얼굴 선택
- **품질 평가**: 얼굴 품질 점수 계산

### 3단계: 데이터 처리
- **그룹핑**: 유사한 얼굴들을 AI가 자동 그룹화
- **이름 지정**: 사용자가 각 얼굴에 이름 부여
- **임베딩 추출**: 얼굴 특징 벡터 생성

### 4단계: 최종 등록
- **품질 검증**: 최종 품질 기준 통과 확인
- **중복 검사**: 기존 등록된 얼굴과 중복 확인
- **시스템 등록**: 얼굴인식 시스템에 최종 등록

## 📊 폴더별 상세 설명

### detected_faces/ - 감지된 얼굴 이미지들

이 폴더는 얼굴 검출 알고리즘이 찾아낸 모든 얼굴 이미지를 저장합니다.

#### auto_collected/
- **목적**: AI가 자동으로 감지한 얼굴들
- **파일명**: `face_{timestamp}_{confidence}.jpg`
- **메타데이터**: `metadata.json` (검출 시간, 신뢰도, 위치 정보)
- **특징**: 실시간 처리 중 자동으로 수집

#### from_captured/
- **목적**: 카메라로 캡처한 전체 이미지에서 추출한 얼굴들
- **파일명**: `face_{original_frame}_{face_index}_{confidence}.jpg`
- **연결 정보**: 원본 프레임과의 연결 정보 포함
- **특징**: 's'키로 저장한 프레임에서 얼굴 추출

#### from_uploads/
- **목적**: 사용자가 업로드한 이미지에서 추출한 얼굴들
- **파일명**: `face_{upload_file}_{face_index}_{confidence}.jpg`
- **연결 정보**: 업로드 파일과의 연결 정보 포함
- **특징**: 배치 처리로 여러 얼굴 동시 추출 가능

#### from_manual/
- **목적**: 사용자가 수동으로 선택한 얼굴들
- **파일명**: `face_{person_name}_{timestamp}_{confidence}.jpg`
- **특징**: 즉시 이름이 지정되어 바로 처리 가능

### processed/ - 처리된 데이터

이 폴더는 최종 처리된 데이터들을 저장합니다.

#### embeddings/
- **목적**: 얼굴 특징 벡터 (임베딩) 저장
- **파일명**: `embedding_{face_id}.json`
- **내용**: 512차원 벡터 + 메타데이터
- **용도**: 얼굴 인식 및 비교에 사용

#### final/
- **목적**: 최종 처리 완료된 데이터
- **파일명**: `{face_id}_final.json`
- **내용**: 완전한 얼굴 정보 + 임베딩 + 품질 점수
- **특징**: 시스템 등록 준비 완료 상태

#### registered/
- **목적**: 시스템에 등록된 데이터
- **파일명**: `{face_id}_registered.json`
- **내용**: 등록 완료된 얼굴 정보
- **특징**: 실제 인식 시스템에서 사용 가능

### raw_input/ - 원본 입력 데이터

이 폴더는 원본 입력 데이터를 저장합니다.

#### captured/
- **목적**: 카메라로 캡처한 전체 프레임
- **파일명**: `frame_{timestamp}.jpg`
- **메타데이터**: 카메라 설정, 타임스탬프
- **특징**: 's'키로 저장한 전체 화면

#### uploads/
- **목적**: 사용자가 업로드한 원본 파일
- **파일명**: `{original_filename}_{timestamp}.{ext}`
- **특징**: 원본 파일명 보존, 타임스탬프 추가

### staging/ - 임시 처리 데이터

이 폴더는 처리 과정의 중간 단계 데이터를 저장합니다.

#### grouped/
- **목적**: AI가 유사한 얼굴들로 그룹화한 결과
- **파일명**: `group_{group_id}_{face_count}.json`
- **내용**: 그룹 정보 + 포함된 얼굴들
- **특징**: 자동 그룹핑 결과

#### named/
- **목적**: 이름이 지정된 얼굴들
- **파일명**: `{person_name}_{timestamp}_{confidence}.jpg`
- **메타데이터**: 이름, 타임스탬프, 신뢰도
- **특징**: 사용자 입력 이름 포함

#### rejected/
- **목적**: 품질 검증에서 거부된 얼굴들
- **파일명**: `rejected_{reason}_{timestamp}.jpg`
- **특징**: 거부 사유 기록, 재처리 가능

## 🎮 사용 방법

### 1. 얼굴 캡처
```bash
# 카메라로 얼굴 캡처
python domains/face_recognition/runners/data_collection/run_enhanced_face_capture.py

# 키보드 단축키:
# 'c': 수동 얼굴 캡처 (이름 즉시 입력)
# 's': 전체 프레임 저장
# 'q': 종료
```

### 2. 배치 처리
```bash
# 업로드된 파일들 배치 처리
python domains/face_recognition/runners/data_collection/run_unified_batch_processor.py

# AI 그룹핑 처리
python domains/face_recognition/runners/data_collection/run_unified_ai_grouping_processor.py
```

### 3. 저장소 관리
```bash
# 저장소 상태 확인 및 관리
python domains/face_recognition/runners/management/run_storage_manager.py
```

## ⚠️ 주의사항

### 데이터 보호
1. **개인정보 보호**: 얼굴 이미지는 개인정보이므로 안전하게 관리
2. **GDPR 준수**: 유럽 사용자의 경우 동의 필요
3. **데이터 암호화**: 민감한 데이터는 암호화 저장

### 파일 관리
1. **정기 정리**: 오래된 임시 파일 정기 삭제
2. **백업**: 중요한 데이터는 정기 백업
3. **권한 관리**: 적절한 파일 접근 권한 설정

### 성능 최적화
1. **이미지 압축**: 저장 공간 절약을 위한 압축
2. **캐시 활용**: 자주 사용되는 데이터 캐시
3. **배치 처리**: 대량 데이터는 배치로 처리

## 🔧 유틸리티 스크립트

### 데이터 정리
```bash
# 임시 파일 정리
python scripts/maintenance/cleanup_face_data.py

# 품질 검증
python scripts/utilities/validate_face_quality.py
```

### 데이터 분석
```bash
# 데이터 통계 확인
python scripts/utilities/analyze_face_data.py

# 품질 분석
python scripts/utilities/analyze_face_quality.py
```

## 📈 모니터링

### 데이터 품질 지표
- **검출률**: 얼굴 검출 성공률
- **품질 점수**: 평균 얼굴 품질 점수
- **등록률**: 최종 등록 성공률
- **중복률**: 중복 얼굴 비율

### 성능 지표
- **처리 속도**: 초당 처리 가능한 얼굴 수
- **정확도**: 얼굴 인식 정확도
- **메모리 사용량**: 데이터 처리 중 메모리 사용량

## 🔗 관련 문서

- [데이터 플로우 가이드](../../../docs/guides/DATA_FLOW_GUIDE.md)
- [배치 처리 가이드](../../../docs/guides/BATCH_PROCESSING_GUIDE.md)
- [얼굴인식 도메인 가이드](../../../domains/face_recognition/README.md)

---

**마지막 업데이트**: 2025-06-29  
**버전**: 1.0  
**관리자**: 얼굴인식 도메인 관리자 