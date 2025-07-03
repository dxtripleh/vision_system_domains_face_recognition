# Data 폴더 구조 상세 가이드

## 📋 목차

1. [전체 구조 개요](#전체-구조-개요)
2. [도메인별 데이터 구조](#도메인별-데이터-구조)
3. [공통 데이터 구조](#공통-데이터-구조)
4. [파일 네이밍 규칙](#파일-네이밍-규칙)
5. [데이터 플로우](#데이터-플로우)
6. [권한 및 접근](#권한-및-접근)

## 전체 구조 개요

```
data/
├── domains/                    # 도메인별 독립 데이터
│   └── face_recognition/      # 얼굴인식 도메인
├── logs/                      # 시스템 로그
├── runtime/                   # 런타임 전용 데이터
├── shared/                    # 공유 데이터
└── backups/                   # 백업 데이터
```

## 도메인별 데이터 구조

### face_recognition 도메인

```
domains/face_recognition/
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

#### detected_faces/ 폴더 상세

**목적**: 얼굴 검출 알고리즘이 찾아낸 얼굴 이미지들을 저장

- **auto_collected/**: AI가 자동으로 감지한 얼굴들
  - 파일명: `face_{timestamp}_{confidence}.jpg`
  - 메타데이터: `metadata.json`

- **from_captured/**: 카메라로 캡처한 전체 이미지에서 추출한 얼굴들
  - 파일명: `face_{original_frame}_{face_index}_{confidence}.jpg`
  - 원본 프레임과의 연결 정보 포함

- **from_uploads/**: 사용자가 업로드한 이미지에서 추출한 얼굴들
  - 파일명: `face_{upload_file}_{face_index}_{confidence}.jpg`
  - 업로드 파일과의 연결 정보 포함

- **from_manual/**: 사용자가 수동으로 선택한 얼굴들
  - 파일명: `face_{person_name}_{timestamp}_{confidence}.jpg`
  - 즉시 이름이 지정됨

#### processed/ 폴더 상세

**목적**: 최종 처리된 데이터들을 저장

- **embeddings/**: 얼굴 특징 벡터 (임베딩)
  - 파일명: `embedding_{face_id}.json`
  - 내용: 512차원 벡터 + 메타데이터

- **final/**: 최종 처리 완료된 데이터
  - 파일명: `{face_id}_final.json`
  - 내용: 완전한 얼굴 정보 + 임베딩 + 품질 점수

- **registered/**: 시스템에 등록된 데이터
  - 파일명: `{face_id}_registered.json`
  - 내용: 등록 완료된 얼굴 정보

#### raw_input/ 폴더 상세

**목적**: 원본 입력 데이터 저장

- **captured/**: 카메라로 캡처한 전체 프레임
  - 파일명: `frame_{timestamp}.jpg`
  - 메타데이터: 카메라 설정, 타임스탬프

- **uploads/**: 사용자가 업로드한 원본 파일
  - 파일명: `{original_filename}_{timestamp}.{ext}`
  - 원본 파일명 보존

#### staging/ 폴더 상세

**목적**: 처리 과정의 중간 단계 데이터

- **grouped/**: AI가 유사한 얼굴들로 그룹화한 결과
  - 파일명: `group_{group_id}_{face_count}.json`
  - 내용: 그룹 정보 + 포함된 얼굴들

- **named/**: 이름이 지정된 얼굴들
  - 파일명: `{person_name}_{timestamp}_{confidence}.jpg`
  - 메타데이터: 이름, 타임스탬프, 신뢰도

- **rejected/**: 품질 검증에서 거부된 얼굴들
  - 파일명: `rejected_{reason}_{timestamp}.jpg`
  - 거부 사유 기록

## 공통 데이터 구조

### logs/ 폴더

```
logs/
├── face_recognition/          # 얼굴인식 관련 로그
│   ├── detection.log          # 얼굴 검출 로그
│   ├── recognition.log        # 얼굴 인식 로그
│   └── registration.log       # 등록 과정 로그
├── system/                    # 시스템 전체 로그
│   ├── startup.log            # 시스템 시작 로그
│   ├── error.log              # 오류 로그
│   └── performance.log        # 성능 로그
└── access/                    # 접근 로그
    ├── api_access.log         # API 접근 로그
    └── file_access.log        # 파일 접근 로그
```

### runtime/ 폴더

```
runtime/
├── logs/                      # 실행 중 로그
│   ├── current_session.log    # 현재 세션 로그
│   └── temp_*.log             # 임시 로그 파일들
├── output/                    # 실행 결과물
│   ├── captured_frames/       # 캡처된 프레임들
│   └── recognition_results/   # 인식 결과들
└── temp/                      # 임시 파일들
    ├── backups/               # 임시 백업
    └── test_data/             # 테스트 데이터
```

### shared/ 폴더

```
shared/
├── cache/                     # 캐시 데이터
│   ├── model_cache/           # 모델 캐시
│   └── image_cache/           # 이미지 캐시
└── models/                    # 임시 모델 파일들
    ├── downloaded/            # 다운로드된 모델들
    └── converted/             # 변환된 모델들
```

### backups/ 폴더

```
backups/
├── migration_20250629_213809/ # 마이그레이션 백업
├── restructure_20250629_214500/ # 재구성 백업
└── backup_info.json           # 백업 정보
```

## 파일 네이밍 규칙

### 기본 규칙

1. **소문자 사용**: 모든 파일명은 소문자로 작성
2. **언더스코어 구분**: 단어 간 언더스코어(_) 사용
3. **타임스탬프 포함**: 중요한 파일에는 타임스탬프 포함
4. **확장자 명시**: 파일 확장자를 명확히 표시

### 도메인별 네이밍

#### 얼굴인식 도메인

- **얼굴 이미지**: `face_{identifier}_{confidence}.jpg`
- **임베딩**: `embedding_{face_id}.json`
- **메타데이터**: `metadata.json`
- **그룹 정보**: `group_{group_id}_{count}.json`

#### 로그 파일

- **일반 로그**: `{component}.log`
- **타임스탬프 로그**: `{component}_{date}.log`
- **임시 로그**: `temp_{timestamp}.log`

#### 백업 파일

- **마이그레이션**: `migration_{date}_{time}/`
- **재구성**: `restructure_{date}_{time}/`
- **정보 파일**: `backup_info.json`

## 데이터 플로우

### 1. 얼굴 검출 플로우

```
카메라 입력 → 얼굴 검출 → detected_faces/auto_collected/
                                    ↓
                            품질 검증 → staging/grouped/
                                    ↓
                            이름 입력 → staging/named/
                                    ↓
                            임베딩 추출 → processed/embeddings/
                                    ↓
                            최종 처리 → processed/final/
                                    ↓
                            시스템 등록 → domains/face_recognition/data/storage/
```

### 2. 파일 업로드 플로우

```
파일 업로드 → raw_input/uploads/
                    ↓
            얼굴 검출 → detected_faces/from_uploads/
                    ↓
            AI 그룹핑 → staging/grouped/
                    ↓
            이름 입력 → staging/named/
                    ↓
            임베딩 추출 → processed/embeddings/
                    ↓
            최종 처리 → processed/final/
                    ↓
            시스템 등록 → domains/face_recognition/data/storage/
```

### 3. 수동 캡처 플로우

```
수동 캡처 → raw_input/captured/
                ↓
        얼굴 검출 → detected_faces/from_captured/
                ↓
        이름 입력 → staging/named/
                ↓
        임베딩 추출 → processed/embeddings/
                ↓
        최종 처리 → processed/final/
                ↓
        시스템 등록 → domains/face_recognition/data/storage/
```

## 권한 및 접근

### 폴더별 권한

- **읽기 전용**: `backups/`, `logs/`
- **읽기/쓰기**: `domains/`, `runtime/`, `shared/`
- **임시 접근**: `runtime/temp/`

### 접근 제한

1. **시스템 실행 중**: 파일 삭제 금지
2. **백업 폴더**: 수동 접근만 허용
3. **로그 파일**: 읽기 전용 (시스템만 쓰기)
4. **임시 파일**: 자동 정리 대상

### 보안 고려사항

- 민감한 데이터는 암호화 저장
- 접근 로그 기록
- 정기적인 백업 생성
- 무결성 검사 수행

---

**문서 버전**: 1.0  
**최종 업데이트**: 2025-06-29  
**작성자**: 시스템 관리자
