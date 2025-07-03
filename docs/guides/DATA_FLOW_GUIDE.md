# 🎯 비전 시스템 완전 데이터 흐름 가이드

## 📊 전체 시스템 개요

```
🎥 입력 소스 → 🔍 얼굴 검출 → 📂 임시 저장 → 🤖 AI 처리 → 🎯 분기 처리 → 📁 최종 저장
```

---

## 🚀 1단계: 데이터 수집 (4가지 방법)

### **방법 1: 카메라 실시간 캡처**
```
📹 카메라 스트림 → 🎯 실시간 캡처 → 📂 data/domains/face_recognition/staging/named/
```
- **메뉴**: 1번 (향상된 데이터 수집), 2번 (실시간 캡처), 3번 (통합 캡처)
- **특징**: 실시간으로 얼굴 검출 후 즉시 저장

### **방법 2: 웹 인터페이스 업로드**
```
🌐 웹 브라우저 → 📤 파일 업로드 → 📂 data/domains/face_recognition/raw_input/uploads/ → 🧠 스마트 배치 처리
```
- **메뉴**: 7번 (웹 인터페이스)
- **특징**: 드래그&드롭으로 쉽게 업로드, 자동 저장

### **방법 3: 파일 직접 복사**
```
💻 파일 탐색기 → 📁 data/domains/face_recognition/raw_input/uploads/ 복사 → 🧠 스마트 배치 처리
```
- **메뉴**: 6번 (스마트 배치 처리)
- **특징**: 대량 파일 처리에 최적화

### **방법 4: 자동 얼굴 수집기**
```
📹 카메라 → 🤖 자동 검출 → 📂 data/domains/face_recognition/detected_faces/auto_collected/ → 📂 data/domains/face_recognition/staging/named/
```
- **메뉴**: 4번 (자동 얼굴 수집기)
- **특징**: AI가 자동으로 얼굴을 찾아서 수집

---

## 🧠 2단계: 스마트 배치 처리 (자동 그룹핑)

### **AI 기반 자동 그룹핑 흐름**
```
📂 data/domains/face_recognition/raw_input/uploads/ (모든 파일)
         ↓
🔍 모든 파일에서 얼굴 검출
         ↓
🧠 임베딩 생성 (512차원 벡터)
         ↓
🤖 유사도 계산 → 자동 클러스터링
         ↓
👥 그룹별 분류 (같은 사람끼리 묶기)
         ↓
🏷️ 그룹별 한 번만 이름 지정
         ↓
📂 data/domains/face_recognition/staging/named/ (공통 허브)
```

### **예시 시나리오**
```
📁 uploads/에 10개 파일 → 총 25개 얼굴 검출
         ↓
🧠 AI 자동 그룹핑:
   • 그룹1: 홍길동 (8개 얼굴)
   • 그룹2: 김철수 (12개 얼굴)  
   • 그룹3: 이영희 (5개 얼굴)
         ↓
🏷️ 3번만 이름 입력하면 끝!
```

---

## 📂 3단계: 공통 허브 (staging/named)

### **공통 허브 역할**
```
📂 data/domains/face_recognition/staging/named/
├── 홍길동_1234567890_frame_0.jpg     # 전체 이미지
├── 홍길동_1234567890_face_0.jpg      # 얼굴 크롭
├── 홍길동_1234567890_meta_0.json     # 메타데이터
├── 김철수_1234567891_frame_0.jpg
├── 김철수_1234567891_face_0.jpg
├── 김철수_1234567891_meta_0.json
└── ...
```

### **메타데이터 구조**
```json
{
  "person_name": "홍길동",
  "timestamp": 1234567890,
  "face_index": 0,
  "bbox": [x, y, w, h],
  "confidence": 0.95,
  "source_path": "data/domains/face_recognition/raw_input/uploads/photo1.jpg",
  "source_type": "image",
  "processing_type": "smart_batch",
  "frame_path": "data/domains/face_recognition/staging/named/홍길동_1234567890_frame_0.jpg",
  "face_path": "data/domains/face_recognition/staging/named/홍길동_1234567890_face_0.jpg",
  "created_at": "2025-01-28T10:30:00"
}
```

---

## 🎯 4단계: 분기 처리 (2가지 경로)

### **분기점: data/domains/face_recognition/staging/named/**
```
📂 data/domains/face_recognition/staging/named/ (공통 허브)
         ↓
🤔 사용자 선택:
   ├── 🚀 즉시 등록 (실시간 사용)
   └── 📚 훈련용 수집 (모델 개발)
```

---

## 🚀 5A단계: 즉시 등록 경로

### **실시간 인식 시스템**
```
📂 data/domains/face_recognition/staging/named/
         ↓
🔍 기존 모델로 임베딩 추출
         ↓
📊 domains/face_recognition/data/storage/에 저장
   ├── persons/ (인물 정보)
   └── faces/ (얼굴 임베딩)
         ↓
✅ 즉시 인식 가능!
```

### **저장 구조**
```
📂 domains/face_recognition/data/storage/
├── persons/
│   ├── person_index.json
│   └── {person_id}.json
└── faces/
    ├── face_index.json
    └── {face_id}.json
```

---

## 📚 5B단계: 훈련용 수집 경로

### **지속적 학습 시스템**
```
📂 data/domains/face_recognition/staging/named/
         ↓
🔍 품질 평가 (OpenCV + NumPy)
         ↓
📊 datasets/에 저장
   ├── raw/ (원본 데이터)
   ├── processed/ (전처리)
   ├── augmented/ (증강)
   └── splits/ (train/val/test)
         ↓
🤖 새 모델 훈련 → 성능 향상
```

### **데이터셋 구조**
```
📂 datasets/face_recognition/
├── raw/
│   ├── original_images/
│   ├── face_crops/
│   └── metadata/
├── processed/
│   ├── aligned/
│   ├── normalized/
│   └── resized/
├── augmented/
│   ├── brightness/
│   ├── contrast/
│   ├── rotated/
│   └── flipped/
└── splits/
    ├── train/
    ├── validation/
    └── test/
```

---

## 🔄 6단계: 실시간 인식 시스템

### **인식 실행 흐름**
```
📹 카메라 스트림
         ↓
🔍 얼굴 검출 (RetinaFace)
         ↓
🧠 임베딩 추출 (ArcFace)
         ↓
📊 domains/face_recognition/data/storage/와 비교
         ↓
✅ 인식 결과 출력
```

### **인식 결과**
```json
{
  "detections": [
    {
      "person_name": "홍길동",
      "confidence": 0.92,
      "bbox": [x, y, w, h],
      "similarity": 0.85
    }
  ]
}
```

---

## 📊 폴더별 역할 정리

### **🎯 운영 데이터 (실시간 사용)**
```
📂 domains/face_recognition/data/storage/
├── persons/     # 등록된 인물 정보
└── faces/       # 얼굴 임베딩 (512차원)
```

### **📚 학습 데이터 (모델 개발)**
```
📂 datasets/
├── raw/         # 원본 수집 데이터
├── processed/   # 전처리된 데이터
├── augmented/   # 증강된 데이터
└── splits/      # 훈련/검증/테스트 분할
```

### **🔄 임시 데이터 (자동 정리)**
```
📂 data/runtime/temp/
├── processing_cache/    # 처리 중 캐시
├── model_outputs/       # 모델 임시 출력
└── test_data/          # 테스트 데이터

📂 data/domains/face_recognition/
├── raw_input/          # 원본 입력
│   ├── captured/       # s키 저장 프레임
│   ├── uploads/        # 웹/파일 업로드
│   └── manual/         # c키 수동 캡처
├── detected_faces/     # 얼굴 검출 결과
│   ├── auto_collected/ # 자동 수집
│   ├── from_captured/  # captured 처리
│   └── from_uploads/   # uploads 처리
├── staging/            # 처리 대기
│   ├── grouped/        # AI 그룹핑
│   ├── named/          # 이름 지정 ✨
│   └── rejected/       # 품질 실패
└── processed/          # 최종 처리
    ├── final/          # 처리 완료 ✨
    ├── embeddings/     # 임베딩
    └── registered/     # 시스템 등록
```

---

## 🎯 완전한 데이터 흐름 예시

### **시나리오: 새로운 인물 등록**

```
1️⃣ 데이터 수집
   📹 카메라로 촬영 → 📂 data/domains/face_recognition/staging/named/

2️⃣ 스마트 처리
   🧠 AI 자동 그룹핑 → 🏷️ 이름 지정

3️⃣ 분기 선택
   🤔 즉시 등록 vs 훈련용 수집

4A️⃣ 즉시 등록 경로
   📊 domains/face_recognition/data/storage/ 저장 → ✅ 실시간 인식 가능

4B️⃣ 훈련용 수집 경로
   📚 datasets/ 저장 → 🤖 향후 모델 훈련에 활용
```

### **시나리오: 대량 파일 처리**

```
1️⃣ 파일 준비
   📁 data/domains/face_recognition/raw_input/uploads/에 20개 사진 넣기

2️⃣ 스마트 배치 처리
   🔍 50개 얼굴 검출 → 🧠 3개 그룹으로 자동 분류
   🏷️ 3번만 이름 입력

3️⃣ 공통 허브로 이동
   📂 data/domains/face_recognition/staging/named/에 체계적 저장

4️⃣ 분기 처리
   🚀 즉시 등록 → 실시간 인식 시스템
   📚 훈련용 수집 → 지속적 학습 시스템
```

---

## 🔧 메뉴별 실행 흐름

### **데이터 수집 메뉴**
```
1️⃣ 향상된 데이터 수집 (카메라)
   📹 카메라 → 📂 datasets/raw/

2️⃣ 실시간 캡처 & 등록
   📹 카메라 → 📂 data/domains/face_recognition/staging/named/ → 📊 domains/face_recognition/data/storage/

3️⃣ 통합 캡처 시스템
   📹 카메라 → 📂 data/domains/face_recognition/staging/named/ → 분기 처리

4️⃣ 자동 얼굴 수집기
   📹 카메라 → 🤖 자동 검출 → 📂 data/domains/face_recognition/staging/named/

5️⃣ 배치 얼굴 처리 (기본)
   📁 파일 → 🔍 개별 처리 → 📂 data/domains/face_recognition/staging/named/

6️⃣ 스마트 배치 처리 (고급) ← 🆕
   📁 파일 → 🧠 자동 그룹핑 → 📂 data/domains/face_recognition/staging/named/
```

### **인식 실행 메뉴**
```
7️⃣ 웹 인터페이스
   🌐 브라우저 → 📤 업로드 → 🔍 즉시 인식

8️⃣ 실시간 데모
   📹 카메라 → 🔍 실시간 인식

9️⃣ 실시간 인식 시스템
   📹 카메라 → 🔍 고성능 인식

🔟 고급 인식 시스템
   📹 카메라 → 🔍 최고 성능 인식
```

---

## 🎉 시스템의 핵심 장점

### **1. 통합된 데이터 흐름**
- 모든 입력이 `data/domains/face_recognition/staging/named/`으로 통합
- 일관된 분기 처리 시스템

### **2. 자동화된 그룹핑**
- AI가 같은 사람끼리 자동으로 묶어줌
- 수동 작업 대폭 감소

### **3. 지속적 학습 지원**
- 운영 데이터와 훈련 데이터 분리
- 향후 모델 개선 가능

### **4. 유연한 입력 방식**
- 카메라, 웹 업로드, 파일 복사 모두 지원
- 사용자 편의성 극대화

### **5. 확장 가능한 구조**
- 새로운 도메인 추가 용이
- 모듈화된 설계

---

## 🚀 사용자 가이드

### **첫 사용자**
```
1. 메뉴 14번: 하드웨어 연결 확인
2. 메뉴 13번: 시스템 상태 점검  
3. 메뉴 7번: 웹 인터페이스 실행
4. 브라우저에서 http://localhost:5000 접속
5. 인물 등록 → 이미지 업로드 → 얼굴 등록
6. 메뉴 8번: 실시간 인식 테스트
```

### **대량 파일 처리**
```
1. data/domains/face_recognition/raw_input/uploads/ 폴더에 파일들 넣기
2. 메뉴 6번: 스마트 배치 처리 실행
3. AI 자동 그룹핑 확인
4. 그룹별로 이름 지정
5. 자동으로 staging/named로 이동
6. 분기 처리 선택
```

---

*최종 업데이트: 2025-06-29 v2.0 (새로운 도메인 구조 반영)* 