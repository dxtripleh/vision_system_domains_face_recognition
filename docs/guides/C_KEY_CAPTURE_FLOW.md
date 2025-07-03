# 🎯 c키 수동 캡처 완전 플로우 가이드

## 🤔 **사용자 질문**
> "수동에서 c로 얼굴인식된 사진을 캡쳐해서 이름까지 지정한 경우는 다음 루트가 어떻게 되는거지? 좀 애매해지는거 같은데?"

## ✅ **명확한 답변**

### **c키 캡처의 특별한 특징**
c키 캡처는 **다른 방법들과 다르게** 얼굴 검출과 이름 지정이 **즉시** 이루어집니다.

## 🔄 **c키 플로우 vs 다른 방법들**

### **1️⃣ c키 수동 캡처 (즉시 처리)**
```
📹 카메라 화면에서 c키 누름
         ↓
🔍 실시간 얼굴 검출 (즉시)
         ↓
✂️ 얼굴 영역만 크롭 (즉시)
         ↓
🏷️ 이름 입력 프롬프트 (즉시)
         ↓
📂 data/domains/face_recognition/raw_input/manual/
   └── {이름}_{timestamp}_manual.jpg
         ↓
✅ 바로 품질 검증 → processed/final/
```

### **2️⃣ s키 프레임 저장 (지연 처리)**
```
📹 카메라 화면에서 s키 누름
         ↓
📸 전체 프레임 저장 (즉시)
         ↓
📂 data/domains/face_recognition/raw_input/captured/
   └── frame_{timestamp}.jpg
         ↓
🔍 나중에 얼굴 검출 → detected_faces/from_captured/
         ↓
🏷️ 나중에 이름 입력 → staging/named/
         ↓
✅ 품질 검증 → processed/final/
```

## 📊 **c키 캡처의 정확한 저장 위치**

### **즉시 저장 위치**
```
📂 data/domains/face_recognition/raw_input/manual/
├── 홍길동_20250629_214530_manual.jpg    # 얼굴만 크롭된 이미지
├── 김철수_20250629_214545_manual.jpg
└── 이영희_20250629_214600_manual.jpg
```

### **메타데이터 저장**
```json
// data/domains/face_recognition/raw_input/manual/홍길동_20250629_214530_manual.json
{
  "person_name": "홍길동",
  "capture_method": "manual_c_key",
  "timestamp": "2025-06-29T21:45:30",
  "face_bbox": [x, y, w, h],
  "confidence": 0.95,
  "camera_id": "camera_0",
  "immediate_processing": true,
  "status": "ready_for_final_processing"
}
```

## 🚀 **c키 캡처 후 즉시 처리**

### **자동 파이프라인**
```
📂 raw_input/manual/ (c키로 저장됨)
         ↓
🔍 품질 검증 (자동)
   ├── ✅ 품질 통과 → processed/final/
   └── ❌ 품질 실패 → staging/rejected/
         ↓
🧠 임베딩 추출 (자동)
         ↓
📊 domains/face_recognition/data/storage/ 등록 (자동)
```

### **최종 저장 형태**
```
📂 data/domains/face_recognition/processed/final/
└── 홍길동_20250629_214530_final.json
{
  "face_id": "uuid-c-key-001",
  "person_name": "홍길동",
  "source_method": "manual_c_key",
  "embedding": [0.1, 0.2, 0.3, ...],
  "quality_score": 0.92,
  "original_path": "raw_input/manual/홍길동_20250629_214530_manual.jpg",
  "processed_at": "2025-06-29T21:45:35"
}
```

## 🔄 **다른 방법들과의 비교**

| 방법 | 저장 위치 | 이름 지정 시점 | 처리 시점 |
|------|-----------|----------------|-----------|
| **c키 캡처** | `raw_input/manual/` | 즉시 | 즉시 |
| **s키 저장** | `raw_input/captured/` | 나중에 | 나중에 |
| **파일 업로드** | `raw_input/uploads/` | 나중에 | 나중에 |
| **자동 수집** | `detected_faces/auto_collected/` | 나중에 | 나중에 |

## ✨ **c키 캡처의 장점**

### **1. 즉시 사용 가능**
- 캡처 → 이름 지정 → 등록이 한 번에 완료
- 다른 방법들처럼 중간 단계 없음

### **2. 높은 품질**
- 실시간으로 얼굴 검출 상태 확인 후 캡처
- 사용자가 직접 최적 타이밍 선택

### **3. 간단한 워크플로우**
- 복잡한 그룹핑이나 배치 처리 불필요
- 한 명씩 정확하게 등록

## 🎯 **c키 사용 시나리오**

### **권장 상황**
```
✅ 신규 인물 1명씩 정확하게 등록
✅ 실시간 데모나 테스트
✅ 높은 품질이 필요한 경우
✅ 즉시 인식 시스템에 반영하고 싶은 경우
```

### **비권장 상황**
```
❌ 대량의 인물을 한 번에 등록
❌ 기존 사진/동영상에서 얼굴 추출
❌ 자동화된 배치 처리가 필요한 경우
```

## 🔧 **구현 예시 코드**

### **c키 캡처 핸들러**
```python
def handle_c_key_capture(frame, camera_id):
    """c키 캡처 처리"""
    
    # 1. 실시간 얼굴 검출
    faces = face_detector.detect(frame)
    if not faces:
        print("❌ 얼굴이 검출되지 않았습니다.")
        return
    
    # 2. 가장 큰 얼굴 선택
    main_face = max(faces, key=lambda f: f.bbox.area)
    
    # 3. 얼굴 크롭
    face_crop = crop_face(frame, main_face.bbox)
    
    # 4. 이름 입력 받기
    person_name = input("이름을 입력하세요: ").strip()
    if not person_name:
        print("❌ 이름이 입력되지 않았습니다.")
        return
    
    # 5. 즉시 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{person_name}_{timestamp}_manual.jpg"
    save_path = STORAGE_PATHS['raw_input']['manual'] / filename
    
    cv2.imwrite(str(save_path), face_crop)
    
    # 6. 메타데이터 저장
    metadata = {
        "person_name": person_name,
        "capture_method": "manual_c_key",
        "timestamp": datetime.now().isoformat(),
        "face_bbox": main_face.bbox.to_list(),
        "confidence": main_face.confidence,
        "camera_id": camera_id,
        "immediate_processing": True
    }
    
    metadata_path = save_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 7. 즉시 처리 파이프라인 시작
    process_manual_capture(save_path, metadata)
    
    print(f"✅ {person_name} 등록 완료!")
```

## 📋 **요약**

### **c키 캡처 = 즉시 완성형 등록**
- **저장 위치**: `data/domains/face_recognition/raw_input/manual/`
- **특징**: 얼굴 검출 + 이름 지정 + 품질 검증이 즉시 완료
- **결과**: 바로 인식 시스템에서 사용 가능

### **다른 방법들 = 단계별 처리**
- **저장 위치**: `raw_input/captured/`, `raw_input/uploads/` 등
- **특징**: 나중에 배치 처리로 그룹핑 및 이름 지정
- **결과**: 중간 단계를 거쳐 최종 등록

**c키 캡처는 "즉시 등록" 방식으로, 다른 방법들과는 완전히 다른 플로우를 가집니다!** ✨

---
*작성일: 2025-06-29*
*버전: v1.0* 