# 특징점을 지원하는 얼굴 검출 모델

## 🎯 특징점 지원 모델 목록

### 1. RetinaFace (5점 특징점)
- **특징점**: 5개 (양쪽 눈, 코, 양쪽 입꼬리)
- **출력 형태**: 
  - `boxes`: (N, 4) - 바운딩 박스
  - `scores`: (N,) - 신뢰도
  - `landmarks`: (N, 5, 2) - 5점 특징점
- **장점**: 높은 정확도, 특징점 정확함
- **단점**: 계산량 많음
- **파일명 패턴**: `face_detection_retinaface_*.onnx`

### 2. SCRFD (5점 특징점)
- **특징점**: 5개 (양쪽 눈, 코, 양쪽 입꼬리)
- **출력 형태**: RetinaFace와 동일
- **장점**: RetinaFace보다 빠름, 정확도 유지
- **단점**: 모델 크기 큼
- **파일명 패턴**: `face_detection_scrfd_*.onnx`

### 3. YOLO-Face (5점 특징점)
- **특징점**: 5개 (양쪽 눈, 코, 양쪽 입꼬리)
- **출력 형태**: YOLO 형식 + 특징점
- **장점**: 실시간 처리 가능
- **단점**: 작은 얼굴 검출 성능 떨어짐
- **파일명 패턴**: `face_detection_yolo_*.onnx`

### 4. BlazeFace (6점 특징점)
- **특징점**: 6개 (양쪽 눈, 코, 양쪽 귀, 입)
- **출력 형태**: MediaPipe 형식
- **장점**: 매우 빠름, 모바일 최적화
- **단점**: 정확도 상대적으로 낮음
- **파일명 패턴**: `face_detection_blazeface_*.onnx`

### 5. MTCNN (5점 특징점)
- **특징점**: 5개 (양쪽 눈, 코, 양쪽 입꼬리)
- **출력 형태**: 3단계 검출
- **장점**: 높은 정확도
- **단점**: 느림, 복잡함
- **파일명 패턴**: `face_detection_mtcnn_*.onnx`

## ❌ 특징점을 지원하지 않는 모델

### 1. UltraFace
- **특징점**: 없음
- **출력**: 바운딩 박스만
- **해결책**: 특징점 생성 함수 사용

### 2. Haar Cascade
- **특징점**: 없음
- **출력**: 바운딩 박스만
- **해결책**: 특징점 생성 함수 사용

### 3. HOG (Histogram of Oriented Gradients)
- **특징점**: 없음
- **출력**: 바운딩 박스만
- **해결책**: 특징점 생성 함수 사용

## 🔧 특징점 생성 함수

특징점을 지원하지 않는 모델을 위한 특징점 생성 함수:

```python
def generate_landmarks_from_bbox(x: int, y: int, w: int, h: int) -> List[List[int]]:
    """바운딩 박스에서 특징점 생성 (5점)"""
    landmarks = []
    
    # 1. 왼쪽 눈 (얼굴의 1/4 지점)
    left_eye_x = x + int(w * 0.25)
    left_eye_y = y + int(h * 0.35)
    landmarks.append([left_eye_x, left_eye_y])
    
    # 2. 오른쪽 눈 (얼굴의 3/4 지점)
    right_eye_x = x + int(w * 0.75)
    right_eye_y = y + int(h * 0.35)
    landmarks.append([right_eye_x, right_eye_y])
    
    # 3. 코 (얼굴의 중앙, 약간 아래)
    nose_x = x + int(w * 0.5)
    nose_y = y + int(h * 0.55)
    landmarks.append([nose_x, nose_y])
    
    # 4. 왼쪽 입꼬리
    left_mouth_x = x + int(w * 0.35)
    left_mouth_y = y + int(h * 0.75)
    landmarks.append([left_mouth_x, left_mouth_y])
    
    # 5. 오른쪽 입꼬리
    right_mouth_x = x + int(w * 0.65)
    right_mouth_y = y + int(h * 0.75)
    landmarks.append([right_mouth_x, right_mouth_y])
    
    return landmarks
```

## 📊 모델 비교표

| 모델 | 특징점 | 속도 | 정확도 | 모델 크기 | GPU 필요 |
|------|--------|------|--------|-----------|----------|
| RetinaFace | ✅ 5점 | 중간 | 높음 | 큼 | 권장 |
| SCRFD | ✅ 5점 | 빠름 | 높음 | 중간 | 권장 |
| YOLO-Face | ✅ 5점 | 빠름 | 중간 | 중간 | 선택 |
| BlazeFace | ✅ 6점 | 매우 빠름 | 중간 | 작음 | 불필요 |
| MTCNN | ✅ 5점 | 느림 | 높음 | 중간 | 권장 |
| UltraFace | ❌ | 빠름 | 중간 | 작음 | 불필요 |
| Haar | ❌ | 빠름 | 낮음 | 작음 | 불필요 |

## 🎯 권장사항

### GPU 환경
1. **RetinaFace ResNet50** - 최고 정확도, 특징점 지원
2. **SCRFD** - 균형잡힌 성능
3. **YOLO-Face** - 실시간 처리

### CPU 환경
1. **UltraFace** - 빠른 처리
2. **Haar Cascade** - 안정적
3. **BlazeFace** - 모바일 최적화

### 특징점이 중요한 경우
1. **RetinaFace** - 가장 정확한 특징점
2. **SCRFD** - 빠르면서 정확
3. **MTCNN** - 높은 정확도 