# Humanoid Face Recognition

인간형 로봇/시스템을 위한 얼굴인식 도메인입니다. ONNX 기반 얼굴인식 모델을 사용하며, Jetson/GPU/CPU 환경 자동 최적화 및 크로스 플랫폼을 지원합니다.

## 🏗️ 구조
```
domains/humanoid/face_recognition/
├── __init__.py
├── model.py
├── run.py
├── test_model.py
└── README.md
```

## 🎯 기능
- ONNX 기반 얼굴 임베딩 추출
- 실시간(카메라) 및 이미지 파일 입력 지원
- Jetson/GPU/CPU 자동 감지 및 최적화
- 시각화(프레임 표시), 임베딩 shape 로그
- 단위 테스트 코드 제공

## 🚀 사용법
### 1. 실시간 얼굴인식
```bash
python domains/humanoid/face_recognition/run.py --source 0
```
### 2. 이미지 파일 얼굴인식
```bash
python domains/humanoid/face_recognition/run.py --source path/to/image.jpg
```
### 3. 단위 테스트
```bash
python domains/humanoid/face_recognition/test_model.py
python domains/humanoid/face_recognition/test_model.py --unittest
```

## 🔧 설정
- 기본 모델: `models/weights/face_recognition_arcface_r100_20250628.onnx`
- 입력 크기: (112, 112)
- 정규화: 0~1

## ⚠️ 주의사항
- ONNX 얼굴인식 모델 필요(ArcFace 등)
- 카메라 연결/권한 필요
- Jetson/GPU 환경은 드라이버/라이브러리 설치 필요

## 🔗 관련 문서
- [프로젝트 개요](../../../README.md)
- [크로스 플랫폼 호환성 규칙](../../../.cursor/rules/CROSS_PLATFORM_COMPATIBILITY.mdc)
- [공통 모듈](../../../shared/README.md) 