# Domains

도메인별 독립 개발을 위한 폴더입니다.

## 구조

```
domains/
├── humanoid/              # 인간형 로봇 관련
│   └── face_recognition/  # 얼굴인식 도메인
├── factory/               # 공장 자동화 관련
│   └── defect_detection/  # 불량 검출 도메인 (향후)
└── infrastructure/        # 인프라 관련
    └── powerline_inspection/ # 활선 검사 도메인 (향후)
```

## 도메인 개발 규칙

1. **독립성**: 각 도메인은 독립적으로 개발되어야 합니다.
2. **표준 구조**: 각 도메인은 다음 구조를 따라야 합니다:
   ```
   <domain_name>/
   ├── __init__.py
   ├── README.md
   ├── model.py        # ONNX 모델 클래스
   ├── run.py          # 실시간 추론 실행
   └── test_model.py   # 단위 테스트
   ```
3. **의존성**: 도메인 간 직접 의존성은 금지됩니다.
4. **공통 모듈**: 공통 기능은 `shared/` 또는 `common/`을 사용합니다.

## 현재 개발 중인 도메인

- **humanoid/face_recognition**: 얼굴인식 시스템 