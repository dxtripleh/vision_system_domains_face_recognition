# Interfaces - 사용자 인터페이스

## 📖 개요

Interfaces 폴더는 Vision System과 사용자가 상호작용할 수 있는 다양한 인터페이스를 제공합니다.

## 📁 폴더 구조

```
interfaces/
├── web/                  # 🌐 웹 인터페이스
├── cli/                  # 💻 명령줄 인터페이스
└── api/                  # 🔌 REST API
```

## 🌐 웹 인터페이스 (web/)

### Flask 기반 웹 애플리케이션
```bash
# 웹 서버 시작
python scripts/interfaces/web/app.py

# 개발 모드로 시작
python scripts/interfaces/web/app.py --debug

# 특정 포트로 시작
python scripts/interfaces/web/app.py --port 5000
```

### 기능
- **이미지 업로드**: 웹을 통한 이미지 파일 업로드
- **실시간 분석**: 업로드된 이미지의 얼굴 분석
- **인물 관리**: 웹 UI를 통한 인물 등록/조회/삭제
- **대시보드**: 시스템 통계 및 성능 모니터링
- **반응형 UI**: Bootstrap 5 기반 모던 UI

### 접근 주소
- 메인 페이지: `http://localhost:5000`
- 업로드 페이지: `http://localhost:5000/upload`
- API 문서: `http://localhost:5000/api/docs`

## 💻 명령줄 인터페이스 (cli/)

### 얼굴인식 CLI 도구
```bash
# CLI 도구 실행
python scripts/interfaces/cli/run_face_recognition_cli.py

# 특정 모드로 실행
python scripts/interfaces/cli/run_face_recognition_cli.py --mode recognition

# 도움말
python scripts/interfaces/cli/run_face_recognition_cli.py --help
```

### 주요 명령
- **인물 등록**: 새로운 인물의 얼굴 데이터 등록
- **얼굴 식별**: 입력 이미지에서 등록된 인물 찾기
- **얼굴 검증**: 특정 인물과의 얼굴 일치 여부 확인
- **인물 관리**: 등록된 인물 목록 조회 및 관리
- **시스템 정보**: 현재 시스템 상태 및 통계

## 🔌 REST API (api/)

### FastAPI 기반 REST API 서버
```bash
# API 서버 시작
python scripts/interfaces/api/face_recognition_api.py

# 개발 모드
python scripts/interfaces/api/face_recognition_api.py --reload

# 특정 포트
python scripts/interfaces/api/face_recognition_api.py --port 8000
```

### API 엔드포인트

#### 얼굴 검출
```http
POST /api/v1/detect
Content-Type: multipart/form-data

# 이미지 파일 업로드하여 얼굴 검출
```

#### 얼굴 인식
```http
POST /api/v1/recognize
Content-Type: multipart/form-data

# 이미지에서 등록된 인물 식별
```

#### 인물 등록
```http
POST /api/v1/persons
Content-Type: multipart/form-data

# 새로운 인물 등록
```

#### 인물 목록
```http
GET /api/v1/persons

# 등록된 모든 인물 목록 조회
```

#### 시스템 상태
```http
GET /api/v1/health

# 시스템 상태 확인
```

### API 문서
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 🔧 설정 및 사용법

### 환경 변수
```bash
# 웹 서버 설정
export FLASK_ENV=development
export FLASK_PORT=5000

# API 서버 설정  
export API_PORT=8000
export API_HOST=0.0.0.0

# 업로드 설정
export MAX_UPLOAD_SIZE=16MB
export ALLOWED_EXTENSIONS=jpg,jpeg,png
```

### 보안 설정
```bash
# CORS 설정 (개발용)
export CORS_ORIGINS="http://localhost:3000,http://localhost:5000"

# API 인증 (프로덕션용)
export API_SECRET_KEY=your-secret-key
export ENABLE_AUTH=true
```

## 📊 사용 예시

### 웹 인터페이스 사용
1. 웹 서버 시작: `python scripts/interfaces/web/app.py`
2. 브라우저에서 `http://localhost:5000` 접속
3. 이미지 업로드 페이지에서 얼굴 이미지 업로드
4. 분석 결과 확인

### CLI 사용
1. CLI 도구 실행: `python scripts/interfaces/cli/run_face_recognition_cli.py`
2. 메뉴에서 원하는 기능 선택
3. 가이드에 따라 이미지 경로 입력
4. 결과 확인

### API 사용 (curl)
```bash
# 얼굴 검출
curl -X POST "http://localhost:8000/api/v1/detect" \
     -F "image=@path/to/image.jpg"

# 인물 등록
curl -X POST "http://localhost:8000/api/v1/persons" \
     -F "name=John Doe" \
     -F "images=@path/to/face1.jpg" \
     -F "images=@path/to/face2.jpg"
```

## 🚫 주의사항

1. **포트 충돌**: 다른 서비스와 포트 충돌 주의
2. **파일 크기**: 대용량 이미지 업로드 시 타임아웃 가능
3. **보안**: 프로덕션 환경에서는 적절한 인증/권한 설정 필요
4. **성능**: 대량 요청 시 리소스 사용량 모니터링

## 🤝 기여하기

새로운 인터페이스를 추가할 때:
1. 적절한 하위 폴더에 배치
2. 일관된 API 설계
3. 에러 처리 및 검증
4. 문서화 및 예시 제공
5. 보안 고려사항 검토 