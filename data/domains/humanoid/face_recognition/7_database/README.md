# 7단계: 데이터베이스 관리 (humanoid/face_recognition)

## 📋 개요

이 폴더는 얼굴인식 도메인의 7단계 데이터베이스 관리 데이터를 저장합니다.
임베딩 벡터, 인식 로그, 성능 통계 등의 데이터베이스 관련 데이터들이 저장됩니다.

## 🏗️ 폴더 구조

```
7_database/
├── vectors/              # 벡터 데이터
│   ├── embeddings/       # 얼굴 임베딩 벡터
│   ├── indices/          # 검색 인덱스
│   └── metadata/         # 벡터 메타데이터
├── logs/                 # 데이터베이스 로그
│   ├── queries/          # 쿼리 로그
│   ├── transactions/     # 트랜잭션 로그
│   └── errors/           # 오류 로그
└── backups/              # 백업 데이터
    ├── daily/            # 일일 백업
    ├── weekly/           # 주간 백업
    └── monthly/          # 월간 백업
```

## 📊 데이터 형식

### 벡터 데이터
- **파일명 패턴**: `{person_id}_emb_{version}.npy`
- **예시**: `person001_emb_v1.0.npy`
- **형식**: NumPy 배열 (.npy)

### 검색 인덱스
- **파일명 패턴**: `index_{type}_{date}.{ext}`
- **예시**: `index_faiss_20250704.faiss`
- **형식**: FAISS, Annoy, HNSW

### 데이터베이스 로그
- **파일명 패턴**: `{type}_{date}.log`
- **예시**: `queries_20250704.log`
- **형식**: 텍스트 로그

### 백업 데이터
- **파일명 패턴**: `backup_{type}_{date}.{ext}`
- **예시**: `backup_daily_20250704.sql`
- **형식**: SQL, JSON, CSV

## 🔧 사용법

### 데이터베이스 관리 실행
```bash
# 7단계 데이터베이스 관리 실행
python domains/humanoid/face_recognition/run_stage_7_database.py

# 벡터 데이터베이스 초기화
python domains/humanoid/face_recognition/run_stage_7_database.py --init

# 백업 생성
python domains/humanoid/face_recognition/run_stage_7_database.py --backup

# 성능 통계 수집
python domains/humanoid/face_recognition/run_stage_7_database.py --stats
```

### 데이터베이스 쿼리
```bash
# 임베딩 벡터 검색
python domains/humanoid/face_recognition/run_stage_7_database.py --search "person001"

# 인식 히스토리 조회
python domains/humanoid/face_recognition/run_stage_7_database.py --history --days 7

# 성능 통계 조회
python domains/humanoid/face_recognition/run_stage_7_database.py --performance --date 20250704
```

## 📈 성능 지표

### 데이터베이스 성능 목표
- **쿼리 응답시간**: 10ms 이하
- **벡터 검색 속도**: 1000개/초 이상
- **동시 접속자**: 100명 이상
- **데이터 무결성**: 99.9% 이상

### 저장소 요구사항
- **벡터 저장소**: 1TB 이상
- **로그 저장소**: 100GB 이상
- **백업 저장소**: 2TB 이상

## 🔄 데이터 흐름

1. **임베딩 벡터** → `vectors/embeddings/`
2. **검색 인덱스** → `vectors/indices/`
3. **쿼리 로그** → `logs/queries/`
4. **백업 데이터** → `backups/`

## 🗄️ 데이터베이스 스키마

### Persons 테이블
```sql
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100),
    embedding_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Recognition_Logs 테이블
```sql
CREATE TABLE recognition_logs (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(50),
    camera_id INTEGER,
    confidence FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT
);
```

### Performance_Metrics 테이블
```sql
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50),
    metric_value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## ⚠️ 주의사항

- 벡터 데이터는 정기적으로 백업해야 합니다
- 로그 파일은 자동으로 로테이션됩니다 (30일 보관)
- 백업 데이터는 암호화하여 저장됩니다
- 데이터베이스 연결은 SSL/TLS를 사용합니다 