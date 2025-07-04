# Pipeline - 얼굴인식 파이프라인 모듈

## 📋 개요

이 폴더는 얼굴인식 기능의 데이터 처리 파이프라인을 포함합니다. 이미지 업로드부터 얼굴 검출, 특징 추출, 클러스터링, 라벨링까지의 전체 워크플로우를 관리합니다.

## 🏗️ 폴더 구조

```
pipeline/
├── __init__.py                    # 파이프라인 패키지 초기화
├── README.md                      # 이 파일
├── pipeline_manager.py            # 파이프라인 관리자
├── stages/                        # 파이프라인 단계들
│   ├── __init__.py
│   ├── upload_stage.py            # 업로드 단계
│   ├── detection_stage.py         # 검출 단계
│   ├── extraction_stage.py        # 특징 추출 단계
│   ├── clustering_stage.py        # 클러스터링 단계
│   └── labeling_stage.py          # 라벨링 단계
└── utils/                         # 파이프라인 유틸리티
    ├── __init__.py
    ├── progress_tracker.py        # 진행 상황 추적
    └── data_validator.py          # 데이터 검증
```

## 🔍 포함된 파이프라인들

### 1. Pipeline Manager (파이프라인 관리자)
- **파일**: `pipeline_manager.py`
- **목적**: 전체 파이프라인 워크플로우 관리
- **기능**: 단계별 실행, 진행 상황 모니터링, 오류 처리

### 2. Upload Stage (업로드 단계)
- **파일**: `stages/upload_stage.py`
- **목적**: 이미지 파일 업로드 및 전처리
- **기능**: 파일 검증, 형식 변환, 메타데이터 추출

### 3. Detection Stage (검출 단계)
- **파일**: `stages/detection_stage.py`
- **목적**: 이미지에서 얼굴 검출
- **기능**: 얼굴 영역 검출, 랜드마크 추출, 품질 평가

### 4. Extraction Stage (추출 단계)
- **파일**: `stages/extraction_stage.py`
- **목적**: 얼굴 이미지에서 특징 벡터 추출
- **기능**: 얼굴 정렬, 특징 추출, 임베딩 생성

### 5. Clustering Stage (클러스터링 단계)
- **파일**: `stages/clustering_stage.py`
- **목적**: 유사한 얼굴들을 그룹화
- **기능**: 유사도 계산, 클러스터링, 그룹 관리

### 6. Labeling Stage (라벨링 단계)
- **파일**: `stages/labeling_stage.py`
- **목적**: 클러스터에 신원 정보 할당
- **기능**: 신원 매칭, 라벨링, 데이터베이스 업데이트

## 🚀 사용법

### 기본 파이프라인 실행
```python
from domains.humanoid.face_recognition.pipeline import PipelineManager

# 파이프라인 관리자 초기화
pipeline = PipelineManager()

# 전체 파이프라인 실행
result = pipeline.run_full_pipeline(
    input_path="data/domains/humanoid/face_recognition/1_raw/uploads/",
    output_path="data/domains/humanoid/face_recognition/4_labeled/"
)

# 결과 확인
print(f"처리된 이미지: {result['processed_images']}")
print(f"검출된 얼굴: {result['detected_faces']}")
print(f"생성된 클러스터: {result['clusters']}")
```

### 단계별 실행
```python
# 특정 단계만 실행
pipeline.run_stage('detection', input_data)
pipeline.run_stage('clustering', detection_results)

# 단계별 결과 확인
detection_results = pipeline.get_stage_results('detection')
clustering_results = pipeline.get_stage_results('clustering')
```

### 파이프라인 모니터링
```python
# 진행 상황 모니터링
progress = pipeline.get_progress()
print(f"진행률: {progress['percentage']:.1f}%")
print(f"현재 단계: {progress['current_stage']}")

# 실시간 진행 상황 추적
pipeline.run_with_progress_callback(callback_function)
```

## 🔧 파이프라인 설정

### 파이프라인 설정 구조
```yaml
# pipeline_config.yaml
pipeline:
  name: "face_recognition_pipeline"
  version: "1.0.0"
  
  stages:
    upload:
      enabled: true
      max_file_size: 10MB
      supported_formats: ["jpg", "png", "bmp"]
      output_dir: "1_raw/uploads"
    
    detection:
      enabled: true
      confidence_threshold: 0.5
      min_face_size: 80
      output_dir: "2_extracted/features"
    
    extraction:
      enabled: true
      embedding_dim: 512
      normalize_embeddings: true
      output_dir: "2_extracted/metadata"
    
    clustering:
      enabled: true
      similarity_threshold: 0.6
      min_cluster_size: 2
      output_dir: "3_clustered/groups"
    
    labeling:
      enabled: true
      auto_labeling: false
      manual_review: true
      output_dir: "4_labeled/groups"

  performance:
    batch_size: 10
    max_workers: 4
    enable_caching: true
    
  monitoring:
    enable_progress_tracking: true
    enable_logging: true
    save_intermediate_results: true
```

## 📊 파이프라인 단계별 상세

### 1. Upload Stage (업로드 단계)
```python
class UploadStage:
    """이미지 업로드 및 전처리 단계"""
    
    def process(self, input_path: str) -> Dict:
        """업로드 처리"""
        # 파일 검증
        valid_files = self.validate_files(input_path)
        
        # 이미지 전처리
        processed_images = []
        for file_path in valid_files:
            image = self.preprocess_image(file_path)
            metadata = self.extract_metadata(file_path)
            processed_images.append({
                'path': file_path,
                'image': image,
                'metadata': metadata
            })
        
        return {
            'stage': 'upload',
            'processed_images': processed_images,
            'total_files': len(valid_files)
        }
```

### 2. Detection Stage (검출 단계)
```python
class DetectionStage:
    """얼굴 검출 단계"""
    
    def process(self, images: List[Dict]) -> Dict:
        """얼굴 검출 처리"""
        detection_results = []
        
        for image_data in images:
            faces = self.detect_faces(image_data['image'])
            
            # 얼굴 이미지 추출
            face_images = []
            for face in faces:
                face_image = self.extract_face_image(
                    image_data['image'], 
                    face['bbox']
                )
                face_images.append({
                    'original_image': image_data['path'],
                    'face_image': face_image,
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face.get('landmarks', [])
                })
            
            detection_results.extend(face_images)
        
        return {
            'stage': 'detection',
            'detected_faces': detection_results,
            'total_faces': len(detection_results)
        }
```

### 3. Extraction Stage (추출 단계)
```python
class ExtractionStage:
    """특징 추출 단계"""
    
    def process(self, face_images: List[Dict]) -> Dict:
        """특징 추출 처리"""
        embeddings = []
        
        for face_data in face_images:
            # 얼굴 정렬
            aligned_face = self.align_face(face_data['face_image'])
            
            # 특징 벡터 추출
            embedding = self.extract_embedding(aligned_face)
            
            embeddings.append({
                'face_id': face_data['face_id'],
                'embedding': embedding,
                'original_image': face_data['original_image'],
                'bbox': face_data['bbox']
            })
        
        return {
            'stage': 'extraction',
            'embeddings': embeddings,
            'embedding_dim': len(embeddings[0]['embedding']) if embeddings else 0
        }
```

### 4. Clustering Stage (클러스터링 단계)
```python
class ClusteringStage:
    """얼굴 클러스터링 단계"""
    
    def process(self, embeddings: List[Dict]) -> Dict:
        """클러스터링 처리"""
        # 유사도 매트릭스 계산
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        
        # 클러스터링 수행
        clusters = self.perform_clustering(similarity_matrix, embeddings)
        
        # 클러스터 품질 평가
        cluster_quality = self.evaluate_cluster_quality(clusters)
        
        return {
            'stage': 'clustering',
            'clusters': clusters,
            'total_clusters': len(clusters),
            'cluster_quality': cluster_quality
        }
```

### 5. Labeling Stage (라벨링 단계)
```python
class LabelingStage:
    """얼굴 라벨링 단계"""
    
    def process(self, clusters: List[Dict]) -> Dict:
        """라벨링 처리"""
        labeled_clusters = []
        
        for cluster in clusters:
            # 신원 매칭
            identity = self.match_identity(cluster)
            
            # 라벨링
            if identity:
                cluster['identity'] = identity
                cluster['labeling_method'] = 'auto'
            else:
                cluster['identity'] = 'unknown'
                cluster['labeling_method'] = 'manual_review'
            
            labeled_clusters.append(cluster)
        
        return {
            'stage': 'labeling',
            'labeled_clusters': labeled_clusters,
            'auto_labeled': len([c for c in labeled_clusters if c['labeling_method'] == 'auto']),
            'manual_review_needed': len([c for c in labeled_clusters if c['labeling_method'] == 'manual_review'])
        }
```

## 🔗 의존성

### 내부 의존성
- `../models/`: 얼굴 검출 및 인식 모델
- `../services/`: 얼굴인식 서비스
- `../utils/`: 유틸리티 함수들
- `common/`: 공통 유틸리티

### 외부 의존성
```python
# requirements.txt
scikit-learn>=1.0.0
scipy>=1.7.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
```

## 🧪 파이프라인 테스트

### 파이프라인 테스트 실행
```bash
# 전체 파이프라인 테스트
python -m pytest tests/test_pipeline.py -v

# 특정 단계 테스트
python -m pytest tests/test_pipeline.py::TestUploadStage -v
python -m pytest tests/test_pipeline.py::TestDetectionStage -v
```

### 테스트 예시
```python
def test_full_pipeline():
    """전체 파이프라인 테스트"""
    pipeline = PipelineManager()
    
    # 테스트 데이터 준비
    test_images = create_test_images(5)
    
    # 파이프라인 실행
    result = pipeline.run_full_pipeline(test_images)
    
    # 결과 검증
    assert result['processed_images'] == 5
    assert result['detected_faces'] > 0
    assert result['clusters'] > 0
    assert result['success'] is True
```

## 📈 성능 최적화

### 파이프라인 성능 최적화
```python
class OptimizedPipelineManager(PipelineManager):
    """최적화된 파이프라인 관리자"""
    
    def __init__(self, config):
        super().__init__(config)
        self.enable_parallel_processing = True
        self.batch_size = config.get('batch_size', 10)
        self.max_workers = config.get('max_workers', 4)
    
    def run_parallel_stage(self, stage_name, data):
        """병렬 처리로 단계 실행"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batches = self.create_batches(data, self.batch_size)
            futures = [executor.submit(self.run_stage, stage_name, batch) for batch in batches]
            results = [future.result() for future in futures]
        
        return self.merge_results(results)
```

## 🐛 문제 해결

### 일반적인 파이프라인 문제들

#### 1. 메모리 부족
```python
# 해결 방법: 배치 처리
def process_in_batches(self, data, batch_size=10):
    """배치 단위로 처리"""
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_result = self.process_batch(batch)
        results.extend(batch_result)
        
        # 메모리 정리
        gc.collect()
    
    return results
```

#### 2. 처리 속도가 느림
```python
# 해결 방법: 병렬 처리
def parallel_process(self, data):
    """병렬 처리"""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(self.process_item, data))
    return results
```

#### 3. 중간 결과 손실
```python
# 해결 방법: 체크포인트 저장
def save_checkpoint(self, stage_name, data):
    """체크포인트 저장"""
    checkpoint_path = f"data/pipeline/checkpoints/{stage_name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
```

## 📝 파이프라인 모니터링

### 진행 상황 추적
```python
class ProgressTracker:
    """진행 상황 추적기"""
    
    def __init__(self, total_stages):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_progress = {}
    
    def update_stage_progress(self, stage_name, progress):
        """단계별 진행 상황 업데이트"""
        self.stage_progress[stage_name] = progress
    
    def get_overall_progress(self):
        """전체 진행률 계산"""
        if not self.stage_progress:
            return 0.0
        
        total_progress = sum(self.stage_progress.values())
        return (total_progress / self.total_stages) * 100
```

## 🔗 관련 문서

- [얼굴인식 기능 문서](../README.md)
- [모델 문서](../models/README.md)
- [서비스 문서](../services/README.md)
- [Humanoid 도메인 문서](../../README.md)
- [프로젝트 전체 문서](../../../../README.md)

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README 파일 확인
2. 상위 폴더의 README.md 확인
3. 파이프라인 설정 파일 참조
4. 개발팀에 문의

---

**마지막 업데이트**: 2025-07-04
**버전**: 1.0.0
**작성자**: Vision System Team 