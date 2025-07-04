# Datasets 폴더

이 폴더는 비전 시스템의 학습 데이터를 체계적으로 관리합니다.

## 📁 폴더 구조

```
datasets/
├── __init__.py                 # Python 패키지 초기화
├── README.md                   # 이 파일
├── humanoid/                   # 인간형 도메인 데이터
│   ├── __init__.py
│   ├── raw/                    # 원본 데이터
│   │   ├── face_images/        # 얼굴 이미지
│   │   ├── emotion_images/     # 감정 이미지 (향후)
│   │   └── pose_images/        # 자세 이미지 (향후)
│   ├── processed/              # 전처리된 데이터
│   │   ├── aligned_faces/      # 정렬된 얼굴
│   │   ├── normalized/         # 정규화된 이미지
│   │   └── resized/            # 크기 조정된 이미지
│   ├── augmented/              # 데이터 증강 결과
│   │   ├── rotation/           # 회전 증강
│   │   ├── brightness/         # 밝기 증강
│   │   ├── flip/               # 좌우 반전
│   │   └── noise/              # 노이즈 추가
│   ├── annotations/            # 라벨링 데이터
│   │   ├── face_landmarks.json # 얼굴 랜드마크
│   │   ├── emotion_labels.csv  # 감정 라벨 (향후)
│   │   └── pose_keypoints.json # 자세 키포인트 (향후)
│   └── splits/                 # 데이터 분할
│       ├── train.txt           # 학습 데이터 목록
│       ├── val.txt             # 검증 데이터 목록
│       └── test.txt            # 테스트 데이터 목록
├── factory/                    # 공장 도메인 데이터 (향후)
│   ├── __init__.py
│   ├── raw/                    # 원본 데이터
│   │   ├── defect_images/      # 불량 이미지
│   │   └── normal_images/      # 정상 이미지
│   ├── processed/              # 전처리된 데이터
│   ├── augmented/              # 데이터 증강 결과
│   ├── annotations/            # 라벨링 데이터
│   │   ├── defect_labels.json  # 불량 라벨
│   │   └── bounding_boxes.json # 바운딩 박스
│   └── splits/                 # 데이터 분할
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── powerline_inspection/       # 활선 검사 도메인 데이터 (향후)
    ├── __init__.py
    ├── raw/                    # 원본 데이터
    ├── processed/              # 전처리된 데이터
    ├── augmented/              # 데이터 증강 결과
    ├── annotations/            # 라벨링 데이터
    └── splits/                 # 데이터 분할
```

## 🎯 주요 기능

### 1. 데이터 구조화
- **도메인별 분류**: humanoid, factory, powerline_inspection
- **단계별 관리**: raw → processed → augmented
- **라벨링 통합**: JSON, CSV, YOLO 형식 지원

### 2. 데이터 증강
- **기하학적 변환**: 회전, 반전, 크기 조정
- **색상 변환**: 밝기, 대비, 채도 조정
- **노이즈 추가**: 가우시안, 솔트&페퍼 노이즈

### 3. 데이터 검증
- **품질 검사**: 이미지 품질, 라벨 정확성
- **일관성 검증**: 파일명, 경로, 형식 일관성
- **통계 분석**: 클래스 분포, 데이터셋 크기

## 📝 데이터 네이밍 규칙

### 이미지 파일 네이밍
**패턴**: `{domain}_{class}_{date}_{info}_{index}.{ext}`

**예시**:
```
humanoid_face_happy_20250628_indoor_001.jpg
humanoid_face_sad_20250628_outdoor_002.jpg
factory_defect_scratch_20250628_line1_001.jpg
factory_normal_product_20250628_line2_001.jpg
```

### 증강 파일 네이밍
**패턴**: `{original_name}_{augment_type}[_{param}].{ext}`

**예시**:
```
humanoid_face_happy_20250628_indoor_001_flip.jpg
humanoid_face_happy_20250628_indoor_001_bright_120.jpg
humanoid_face_happy_20250628_indoor_001_rot_15.jpg
```

## 🔧 사용 예시

### 데이터셋 로딩
```python
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

class DatasetLoader:
    """데이터셋 로더"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.splits_path = self.dataset_path / "splits"
        self.annotations_path = self.dataset_path / "annotations"
    
    def load_split(self, split_name: str) -> List[Dict]:
        """데이터 분할 로딩"""
        
        split_file = self.splits_path / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"분할 파일을 찾을 수 없습니다: {split_file}")
        
        data_list = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(self._load_data_item(line))
        
        return data_list
    
    def _load_data_item(self, image_path: str) -> Dict:
        """데이터 아이템 로딩"""
        
        full_path = self.dataset_path / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {full_path}")
        
        # 이미지 로딩
        image = cv2.imread(str(full_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {full_path}")
        
        # 라벨 로딩 (있는 경우)
        label_path = self._get_label_path(image_path)
        label = self._load_label(label_path) if label_path else None
        
        return {
            "image_path": str(full_path),
            "image": image,
            "label": label,
            "metadata": self._extract_metadata(image_path)
        }
    
    def _get_label_path(self, image_path: str) -> Path:
        """라벨 파일 경로 생성"""
        
        # 이미지 경로에서 상대 경로 추출
        rel_path = Path(image_path)
        
        # 라벨 파일 경로 생성 (여러 형식 지원)
        label_paths = [
            self.annotations_path / f"{rel_path.stem}.json",
            self.annotations_path / f"{rel_path.stem}.csv",
            self.annotations_path / f"{rel_path.stem}.txt"
        ]
        
        for label_path in label_paths:
            if label_path.exists():
                return label_path
        
        return None
    
    def _load_label(self, label_path: Path) -> Dict:
        """라벨 파일 로딩"""
        
        if label_path.suffix == ".json":
            with open(label_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif label_path.suffix == ".csv":
            return pd.read_csv(label_path).to_dict('records')
        elif label_path.suffix == ".txt":
            with open(label_path, 'r', encoding='utf-8') as f:
                return {"text": f.read().strip()}
        
        return None
    
    def _extract_metadata(self, image_path: str) -> Dict:
        """이미지 경로에서 메타데이터 추출"""
        
        # 파일명 파싱: domain_class_date_info_index.ext
        filename = Path(image_path).stem
        parts = filename.split('_')
        
        if len(parts) >= 4:
            return {
                "domain": parts[0],
                "class": parts[1],
                "date": parts[2],
                "info": parts[3],
                "index": parts[4] if len(parts) > 4 else "001"
            }
        
        return {"filename": filename}

# 사용 예시
loader = DatasetLoader("datasets/humanoid")
train_data = loader.load_split("train")
val_data = loader.load_split("val")
test_data = loader.load_split("test")

print(f"학습 데이터: {len(train_data)}개")
print(f"검증 데이터: {len(val_data)}개")
print(f"테스트 데이터: {len(test_data)}개")
```

### 데이터 증강 파이프라인
```python
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

class DataAugmentation:
    """데이터 증강 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_methods = {
            "rotation": self._rotate_image,
            "brightness": self._adjust_brightness,
            "contrast": self._adjust_contrast,
            "flip": self._flip_image,
            "noise": self._add_noise,
            "blur": self._add_blur
        }
    
    def augment_dataset(self, input_dir: str, output_dir: str, num_augmentations: int = 5):
        """데이터셋 증강"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 목록
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        total_augmented = 0
        
        for image_file in image_files:
            # 이미지 로딩
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # 증강 수행
            augmented_images = self._augment_single_image(image, num_augmentations)
            
            # 증강된 이미지 저장
            base_name = image_file.stem
            for i, aug_image in enumerate(augmented_images):
                output_file = output_path / f"{base_name}_aug_{i:03d}.jpg"
                cv2.imwrite(str(output_file), aug_image)
                total_augmented += 1
            
            print(f"처리 완료: {image_file.name} -> {num_augmentations}개 증강")
        
        print(f"총 증강된 이미지 수: {total_augmented}")
    
    def _augment_single_image(self, image: np.ndarray, num_augmentations: int) -> List[np.ndarray]:
        """단일 이미지 증강"""
        
        augmented_images = []
        
        for _ in range(num_augmentations):
            aug_image = image.copy()
            
            # 랜덤하게 증강 방법 선택
            methods = list(self.augmentation_methods.keys())
            selected_methods = np.random.choice(
                methods, 
                size=min(3, len(methods)), 
                replace=False
            )
            
            # 선택된 방법들 적용
            for method in selected_methods:
                if np.random.random() < self.config.get(f"{method}_probability", 0.5):
                    aug_image = self.augmentation_methods[method](aug_image)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 회전"""
        angle = np.random.uniform(
            self.config.get("rotation_range", (-15, 15))[0],
            self.config.get("rotation_range", (-15, 15))[1]
        )
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """밝기 조정"""
        factor = np.random.uniform(
            self.config.get("brightness_range", (0.8, 1.2))[0],
            self.config.get("brightness_range", (0.8, 1.2))[1]
        )
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted_image
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 조정"""
        factor = np.random.uniform(
            self.config.get("contrast_range", (0.8, 1.2))[0],
            self.config.get("contrast_range", (0.8, 1.2))[1]
        )
        
        adjusted_image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted_image
    
    def _flip_image(self, image: np.ndarray) -> np.ndarray:
        """좌우 반전"""
        return cv2.flip(image, 1)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """노이즈 추가"""
        noise_type = np.random.choice(["gaussian", "salt_pepper"])
        
        if noise_type == "gaussian":
            mean = 0
            std = np.random.uniform(5, 25)
            noise = np.random.normal(mean, std, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        else:  # salt_pepper
            noisy_image = image.copy()
            prob = np.random.uniform(0.001, 0.01)
            
            # Salt noise
            salt_mask = np.random.random(image.shape[:2]) < prob
            noisy_image[salt_mask] = 255
            
            # Pepper noise
            pepper_mask = np.random.random(image.shape[:2]) < prob
            noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """블러 추가"""
        kernel_size = np.random.choice([3, 5, 7])
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

# 사용 예시
augmentation_config = {
    "rotation_range": (-15, 15),
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "rotation_probability": 0.7,
    "brightness_probability": 0.6,
    "contrast_probability": 0.6,
    "flip_probability": 0.5,
    "noise_probability": 0.3,
    "blur_probability": 0.2
}

augmenter = DataAugmentation(augmentation_config)
augmenter.augment_dataset(
    input_dir="datasets/humanoid/raw/face_images",
    output_dir="datasets/humanoid/augmented",
    num_augmentations=5
)
```

### 데이터 검증 시스템
```python
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

class DatasetValidator:
    """데이터셋 검증 클래스"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.validation_results = {}
    
    def validate_dataset(self) -> Dict:
        """전체 데이터셋 검증"""
        
        validation_results = {
            "structure": self._validate_structure(),
            "images": self._validate_images(),
            "annotations": self._validate_annotations(),
            "splits": self._validate_splits(),
            "statistics": self._calculate_statistics()
        }
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_structure(self) -> Dict:
        """폴더 구조 검증"""
        
        required_folders = ["raw", "processed", "augmented", "annotations", "splits"]
        structure_results = {"valid": True, "missing_folders": []}
        
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                structure_results["missing_folders"].append(folder)
                structure_results["valid"] = False
        
        return structure_results
    
    def _validate_images(self) -> Dict:
        """이미지 파일 검증"""
        
        image_results = {
            "valid": True,
            "total_images": 0,
            "corrupted_images": [],
            "invalid_formats": [],
            "size_issues": []
        }
        
        # 모든 이미지 파일 검사
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
            image_files.extend(self.dataset_path.rglob(f"*{ext.upper()}"))
        
        image_results["total_images"] = len(image_files)
        
        for image_file in image_files:
            # 이미지 로딩 테스트
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    image_results["corrupted_images"].append(str(image_file))
                    image_results["valid"] = False
                else:
                    # 크기 검증
                    height, width = image.shape[:2]
                    if height < 50 or width < 50:
                        image_results["size_issues"].append({
                            "file": str(image_file),
                            "size": (width, height)
                        })
            except Exception as e:
                image_results["corrupted_images"].append(str(image_file))
                image_results["valid"] = False
        
        return image_results
    
    def _validate_annotations(self) -> Dict:
        """라벨링 데이터 검증"""
        
        annotation_results = {
            "valid": True,
            "total_annotations": 0,
            "missing_annotations": [],
            "invalid_formats": [],
            "inconsistent_labels": []
        }
        
        annotations_path = self.dataset_path / "annotations"
        if not annotations_path.exists():
            annotation_results["valid"] = False
            return annotation_results
        
        # 라벨 파일 검사
        label_files = list(annotations_path.glob("*.json")) + \
                     list(annotations_path.glob("*.csv")) + \
                     list(annotations_path.glob("*.txt"))
        
        annotation_results["total_annotations"] = len(label_files)
        
        for label_file in label_files:
            try:
                if label_file.suffix == ".json":
                    with open(label_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif label_file.suffix == ".csv":
                    data = pd.read_csv(label_file)
                elif label_file.suffix == ".txt":
                    with open(label_file, 'r', encoding='utf-8') as f:
                        data = f.read().strip()
                else:
                    annotation_results["invalid_formats"].append(str(label_file))
                    annotation_results["valid"] = False
                    continue
                
                # 라벨 내용 검증 (도메인별)
                if not self._validate_label_content(data, label_file):
                    annotation_results["inconsistent_labels"].append(str(label_file))
                    annotation_results["valid"] = False
                    
            except Exception as e:
                annotation_results["missing_annotations"].append(str(label_file))
                annotation_results["valid"] = False
        
        return annotation_results
    
    def _validate_label_content(self, data: any, label_file: Path) -> bool:
        """라벨 내용 검증"""
        
        # 도메인별 검증 로직
        if "humanoid" in str(self.dataset_path):
            return self._validate_humanoid_labels(data)
        elif "factory" in str(self.dataset_path):
            return self._validate_factory_labels(data)
        
        return True
    
    def _validate_humanoid_labels(self, data: any) -> bool:
        """인간형 도메인 라벨 검증"""
        
        # 얼굴 랜드마크 검증
        if isinstance(data, dict) and "landmarks" in data:
            landmarks = data["landmarks"]
            if len(landmarks) != 5:  # 5점 랜드마크
                return False
            
            # 좌표 범위 검증
            for point in landmarks:
                if len(point) != 2 or point[0] < 0 or point[1] < 0:
                    return False
        
        return True
    
    def _validate_factory_labels(self, data: any) -> bool:
        """공장 도메인 라벨 검증"""
        
        # 불량 라벨 검증
        if isinstance(data, dict) and "defect_type" in data:
            valid_defect_types = ["scratch", "dent", "crack", "discoloration"]
            if data["defect_type"] not in valid_defect_types:
                return False
        
        return True
    
    def _validate_splits(self) -> Dict:
        """데이터 분할 검증"""
        
        splits_results = {
            "valid": True,
            "missing_splits": [],
            "empty_splits": [],
            "overlap_issues": []
        }
        
        splits_path = self.dataset_path / "splits"
        required_splits = ["train.txt", "val.txt", "test.txt"]
        
        split_files = {}
        for split_name in required_splits:
            split_file = splits_path / split_name
            if not split_file.exists():
                splits_results["missing_splits"].append(split_name)
                splits_results["valid"] = False
            else:
                with open(split_file, 'r', encoding='utf-8') as f:
                    files = [line.strip() for line in f if line.strip()]
                    split_files[split_name] = files
                    
                    if len(files) == 0:
                        splits_results["empty_splits"].append(split_name)
                        splits_results["valid"] = False
        
        # 중복 검사
        all_files = []
        for split_name, files in split_files.items():
            for file in files:
                if file in all_files:
                    splits_results["overlap_issues"].append(file)
                    splits_results["valid"] = False
                all_files.append(file)
        
        return splits_results
    
    def _calculate_statistics(self) -> Dict:
        """데이터셋 통계 계산"""
        
        stats = {
            "total_images": 0,
            "class_distribution": {},
            "size_distribution": {},
            "format_distribution": {}
        }
        
        # 이미지 파일 통계
        image_files = list(self.dataset_path.rglob("*.jpg")) + \
                     list(self.dataset_path.rglob("*.jpeg")) + \
                     list(self.dataset_path.rglob("*.png")) + \
                     list(self.dataset_path.rglob("*.bmp"))
        
        stats["total_images"] = len(image_files)
        
        for image_file in image_files:
            # 형식 분포
            ext = image_file.suffix.lower()
            stats["format_distribution"][ext] = stats["format_distribution"].get(ext, 0) + 1
            
            # 크기 분포
            try:
                image = cv2.imread(str(image_file))
                if image is not None:
                    height, width = image.shape[:2]
                    size_category = f"{width}x{height}"
                    stats["size_distribution"][size_category] = stats["size_distribution"].get(size_category, 0) + 1
            except:
                pass
            
            # 클래스 분포 (파일명에서 추출)
            filename = image_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                class_name = parts[1]
                stats["class_distribution"][class_name] = stats["class_distribution"].get(class_name, 0) + 1
        
        return stats
    
    def generate_report(self) -> str:
        """검증 리포트 생성"""
        
        if not self.validation_results:
            self.validate_dataset()
        
        report = []
        report.append("# 데이터셋 검증 리포트")
        report.append(f"데이터셋 경로: {self.dataset_path}")
        report.append("")
        
        # 구조 검증 결과
        structure = self.validation_results["structure"]
        report.append("## 1. 폴더 구조 검증")
        if structure["valid"]:
            report.append("✅ 폴더 구조가 올바릅니다.")
        else:
            report.append("❌ 폴더 구조에 문제가 있습니다.")
            for folder in structure["missing_folders"]:
                report.append(f"  - 누락된 폴더: {folder}")
        report.append("")
        
        # 이미지 검증 결과
        images = self.validation_results["images"]
        report.append("## 2. 이미지 파일 검증")
        report.append(f"총 이미지 수: {images['total_images']}")
        if images["valid"]:
            report.append("✅ 모든 이미지가 올바릅니다.")
        else:
            report.append("❌ 이미지 파일에 문제가 있습니다.")
            if images["corrupted_images"]:
                report.append(f"  - 손상된 이미지: {len(images['corrupted_images'])}개")
            if images["size_issues"]:
                report.append(f"  - 크기 문제: {len(images['size_issues'])}개")
        report.append("")
        
        # 라벨링 검증 결과
        annotations = self.validation_results["annotations"]
        report.append("## 3. 라벨링 데이터 검증")
        report.append(f"총 라벨 파일 수: {annotations['total_annotations']}")
        if annotations["valid"]:
            report.append("✅ 모든 라벨링이 올바릅니다.")
        else:
            report.append("❌ 라벨링에 문제가 있습니다.")
            if annotations["missing_annotations"]:
                report.append(f"  - 누락된 라벨: {len(annotations['missing_annotations'])}개")
            if annotations["inconsistent_labels"]:
                report.append(f"  - 일관성 문제: {len(annotations['inconsistent_labels'])}개")
        report.append("")
        
        # 분할 검증 결과
        splits = self.validation_results["splits"]
        report.append("## 4. 데이터 분할 검증")
        if splits["valid"]:
            report.append("✅ 데이터 분할이 올바릅니다.")
        else:
            report.append("❌ 데이터 분할에 문제가 있습니다.")
            if splits["missing_splits"]:
                report.append(f"  - 누락된 분할: {splits['missing_splits']}")
            if splits["empty_splits"]:
                report.append(f"  - 빈 분할: {splits['empty_splits']}")
        report.append("")
        
        # 통계 정보
        stats = self.validation_results["statistics"]
        report.append("## 5. 데이터셋 통계")
        report.append(f"총 이미지 수: {stats['total_images']}")
        report.append("클래스 분포:")
        for class_name, count in stats["class_distribution"].items():
            report.append(f"  - {class_name}: {count}개")
        report.append("")
        
        return "\n".join(report)

# 사용 예시
validator = DatasetValidator("datasets/humanoid")
validation_results = validator.validate_dataset()
report = validator.generate_report()

print(report)

# 검증 결과 저장
with open("datasets/validation_report.md", "w", encoding="utf-8") as f:
    f.write(report)
```

## 📊 데이터셋 통계

### 클래스 분포 분석
```python
def analyze_class_distribution(dataset_path: str) -> Dict:
    """클래스 분포 분석"""
    
    class_counts = {}
    dataset_path = Path(dataset_path)
    
    # 모든 이미지 파일 검사
    image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
    
    for image_file in image_files:
        # 파일명에서 클래스 추출
        filename = image_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 2:
            class_name = parts[1]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 통계 계산
    total_images = sum(class_counts.values())
    class_distribution = {
        class_name: {
            "count": count,
            "percentage": (count / total_images) * 100
        }
        for class_name, count in class_counts.items()
    }
    
    return {
        "total_images": total_images,
        "class_distribution": class_distribution,
        "imbalance_ratio": max(class_counts.values()) / min(class_counts.values()) if class_counts else 0
    }

# 사용 예시
stats = analyze_class_distribution("datasets/humanoid")
print(f"총 이미지 수: {stats['total_images']}")
print("클래스 분포:")
for class_name, info in stats["class_distribution"].items():
    print(f"  {class_name}: {info['count']}개 ({info['percentage']:.1f}%)")
print(f"불균형 비율: {stats['imbalance_ratio']:.2f}")
```

## 🚨 주의사항

### 1. 데이터 품질
- **이미지 품질**: 해상도, 노이즈, 블러 등 품질 검증
- **라벨 정확성**: 라벨링 오류 및 일관성 검증
- **데이터 중복**: 중복 이미지 및 라벨 제거

### 2. 데이터 보안
- **개인정보 보호**: 얼굴 데이터 익명화 및 암호화
- **접근 제어**: 데이터 접근 권한 관리
- **백업**: 중요 데이터 정기 백업

### 3. 성능 고려사항
- **저장 공간**: 대용량 데이터 압축 및 최적화
- **로딩 속도**: 데이터 로딩 최적화
- **메모리 사용량**: 배치 처리 및 스트리밍

### 4. 유지보수 고려사항
- **버전 관리**: 데이터셋 버전 관리 시스템
- **문서화**: 데이터셋 구조 및 사용법 문서화
- **테스트**: 정기적인 데이터셋 검증

## 📞 지원

### 문제 해결
1. **데이터 로딩 실패**: 파일 경로 및 형식 확인
2. **메모리 부족**: 배치 크기 및 이미지 크기 조정
3. **라벨링 오류**: 라벨 형식 및 내용 검증
4. **성능 저하**: 데이터 전처리 최적화

### 추가 도움말
- 각 도메인별 데이터 구조 문서 참조
- 데이터 증강 파라미터 튜닝 가이드
- 성능 벤치마크 결과 확인
- GitHub Issues에서 유사한 문제 검색

### 기여 방법
1. 새로운 데이터셋 추가 시 구조 규칙 준수
2. 데이터 품질 검증 및 문서화 포함
3. 성능 벤치마크 결과 제공
4. 사용 예시 및 가이드 포함 