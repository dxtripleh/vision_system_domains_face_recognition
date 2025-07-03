#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
범용 네이밍 시스템 (Universal Naming System)

이 모듈은 비전 시스템의 모든 도메인에서 사용할 수 있는 통일된 파일명 및 메타데이터 규칙을 제공합니다.

## 파일명 생성 규칙

### 1. 캡처 파일명 (Capture Files)
- **패턴**: `cap_HHMMSS.jpg`
- **예시**: `cap_094500.jpg`
- **설명**: 폴더 구조(YYYYMMDD)로 날짜 관리, 파일명은 시간만 포함

### 2. 검출 파일명 (Detection Files)  
- **패턴**: `{prefix}_{type}_{seq}_{src}_{conf}.jpg`
- **예시**: `fr_face_01_cap500_85.jpg`
- **구성요소**:
  - `fr`: 도메인 접두사 (face_recognition)
  - `face`: 객체 타입
  - `01`: 순번 (같은 소스에서 검출된 객체 구분)
  - `cap500`: 원본 식별자 (cap_20250703_094500 → cap500)
  - `85`: 신뢰도 (0.85 → 85)

### 3. 그룹핑 파일명 (Grouping Files)
- **패턴**: `{prefix}_grp_{gid}_{seq}.jpg`
- **예시**: `fr_grp_001_01.jpg`
- **구성요소**:
  - `fr`: 도메인 접두사
  - `grp`: 그룹핑 타입
  - `001`: 그룹 ID (3자리)
  - `01`: 그룹 내 순번

### 4. 도메인별 접두사 규칙
- `fr`: face_recognition (얼굴인식)
- `fd`: factory_defect (공장 불량 검출)
- `pi`: powerline_inspection (전선 검사)

### 5. 원본 식별자 생성 규칙
- `cap_YYYYMMDD_HHMMSS` → `cap{SSS}` (마지막 3자리)
- `image01` → `img01` (앞 2자리)
- 기타 파일명 → 앞 3자리 사용

### 6. 폴더 구조 규칙
```
data/domains/{domain}/
├── raw_input/
│   ├── captured/
│   │   └── YYYYMMDD/         # 날짜별 폴더
│   │       └── cap_HHMMSS.jpg
│   └── uploads/
│       └── original_files.*
├── detected_faces/
│   └── YYYYMMDD/             # 날짜별 폴더
│       ├── fr_face_01_cap500_85.jpg
│       └── fr_face_01_cap500_85.json
└── staging/
    └── grouped/
        └── group_001/
            ├── fr_grp_001_01.jpg
            └── group_metadata.json
```

### 7. 향후 확장 규칙
- 새 도메인 추가 시 DOMAIN_PREFIXES에 2-3자리 접두사 추가
- 새 객체 타입 추가 시 OBJECT_TYPES에 명확한 타입명 추가
- 원본 식별자는 항상 3-4자리로 유지하여 파일명 길이 제한
- 신뢰도는 항상 정수화(0-100)하여 일관성 유지
"""

from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path

class UniversalNamingSystem:
    """범용 네이밍 시스템"""
    
    # 도메인별 접두사 정의
    DOMAIN_PREFIXES = {
        'face_recognition': 'fr',
        'factory_defect': 'fd', 
        'powerline_inspection': 'pi'
    }
    
    # 도메인별 객체 타입 정의
    OBJECT_TYPES = {
        'face_recognition': 'face',
        'factory_defect': 'defect',
        'powerline_inspection': 'issue'
    }
    
    @classmethod
    def create_capture_filename(cls, timestamp: str) -> str:
        """
        캡처 파일명 생성 (모든 도메인 공통)
        
        Args:
            timestamp: 전체 타임스탬프 (YYYYMMDD_HHMMSS_mmm)
            
        Returns:
            str: 캡처 파일명 (cap_YYYYMMDD_HHMMSS.jpg)
        """
        date_time = timestamp[:14]  # YYYYMMDD_HHMMSS
        return f"cap_{date_time}.jpg"
    
    @classmethod
    def create_detection_filename(cls, domain: str, timestamp: str, 
                                sequence: int, confidence: float,
                                original_filename: str = "") -> str:
        """
        검출 파일명 생성 (도메인별) - 최소화된 원본 추적
        
        Args:
            domain: 도메인명
            timestamp: 전체 타임스탬프
            sequence: 순번
            confidence: 신뢰도
            original_filename: 원본 파일명 (선택사항)
            
        Returns:
            str: 검출 파일명 (fr_face_01_cap500_85.jpg)
        """
        prefix = cls.DOMAIN_PREFIXES.get(domain, 'unknown')
        obj_type = cls.OBJECT_TYPES.get(domain, 'object')
        
        # 원본 파일 식별자 생성 (최소화)
        if original_filename:
            original_stem = Path(original_filename).stem
            if original_stem.startswith("cap_"):
                # cap_20250703_094500 -> cap500 (마지막 3자리)
                original_id = f"cap{original_stem[-3:]}"
            elif original_stem.startswith("image"):
                # image01 -> img01
                original_id = f"img{original_stem[-2:]}"
            else:
                # 기타 파일명은 앞 3자리
                original_id = original_stem[:3]
        else:
            # 타임스탬프에서 마지막 3자리
            original_id = timestamp[-3:]
        
        # 신뢰도 정수화 (0.85 -> 85)
        conf_int = int(confidence * 100)
        
        # 최소화된 파일명: fr_face_01_cap500_85.jpg
        return f"{prefix}_{obj_type}_{sequence:02d}_{original_id}_{conf_int}.jpg"
    
    @classmethod
    def create_grouping_filename(cls, domain: str, group_id: int, 
                               sequence: int, source_detection: str = "") -> str:
        """
        그룹핑 파일명 생성 (detected_faces 연계)
        
        Args:
            domain: 도메인명
            group_id: 그룹 ID
            sequence: 그룹 내 순번
            source_detection: 원본 검출 파일명
            
        Returns:
            str: 그룹핑 파일명 (fr_grp_001_01.jpg)
        """
        prefix = cls.DOMAIN_PREFIXES.get(domain, 'unknown')
        
        # 그룹핑 파일명: fr_grp_001_01.jpg
        return f"{prefix}_grp_{group_id:03d}_{sequence:02d}.jpg"
    
    @classmethod
    def create_group_filename(cls, domain: str, group_id: int, 
                            sequence: int) -> str:
        """
        그룹 파일명 생성 (도메인별)
        
        Args:
            domain: 도메인명
            group_id: 그룹 ID
            sequence: 그룹 내 순번
            
        Returns:
            str: 그룹 파일명 (fr_group_001_face_01.jpg)
        """
        prefix = cls.DOMAIN_PREFIXES.get(domain, 'unknown')
        obj_type = cls.OBJECT_TYPES.get(domain, 'object')
        
        return f"{prefix}_group_{group_id:03d}_{obj_type}_{sequence:02d}.jpg"
    
    @classmethod
    def extract_face_id_from_filename(cls, filename: str) -> str:
        """
        파일명에서 얼굴 ID 추출
        
        Args:
            filename: 파일명
            
        Returns:
            str: 얼굴 ID (YYYYMMDD_HHMMSS_01)
        """
        # fr_face_20250630_110815_01_conf0.80.jpg -> 20250630_110815_01
        if filename.startswith("fr_face_"):
            parts = filename.replace(".jpg", "").split("_")
            if len(parts) >= 4:
                return f"{parts[2]}_{parts[3]}_{parts[4]}"
        return ""
    
    @classmethod
    def get_original_capture_from_face_id(cls, face_id: str) -> str:
        """
        얼굴 ID에서 원본 캡처 파일명 생성
        
        Args:
            face_id: 얼굴 ID (20250630_110815_01)
            
        Returns:
            str: 원본 캡처 파일명 (cap_20250630_110815.jpg)
        """
        date_time = face_id[:14]  # YYYYMMDD_HHMMSS
        return f"cap_{date_time}.jpg"

class UniversalMetadata:
    """범용 메타데이터 구조"""
    
    @staticmethod
    def create_base_metadata(domain: str, timestamp: str, 
                           source_file: str) -> dict:
        """
        기본 메타데이터 생성
        
        Args:
            domain: 도메인명
            timestamp: 타임스탬프
            source_file: 소스 파일 경로
            
        Returns:
            dict: 기본 메타데이터
        """
        return {
            "domain": domain,
            "timestamp": timestamp,
            "source_file": source_file,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
            "version": "1.0"
        }
    
    @staticmethod
    def create_detection_metadata(domain: str, object_id: str,
                                confidence: float, bbox: list,
                                original_capture: str) -> dict:
        """
        검출 메타데이터 생성
        
        Args:
            domain: 도메인명
            object_id: 객체 ID
            confidence: 신뢰도
            bbox: 바운딩 박스
            original_capture: 원본 캡처 파일명
            
        Returns:
            dict: 검출 메타데이터
        """
        base = UniversalMetadata.create_base_metadata(domain, 
                                                     object_id.split('_')[0], 
                                                     original_capture)
        base.update({
            "object_id": object_id,
            "detection_confidence": confidence,
            "bbox": bbox,
            "object_type": UniversalNamingSystem.OBJECT_TYPES.get(domain, 'object')
        })
        return base
    
    @staticmethod
    def create_group_metadata(group_id: int, faces: list,
                            similarity_threshold: float) -> dict:
        """
        그룹 메타데이터 생성
        
        Args:
            group_id: 그룹 ID
            faces: 얼굴 정보 리스트
            similarity_threshold: 유사도 임계값
            
        Returns:
            dict: 그룹 메타데이터
        """
        return {
            "group_id": group_id,
            "face_count": len(faces),
            "similarity_threshold": similarity_threshold,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
            "faces": faces
        }

class DomainSpecificNaming:
    """도메인별 특화 네이밍"""
    
    @staticmethod
    def face_recognition_naming(face_id: str, confidence: float) -> str:
        """얼굴인식 특화 네이밍"""
        return f"fr_face_{face_id}_conf{confidence:.2f}.jpg"
    
    @staticmethod
    def factory_defect_naming(defect_type: str, severity: str, 
                            confidence: float) -> str:
        """공장 불량 검출 특화 네이밍"""
        return f"fd_{defect_type}_{severity}_conf{confidence:.2f}.jpg"
    
    @staticmethod
    def powerline_inspection_naming(component: str, issue_type: str,
                                  confidence: float) -> str:
        """전선 검사 특화 네이밍"""
        return f"pi_{component}_{issue_type}_conf{confidence:.2f}.jpg" 