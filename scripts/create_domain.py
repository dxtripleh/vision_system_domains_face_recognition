#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
도메인 자동 생성 스크립트 (create_domain.py)

새로운 도메인 생성 시 9단계 구조를 자동으로 생성합니다.
간략화된 파일명 패턴을 적용합니다.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

def create_domain_structure(domain: str, feature: str, base_path: str = "data/domains"):
    """
    새로운 도메인의 9단계 구조를 자동으로 생성합니다.
    
    Args:
        domain: 도메인명 (humanoid, factory, powerline_inspection)
        feature: 기능명 (face_recognition, defect_detection, inspection)
        base_path: 기본 경로
    """
    
    # 도메인 경로 생성
    domain_path = Path(base_path) / domain / feature
    
    print(f" 도메인 구조 생성 중: {domain}/{feature}")
    
    # 9단계 폴더 구조 정의 (간략화된 버전)
    folder_structure = {
        "1_raw": {
            "uploads": "사용자 직접 업로드",
            "captures": "하드웨어 자동 캡처", 
            "imports": "외부 시스템 연동"
        },
        "2_extracted": {
            "features": "검출된 특이점",
            "metadata": "검출 메타데이터"
        },
        "3_clustered": {
            "groups": "자동 그룹핑 결과",
            "metadata": "클러스터링 메타데이터"
        },
        "4_labeled": {
            "groups": "라벨링된 그룹들",
            "unknown": "미분류 데이터"
        },
        "5_embeddings": {
            "vectors": "임베딩 벡터",
            "index": "검색 인덱스"
        },
        "cache": {},
        "models": {},
        "traceability": {}
    }
    
    # 폴더 생성
    for stage, subfolders in folder_structure.items():
        stage_path = domain_path / stage
        stage_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(subfolders, dict):
            for subfolder in subfolders.keys():
                (stage_path / subfolder).mkdir(exist_ok=True)
        
        print(f"   {stage_path}")
    
    # README.md 파일들 생성
    create_readme_files(domain_path, domain, feature)
    
    # 추적성 파일 초기화
    create_traceability_files(domain_path)
    
    # 파이프라인 진행 상황 파일 초기화
    create_progress_file(domain_path)
    
    print(f" 도메인 {domain}/{feature} 구조 생성 완료!")

def create_readme_files(domain_path: Path, domain: str, feature: str):
    """각 단계별 README.md 파일을 생성합니다."""
    
    readme_contents = {
        "1_raw": f"""# 1단계: Raw Data Collection ({domain}/{feature})

## 파일명 패턴
- 패턴: `{{timestamp}}_{{idx}}.{{ext}}`
- 예시: `20250728_143022_001.jpg`

## 폴더 구조
- `uploads/`: 사용자가 직접 업로드한 파일
- `captures/`: 카메라/센서에서 자동 캡처한 파일
- `imports/`: 외부 시스템에서 가져온 파일

## 사용법
```bash
# 파일 업로드 후 다음 단계 실행
python scripts/run_pipeline.py {domain} {feature} --stage 2
```
""",
        "2_extracted": f"""# 2단계: Feature Extraction ({domain}/{feature})

## 파일명 패턴
- 패턴: `{{raw_id}}_f{{feature_idx}}.{{ext}}`
- 예시: `20250728_143022_001_f01.jpg`

## 폴더 구조
- `features/`: 검출된 특이점 파일들
- `metadata/`: 검출 과정의 메타데이터

## 추적성
- 원본 파일과 추출된 특이점 간의 매핑이 `traceability/trace.json`에 저장됩니다.
""",
        "3_clustered": f"""# 3단계: Similarity Clustering ({domain}/{feature})

## 파일명 패턴
- 패턴: `g{{group_id}}_{{count}}.{{ext}}`
- 예시: `g001_05.jpg`

## 폴더 구조
- `groups/`: 자동 그룹핑된 결과
- `metadata/`: 클러스터링 메타데이터

## 추적성
- 특이점과 그룹 간의 매핑이 `traceability/trace.json`에 저장됩니다.
"""
    }
    
    for stage, content in readme_contents.items():
        readme_path = domain_path / stage / "README.md"
        readme_path.write_text(content, encoding="utf-8")

def create_traceability_files(domain_path: Path):
    """추적성 파일들을 초기화합니다."""
    
    # 메인 추적성 파일
    trace_file = domain_path / "traceability" / "trace.json"
    initial_trace = {
        "stage_1_to_2": {},
        "stage_2_to_3": {},
        "stage_3_to_4": {},
        "stage_4_to_5": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    }
    
    trace_file.write_text(json.dumps(initial_trace, indent=2, ensure_ascii=False), encoding="utf-8")

def create_progress_file(domain_path: Path):
    """파이프라인 진행 상황 파일을 초기화합니다."""
    
    progress_file = domain_path / "pipeline_progress.json"
    initial_progress = {
        "1_raw": {"status": "ready", "files_count": 0},
        "2_extracted": {"status": "pending", "files_count": 0},
        "3_clustered": {"status": "pending", "files_count": 0},
        "4_labeled": {"status": "pending", "files_count": 0},
        "5_embeddings": {"status": "pending", "files_count": 0},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    }
    
    progress_file.write_text(json.dumps(initial_progress, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="도메인 자동 생성 스크립트")
    parser.add_argument("domain", help="도메인명 (humanoid, factory, powerline_inspection)")
    parser.add_argument("feature", help="기능명 (face_recognition, defect_detection, inspection)")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로 (기본값: data/domains)")
    
    args = parser.parse_args()
    
    try:
        create_domain_structure(args.domain, args.feature, args.base_path)
    except Exception as e:
        print(f" 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
