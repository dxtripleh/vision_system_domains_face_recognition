#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파이프라인 자동 실행 스크립트 (run_pipeline.py)

9단계 파이프라인을 순차적으로 실행합니다.
간략화된 파일명 패턴을 사용합니다.
"""

import os
import sys
import argparse
import json
import time
import shutil
import platform
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

def get_optimal_config():
    """하드웨어 환경에 따른 최적 설정 자동 선택"""
    system = platform.system().lower()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3) if gpu_available else 0
    except:
        gpu_available = False
        gpu_memory = 0
    
    if gpu_available and gpu_memory >= 16:
        return {"device": "cuda", "batch_size": 16, "model_size": "large", "precision": "fp16"}
    elif gpu_available and gpu_memory >= 4:
        return {"device": "cuda", "batch_size": 4, "model_size": "medium", "precision": "fp16"}
    else:
        return {"device": "cpu", "batch_size": 1, "model_size": "small", "precision": "fp32"}

def is_jetson():
    """Jetson 환경 감지"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "jetson" in f.read().lower()
    except:
        return False

def create_platform_camera(camera_id=0, config=None):
    """플랫폼별 카메라 생성"""
    import cv2
    system = platform.system().lower()
    
    if system == "windows":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    elif system == "linux":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if is_jetson():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(camera_id)
    
    return cap

def verify_hardware_connection():
    """하드웨어 연결 상태를 확인합니다."""
    # 시뮬레이션 방지
    if os.environ.get("USE_SIMULATION", "False").lower() == "true":
        raise RuntimeError("시뮬레이션 모드는 금지되어 있습니다. 실제 하드웨어를 연결하세요.")
    
    # 카메라 연결 확인
    try:
        cap = create_platform_camera(0)
        if not cap.isOpened():
            raise RuntimeError("카메라 연결 실패. 하드웨어 연결을 확인하세요.")
        cap.release()
        return True
    except Exception as e:
        raise RuntimeError(f"하드웨어 연결 확인 중 오류 발생: {str(e)}")

def validate_domain_exists(domain: str, feature: str, base_path: str = "data/domains"):
    """도메인 구조가 존재하는지 확인합니다."""
    domain_path = Path(base_path) / domain / feature
    
    if not domain_path.exists():
        raise FileNotFoundError(f"도메인 {domain}/{feature}가 존재하지 않습니다. 먼저 create_domain.py를 실행하세요.")
    
    required_stages = ["1_raw", "2_extracted", "3_clustered", "4_labeled", "5_embeddings"]
    for stage in required_stages:
        stage_path = domain_path / stage
        if not stage_path.exists():
            raise FileNotFoundError(f"필수 단계 폴더가 없습니다: {stage_path}")
    
    return domain_path

def load_progress(domain_path: Path):
    """파이프라인 진행 상황을 로드합니다."""
    progress_file = domain_path / "pipeline_progress.json"
    
    if progress_file.exists():
        return json.loads(progress_file.read_text(encoding="utf-8"))
    else:
        return {
            "1_raw": {"status": "ready", "files_count": 0},
            "2_extracted": {"status": "pending", "files_count": 0},
            "3_clustered": {"status": "pending", "files_count": 0},
            "4_labeled": {"status": "pending", "files_count": 0},
            "5_embeddings": {"status": "pending", "files_count": 0}
        }

def save_progress(domain_path: Path, progress: dict):
    """파이프라인 진행 상황을 저장합니다."""
    progress_file = domain_path / "pipeline_progress.json"
    progress["metadata"] = {
        "last_updated": datetime.now().isoformat()
    }
    progress_file.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

def count_files_in_stage(domain_path: Path, stage: str):
    """특정 단계의 파일 수를 계산합니다."""
    stage_path = domain_path / stage
    file_count = 0
    
    # 메타데이터 폴더 제외
    exclude_folders = {"metadata", "traceability"}
    
    for subfolder in stage_path.iterdir():
        if (subfolder.is_dir() and 
            subfolder.name not in exclude_folders):
            for file_path in subfolder.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith((".json", ".md")):
                    file_count += 1
    
    return file_count

def run_stage_1(domain_path: Path, progress: dict):
    """1단계: Raw Data Collection 상태 확인"""
    print(" 1단계: Raw Data Collection 확인 중...")
    
    file_count = count_files_in_stage(domain_path, "1_raw")
    
    if file_count == 0:
        print("  1단계에 파일이 없습니다. 다음 위치에 파일을 추가하세요:")
        print(f"   - {domain_path / '1_raw' / 'uploads'}")
        print(f"   - {domain_path / '1_raw' / 'captures'}")
        print(f"   - {domain_path / '1_raw' / 'imports'}")
        return False
    
    progress["1_raw"]["status"] = "completed"
    progress["1_raw"]["files_count"] = file_count
    progress["1_raw"]["completed_at"] = datetime.now().isoformat()
    
    print(f" 1단계 완료: {file_count}개 파일 발견")
    return True

def run_stage_2(domain_path: Path, progress: dict):
    """2단계: Feature Extraction 실행"""
    print(" 2단계: Feature Extraction 실행 중...")
    
    # 1단계 파일들 확인
    raw_files = []
    raw_path = domain_path / "1_raw"
    
    for subfolder in ["uploads", "captures", "imports"]:
        subfolder_path = raw_path / subfolder
        if subfolder_path.exists():
            for file_path in subfolder_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    raw_files.append(file_path)
    
    if not raw_files:
        print(" 1단계에 처리할 이미지 파일이 없습니다.")
        return False
    
    # 특이점 추출 시뮬레이션
    extracted_path = domain_path / "2_extracted" / "features"
    extracted_path.mkdir(exist_ok=True)
    
    trace_data = {}
    extracted_count = 0
    
    for raw_file in raw_files:
        raw_id = raw_file.stem
        
        # 시뮬레이션: 각 이미지에서 1-3개의 특이점 추출
        num_features = min(3, max(1, len(raw_id) % 3 + 1))
        extracted_files = []
        
        for i in range(num_features):
            feature_filename = f"{raw_id}_f{i+1:02d}{raw_file.suffix}"
            feature_path = extracted_path / feature_filename
            
            try:
                shutil.copy2(raw_file, feature_path)
                extracted_files.append(feature_filename)
                extracted_count += 1
            except Exception as e:
                print(f"  파일 복사 오류: {e}")
        
        trace_data[raw_file.name] = extracted_files
    
    # 추적성 정보 업데이트
    update_traceability(domain_path, "stage_1_to_2", trace_data)
    
    progress["2_extracted"]["status"] = "completed"
    progress["2_extracted"]["files_count"] = extracted_count
    progress["2_extracted"]["completed_at"] = datetime.now().isoformat()
    
    print(f" 2단계 완료: {len(raw_files)}개 원본에서 {extracted_count}개 특이점 추출")
    return True

def run_stage_3(domain_path: Path, progress: dict):
    """3단계: Similarity Clustering 실행"""
    print(" 3단계: Similarity Clustering 실행 중...")
    
    features_path = domain_path / "2_extracted" / "features"
    feature_files = list(features_path.glob("*"))
    
    if not feature_files:
        print(" 2단계에 특이점 파일이 없습니다.")
        return False
    
    # 클러스터링 시뮬레이션
    clustered_path = domain_path / "3_clustered" / "groups"
    clustered_path.mkdir(exist_ok=True)
    
    groups = {}
    trace_data = {}
    
    # 시뮬레이션: 5개씩 그룹으로 묶기
    group_size = 5
    
    for i, feature_file in enumerate(feature_files):
        group_id = f"g{((i // group_size) + 1):03d}"
        
        if group_id not in groups:
            groups[group_id] = []
        
        groups[group_id].append(feature_file)
        trace_data[feature_file.name] = f"{group_id}_{len(groups[group_id]):02d}.jpg"
    
    # 그룹 파일 생성
    for group_id, files in groups.items():
        group_filename = f"{group_id}_{len(files):02d}.jpg"
        group_path = clustered_path / group_filename
        
        try:
            shutil.copy2(files[0], group_path)
        except Exception as e:
            print(f"  그룹 파일 생성 오류: {e}")
    
    # 추적성 정보 업데이트
    update_traceability(domain_path, "stage_2_to_3", trace_data)
    
    progress["3_clustered"]["status"] = "completed" 
    progress["3_clustered"]["files_count"] = len(groups)
    progress["3_clustered"]["completed_at"] = datetime.now().isoformat()
    
    print(f" 3단계 완료: {len(feature_files)}개 특이점을 {len(groups)}개 그룹으로 클러스터링")
    return True

def update_traceability(domain_path: Path, stage_key: str, mapping_data: dict):
    """추적성 정보를 업데이트합니다."""
    trace_file = domain_path / "traceability" / "trace.json"
    
    if trace_file.exists():
        trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    else:
        trace_data = {
            "stage_1_to_2": {},
            "stage_2_to_3": {},
            "stage_3_to_4": {},
            "stage_4_to_5": {}
        }
    
    trace_data[stage_key].update(mapping_data)
    trace_data["metadata"] = {
        "last_updated": datetime.now().isoformat()
    }
    
    trace_file.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8")

def run_pipeline(domain: str, feature: str, stage: int = None, base_path: str = "data/domains"):
    """파이프라인을 실행합니다."""
    
    print(f" 파이프라인 실행: {domain}/{feature}")
    
    # 도메인 검증
    domain_path = validate_domain_exists(domain, feature, base_path)
    
    # 진행 상황 로드
    progress = load_progress(domain_path)
    
    # 실행할 단계 결정
    stages_to_run = []
    if stage:
        stages_to_run = [stage]
    else:
        # 완료되지 않은 단계들 찾기
        for i in range(1, 6):
            stage_key = f"{i}_{'raw' if i == 1 else 'extracted' if i == 2 else 'clustered' if i == 3 else 'labeled' if i == 4 else 'embeddings'}"
            if progress.get(stage_key, {}).get("status") != "completed":
                stages_to_run.append(i)
    
    # 단계별 실행
    success = True
    for stage_num in stages_to_run:
        if stage_num == 1:
            success = run_stage_1(domain_path, progress)
        elif stage_num == 2:
            success = run_stage_2(domain_path, progress)
        elif stage_num == 3:
            success = run_stage_3(domain_path, progress)
        elif stage_num == 4:
            print(" 4단계 (라벨링)는 사용자 개입이 필요합니다.")
            break
        elif stage_num == 5:
            print(" 5단계 (임베딩)는 4단계 완료 후 실행 가능합니다.")
            break
        
        if not success:
            break
        
        # 진행 상황 저장
        save_progress(domain_path, progress)
        time.sleep(0.5)
    
    if success:
        print(f" 파이프라인 실행 완료!")
    else:
        print(f"  파이프라인 실행 중 문제가 발생했습니다.")

def main():
    parser = argparse.ArgumentParser(description="파이프라인 자동 실행 스크립트")
    parser.add_argument("domain", help="도메인명 (humanoid, factory, powerline_inspection)")
    parser.add_argument("feature", help="기능명 (face_recognition, defect_detection, inspection)")
    parser.add_argument("--stage", type=int, choices=[1,2,3,4,5], help="특정 단계만 실행 (1-5)")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로 (기본값: data/domains)")
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.domain, args.feature, args.stage, args.base_path)
    except Exception as e:
        print(f" 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
