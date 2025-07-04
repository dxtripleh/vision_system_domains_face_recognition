#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파이프라인 검증 스크립트 (validate_pipeline.py)

9단계 파이프라인 구조와 데이터 무결성을 검증합니다.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def validate_folder_structure(domain_path: Path) -> Dict[str, bool]:
    """폴더 구조 검증"""
    required_stages = [
        "1_raw",
        "2_extracted", 
        "3_clustered",
        "4_labeled",
        "5_embeddings"
    ]
    
    required_subfolders = {
        "1_raw": ["uploads", "captures", "imports"],
        "2_extracted": ["features", "metadata"],
        "3_clustered": ["groups", "metadata"],
        "4_labeled": ["groups", "unknown"],
        "5_embeddings": ["vectors", "index"]
    }
    
    results = {}
    
    for stage in required_stages:
        stage_path = domain_path / stage
        stage_valid = stage_path.exists()
        
        if stage_valid:
            # 서브폴더 검증
            subfolder_results = {}
            for subfolder in required_subfolders.get(stage, []):
                subfolder_path = stage_path / subfolder
                subfolder_results[subfolder] = subfolder_path.exists()
            
            results[stage] = all(subfolder_results.values())
            results[f"{stage}_subfolders"] = subfolder_results
        else:
            results[stage] = False
    
    return results

def validate_file_naming_patterns(domain_path: Path) -> Dict[str, List[str]]:
    """파일명 패턴 검증"""
    violations = {}
    
    # 1단계: 원본 데이터 파일명 패턴
    raw_path = domain_path / "1_raw"
    if raw_path.exists():
        raw_violations = []
        for subfolder in ["uploads", "captures", "imports"]:
            subfolder_path = raw_path / subfolder
            if subfolder_path.exists():
                for file_path in subfolder_path.glob("*"):
                    if file_path.is_file():
                        filename = file_path.name
                        # 패턴: YYYYMMDD_HHMMSS_XXX.ext
                        if not (len(filename) >= 15 and 
                               filename[:8].isdigit() and 
                               filename[8] == "_" and
                               filename[9:15].isdigit() and
                               filename[15] == "_"):
                            raw_violations.append(str(file_path))
        
        if raw_violations:
            violations["1_raw"] = raw_violations
    
    # 2단계: 특이점 파일명 패턴
    extracted_path = domain_path / "2_extracted" / "features"
    if extracted_path.exists():
        extracted_violations = []
        for file_path in extracted_path.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                # 패턴: YYYYMMDD_HHMMSS_XXX_fXX.ext
                if not (len(filename) >= 18 and 
                       filename[:8].isdigit() and 
                       filename[8] == "_" and
                       filename[9:15].isdigit() and
                       filename[15] == "_" and
                       "_f" in filename):
                    extracted_violations.append(str(file_path))
        
        if extracted_violations:
            violations["2_extracted"] = extracted_violations
    
    # 3단계: 클러스터 파일명 패턴
    clustered_path = domain_path / "3_clustered" / "groups"
    if clustered_path.exists():
        clustered_violations = []
        for file_path in clustered_path.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                # 패턴: gXXX_XX.ext
                if not (filename.startswith("g") and 
                       filename[1:4].isdigit() and
                       filename[4] == "_" and
                       filename[5:7].isdigit()):
                    clustered_violations.append(str(file_path))
        
        if clustered_violations:
            violations["3_clustered"] = clustered_violations
    
    return violations

def validate_traceability(domain_path: Path) -> Dict[str, bool]:
    """추적성 정보 검증"""
    trace_file = domain_path / "traceability" / "trace.json"
    
    if not trace_file.exists():
        return {"trace_file_exists": False}
    
    try:
        trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
        
        required_keys = ["stage_1_to_2", "stage_2_to_3", "stage_3_to_4", "stage_4_to_5"]
        results = {}
        
        for key in required_keys:
            results[f"{key}_exists"] = key in trace_data
            if key in trace_data:
                results[f"{key}_has_data"] = len(trace_data[key]) > 0
        
        results["metadata_exists"] = "metadata" in trace_data
        results["last_updated_exists"] = "metadata" in trace_data and "last_updated" in trace_data["metadata"]
        
        return results
        
    except Exception as e:
        return {"trace_file_valid": False, "error": str(e)}

def validate_progress_tracking(domain_path: Path) -> Dict[str, bool]:
    """진행 상황 추적 검증"""
    progress_file = domain_path / "pipeline_progress.json"
    
    if not progress_file.exists():
        return {"progress_file_exists": False}
    
    try:
        progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
        
        required_stages = ["1_raw", "2_extracted", "3_clustered", "4_labeled", "5_embeddings"]
        results = {}
        
        for stage in required_stages:
            results[f"{stage}_exists"] = stage in progress_data
            if stage in progress_data:
                stage_data = progress_data[stage]
                results[f"{stage}_has_status"] = "status" in stage_data
                results[f"{stage}_has_files_count"] = "files_count" in stage_data
        
        results["metadata_exists"] = "metadata" in progress_data
        results["last_updated_exists"] = "metadata" in progress_data and "last_updated" in progress_data["metadata"]
        
        return results
        
    except Exception as e:
        return {"progress_file_valid": False, "error": str(e)}

def validate_data_integrity(domain_path: Path) -> Dict[str, bool]:
    """데이터 무결성 검증"""
    results = {}
    
    # 1단계  2단계 무결성
    raw_files = []
    raw_path = domain_path / "1_raw"
    if raw_path.exists():
        for subfolder in ["uploads", "captures", "imports"]:
            subfolder_path = raw_path / subfolder
            if subfolder_path.exists():
                for file_path in subfolder_path.glob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        raw_files.append(file_path.name)
    
    extracted_files = []
    extracted_path = domain_path / "2_extracted" / "features"
    if extracted_path.exists():
        for file_path in extracted_path.glob("*"):
            if file_path.is_file():
                extracted_files.append(file_path.name)
    
    # 추적성 파일에서 매핑 확인
    trace_file = domain_path / "traceability" / "trace.json"
    if trace_file.exists():
        try:
            trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
            stage_1_to_2 = trace_data.get("stage_1_to_2", {})
            
            # 모든 원본 파일이 추적성에 기록되어 있는지 확인
            all_traced_raw = set(stage_1_to_2.keys())
            all_actual_raw = set(raw_files)
            
            results["all_raw_files_traced"] = all_actual_raw.issubset(all_traced_raw)
            results["no_extra_traced_raw"] = all_traced_raw.issubset(all_actual_raw)
            
            # 모든 추출 파일이 추적성에 기록되어 있는지 확인
            all_traced_extracted = set()
            for extracted_list in stage_1_to_2.values():
                if isinstance(extracted_list, list):
                    all_traced_extracted.update(extracted_list)
            
            all_actual_extracted = set(extracted_files)
            results["all_extracted_files_traced"] = all_actual_extracted.issubset(all_traced_extracted)
            
        except Exception as e:
            results["traceability_check_error"] = str(e)
    
    return results

def generate_validation_report(domain: str, feature: str, base_path: str = "data/domains") -> Dict:
    """검증 리포트 생성"""
    domain_path = Path(base_path) / domain / feature
    
    if not domain_path.exists():
        return {
            "domain_exists": False,
            "error": f"도메인 {domain}/{feature}가 존재하지 않습니다."
        }
    
    report = {
        "domain": domain,
        "feature": feature,
        "validation_time": datetime.now().isoformat(),
        "domain_exists": True,
        "folder_structure": validate_folder_structure(domain_path),
        "file_naming": validate_file_naming_patterns(domain_path),
        "traceability": validate_traceability(domain_path),
        "progress_tracking": validate_progress_tracking(domain_path),
        "data_integrity": validate_data_integrity(domain_path)
    }
    
    # 전체 통과 여부 계산
    all_checks = []
    
    # 폴더 구조 검증
    for stage, valid in report["folder_structure"].items():
        if not stage.endswith("_subfolders"):
            all_checks.append(valid)
    
    # 파일명 패턴 검증
    all_checks.append(len(report["file_naming"]) == 0)  # 위반이 없어야 함
    
    # 추적성 검증
    trace_results = report["traceability"]
    if "trace_file_exists" in trace_results:
        all_checks.append(trace_results["trace_file_exists"])
    
    # 진행 상황 추적 검증
    progress_results = report["progress_tracking"]
    if "progress_file_exists" in progress_results:
        all_checks.append(progress_results["progress_file_exists"])
    
    # 데이터 무결성 검증
    integrity_results = report["data_integrity"]
    if "all_raw_files_traced" in integrity_results:
        all_checks.append(integrity_results["all_raw_files_traced"])
    
    report["overall_passed"] = all(all_checks) if all_checks else False
    
    return report

def print_validation_report(report: Dict):
    """검증 리포트 출력"""
    print(f" 파이프라인 검증 리포트")
    print(f"도메인: {report['domain']}/{report['feature']}")
    print(f"검증 시간: {report['validation_time']}")
    print(f"전체 통과: {'' if report['overall_passed'] else ''}")
    print()
    
    if not report.get("domain_exists", True):
        print(f" {report.get('error', '도메인이 존재하지 않습니다.')}")
        return
    
    # 폴더 구조 검증 결과
    print(" 폴더 구조 검증:")
    for stage, valid in report["folder_structure"].items():
        if not stage.endswith("_subfolders"):
            status = "" if valid else ""
            print(f"  {status} {stage}")
    
    # 파일명 패턴 검증 결과
    print("\n 파일명 패턴 검증:")
    if report["file_naming"]:
        for stage, violations in report["file_naming"].items():
            print(f"   {stage}: {len(violations)}개 위반")
            for violation in violations[:3]:  # 처음 3개만 표시
                print(f"    - {violation}")
            if len(violations) > 3:
                print(f"    ... 외 {len(violations) - 3}개")
    else:
        print("   모든 파일명 패턴 준수")
    
    # 추적성 검증 결과
    print("\n 추적성 검증:")
    trace_results = report["traceability"]
    if "trace_file_exists" in trace_results:
        status = "" if trace_results["trace_file_exists"] else ""
        print(f"  {status} 추적성 파일 존재")
    
    # 진행 상황 추적 검증 결과
    print("\n 진행 상황 추적 검증:")
    progress_results = report["progress_tracking"]
    if "progress_file_exists" in progress_results:
        status = "" if progress_results["progress_file_exists"] else ""
        print(f"  {status} 진행 상황 파일 존재")
    
    # 데이터 무결성 검증 결과
    print("\n 데이터 무결성 검증:")
    integrity_results = report["data_integrity"]
    if "all_raw_files_traced" in integrity_results:
        status = "" if integrity_results["all_raw_files_traced"] else ""
        print(f"  {status} 모든 원본 파일 추적됨")

def validate_pipeline(domain: str, feature: str, base_path: str = "data/domains", output_file: str = None):
    """파이프라인 검증 실행"""
    report = generate_validation_report(domain, feature, base_path)
    
    print_validation_report(report)
    
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n 검증 리포트가 {output_file}에 저장되었습니다.")
    
    return report["overall_passed"]

def main():
    parser = argparse.ArgumentParser(description="파이프라인 검증 스크립트")
    parser.add_argument("domain", help="도메인명 (humanoid, factory, powerline_inspection)")
    parser.add_argument("feature", help="기능명 (face_recognition, defect_detection, inspection)")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로 (기본값: data/domains)")
    parser.add_argument("--output", help="검증 리포트 출력 파일")
    
    args = parser.parse_args()
    
    try:
        success = validate_pipeline(args.domain, args.feature, args.base_path, args.output)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f" 검증 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
