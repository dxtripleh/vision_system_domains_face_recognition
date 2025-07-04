#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파일명 검증 스크립트 (validate_filenames.py)

파일명 패턴 규칙 준수 여부를 검증합니다.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 파일명 패턴 정의
FILENAME_PATTERNS = {
    "1_raw": {
        "pattern": r"^\d{8}_\d{6}_\d{3}\.[a-zA-Z]+$",
        "description": "YYYYMMDD_HHMMSS_XXX.ext",
        "example": "20250703_143022_001.jpg"
    },
    "2_extracted": {
        "pattern": r"^\d{8}_\d{6}_\d{3}_f\d{2}\.[a-zA-Z]+$",
        "description": "YYYYMMDD_HHMMSS_XXX_fXX.ext",
        "example": "20250703_143022_001_f01.jpg"
    },
    "3_clustered": {
        "pattern": r"^g\d{3}_\d{2}\.[a-zA-Z]+$",
        "description": "gXXX_XX.ext",
        "example": "g001_05.jpg"
    },
    "4_labeled": {
        "pattern": r"^g\d{3}_\d{2}_[a-zA-Z0-9_]+\.(jpg|jpeg|png|bmp)$",
        "description": "gXXX_XX_label.ext",
        "example": "g001_05_person_001.jpg"
    },
    "5_embeddings": {
        "pattern": r"^g\d{3}_\d{2}_[a-zA-Z0-9_]+_emb\.(npy|pkl|json)$",
        "description": "gXXX_XX_label_emb.ext",
        "example": "g001_05_person_001_emb.npy"
    }
}

def validate_filename_pattern(filename: str, stage: str) -> Tuple[bool, str]:
    """파일명 패턴을 검증합니다."""
    if stage not in FILENAME_PATTERNS:
        return False, f"알 수 없는 단계: {stage}"
    
    pattern_info = FILENAME_PATTERNS[stage]
    pattern = pattern_info["pattern"]
    
    if re.match(pattern, filename):
        return True, "패턴 준수"
    else:
        return False, f"패턴 불일치 (예상: {pattern_info['example']})"

def scan_stage_files(domain_path: Path, stage: str) -> List[Path]:
    """특정 단계의 파일들을 스캔합니다."""
    stage_path = domain_path / stage
    files = []
    
    if not stage_path.exists():
        return files
    
    # 단계별 하위 폴더 스캔
    subfolders = {
        "1_raw": ["uploads", "captures", "imports"],
        "2_extracted": ["features", "metadata"],
        "3_clustered": ["groups", "metadata"],
        "4_labeled": ["groups", "unknown"],
        "5_embeddings": ["vectors", "index"]
    }
    
    target_subfolders = subfolders.get(stage, [])
    
    for subfolder in target_subfolders:
        subfolder_path = stage_path / subfolder
        if subfolder_path.exists():
            for file_path in subfolder_path.glob("*"):
                if file_path.is_file() and not file_path.name.endswith((".json", ".md")):
                    files.append(file_path)
    
    return files

def validate_stage_filenames(domain_path: Path, stage: str) -> Dict[str, List[Dict]]:
    """특정 단계의 모든 파일명을 검증합니다."""
    files = scan_stage_files(domain_path, stage)
    
    results = {
        "valid": [],
        "invalid": []
    }
    
    for file_path in files:
        filename = file_path.name
        is_valid, message = validate_filename_pattern(filename, stage)
        
        file_info = {
            "file_path": str(file_path),
            "filename": filename,
            "message": message
        }
        
        if is_valid:
            results["valid"].append(file_info)
        else:
            results["invalid"].append(file_info)
    
    return results

def generate_filename_report(domain_path: Path) -> Dict:
    """전체 파일명 검증 리포트를 생성합니다."""
    report = {
        "domain_path": str(domain_path),
        "stages": {},
        "summary": {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "compliance_rate": 0.0
        }
    }
    
    total_files = 0
    total_valid = 0
    total_invalid = 0
    
    for stage in FILENAME_PATTERNS.keys():
        stage_results = validate_stage_filenames(domain_path, stage)
        
        stage_summary = {
            "valid_count": len(stage_results["valid"]),
            "invalid_count": len(stage_results["invalid"]),
            "total_count": len(stage_results["valid"]) + len(stage_results["invalid"]),
            "valid_files": stage_results["valid"],
            "invalid_files": stage_results["invalid"]
        }
        
        if stage_summary["total_count"] > 0:
            stage_summary["compliance_rate"] = stage_summary["valid_count"] / stage_summary["total_count"]
        else:
            stage_summary["compliance_rate"] = 1.0
        
        report["stages"][stage] = stage_summary
        
        total_files += stage_summary["total_count"]
        total_valid += stage_summary["valid_count"]
        total_invalid += stage_summary["invalid_count"]
    
    # 전체 요약 계산
    report["summary"]["total_files"] = total_files
    report["summary"]["valid_files"] = total_valid
    report["summary"]["invalid_files"] = total_invalid
    
    if total_files > 0:
        report["summary"]["compliance_rate"] = total_valid / total_files
    else:
        report["summary"]["compliance_rate"] = 1.0
    
    return report

def print_filename_report(report: Dict):
    """파일명 검증 리포트를 출력합니다."""
    print(f" 파일명 패턴 검증 리포트")
    print(f"도메인 경로: {report['domain_path']}")
    print(f"전체 파일: {report['summary']['total_files']}개")
    print(f"준수 파일: {report['summary']['valid_files']}개")
    print(f"위반 파일: {report['summary']['invalid_files']}개")
    print(f"준수율: {report['summary']['compliance_rate']:.1%}")
    print()
    
    for stage, stage_info in report["stages"].items():
        if stage_info["total_count"] == 0:
            continue
        
        print(f" {stage}:")
        print(f"  - 총 파일: {stage_info['total_count']}개")
        print(f"  - 준수: {stage_info['valid_count']}개")
        print(f"  - 위반: {stage_info['invalid_count']}개")
        print(f"  - 준수율: {stage_info['compliance_rate']:.1%}")
        
        if stage_info["invalid_files"]:
            print(f"  - 위반 파일들:")
            for invalid_file in stage_info["invalid_files"][:3]:  # 처음 3개만 표시
                print(f"     {invalid_file['filename']}: {invalid_file['message']}")
            if len(stage_info["invalid_files"]) > 3:
                print(f"    ... 외 {len(stage_info['invalid_files']) - 3}개")
        print()

def suggest_filename_corrections(domain_path: Path) -> Dict[str, List[Dict]]:
    """파일명 수정 제안을 생성합니다."""
    suggestions = {}
    
    for stage in FILENAME_PATTERNS.keys():
        stage_results = validate_stage_filenames(domain_path, stage)
        stage_suggestions = []
        
        for invalid_file in stage_results["invalid"]:
            filename = invalid_file["filename"]
            suggestion = suggest_correct_filename(filename, stage)
            
            if suggestion:
                stage_suggestions.append({
                    "current_filename": filename,
                    "suggested_filename": suggestion,
                    "file_path": invalid_file["file_path"]
                })
        
        if stage_suggestions:
            suggestions[stage] = stage_suggestions
    
    return suggestions

def suggest_correct_filename(filename: str, stage: str) -> str:
    """파일명 수정 제안을 생성합니다."""
    if stage not in FILENAME_PATTERNS:
        return None
    
    pattern_info = FILENAME_PATTERNS[stage]
    
    # 기본 확장자 추출
    if "." in filename:
        name_part, ext_part = filename.rsplit(".", 1)
        ext = ext_part.lower()
    else:
        name_part = filename
        ext = "jpg"  # 기본값
    
    # 단계별 수정 제안
    if stage == "1_raw":
        # YYYYMMDD_HHMMSS_XXX.ext 패턴으로 수정
        if re.match(r"^\d{8}_\d{6}", name_part):
            # 이미 날짜/시간 형식이 있으면 그대로 사용
            return f"{name_part}_001.{ext}"
        else:
            # 현재 시간으로 새로 생성
            from datetime import datetime
            now = datetime.now()
            return f"{now.strftime('%Y%m%d_%H%M%S')}_001.{ext}"
    
    elif stage == "2_extracted":
        # YYYYMMDD_HHMMSS_XXX_fXX.ext 패턴으로 수정
        if re.match(r"^\d{8}_\d{6}_\d{3}", name_part):
            return f"{name_part}_f01.{ext}"
        else:
            return None
    
    elif stage == "3_clustered":
        # gXXX_XX.ext 패턴으로 수정
        if name_part.startswith("g") and re.match(r"^g\d+", name_part):
            # 이미 g로 시작하면 그대로 사용
            return f"{name_part}_01.{ext}"
        else:
            return f"g001_01.{ext}"
    
    elif stage == "4_labeled":
        # gXXX_XX_label.ext 패턴으로 수정
        if name_part.startswith("g") and re.match(r"^g\d+_\d+", name_part):
            return f"{name_part}_unknown.{ext}"
        else:
            return f"g001_01_unknown.{ext}"
    
    elif stage == "5_embeddings":
        # gXXX_XX_label_emb.ext 패턴으로 수정
        if name_part.startswith("g") and re.match(r"^g\d+_\d+", name_part):
            if name_part.endswith("_emb"):
                return f"{name_part}.npy"
            else:
                return f"{name_part}_emb.npy"
        else:
            return f"g001_01_unknown_emb.npy"
    
    return None

def print_suggestions(suggestions: Dict[str, List[Dict]]):
    """수정 제안을 출력합니다."""
    if not suggestions:
        print(" 모든 파일명이 패턴을 준수합니다.")
        return
    
    print(" 파일명 수정 제안:")
    print()
    
    for stage, stage_suggestions in suggestions.items():
        if stage_suggestions:
            print(f" {stage}:")
            for suggestion in stage_suggestions[:5]:  # 처음 5개만 표시
                print(f"  {suggestion['current_filename']}  {suggestion['suggested_filename']}")
            if len(stage_suggestions) > 5:
                print(f"  ... 외 {len(stage_suggestions) - 5}개")
            print()

def validate_filenames(domain: str, feature: str, base_path: str = "data/domains", 
                      show_suggestions: bool = False, output_file: str = None):
    """파일명 검증을 실행합니다."""
    domain_path = Path(base_path) / domain / feature
    
    if not domain_path.exists():
        print(f" 도메인이 존재하지 않습니다: {domain_path}")
        return False
    
    # 검증 리포트 생성
    report = generate_filename_report(domain_path)
    
    # 리포트 출력
    print_filename_report(report)
    
    # 수정 제안 출력 (요청된 경우)
    if show_suggestions and report["summary"]["invalid_files"] > 0:
        suggestions = suggest_filename_corrections(domain_path)
        print_suggestions(suggestions)
    
    # 파일로 저장 (요청된 경우)
    if output_file:
        import json
        output_path = Path(output_file)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f" 검증 리포트가 {output_file}에 저장되었습니다.")
    
    return report["summary"]["compliance_rate"] == 1.0

def main():
    parser = argparse.ArgumentParser(description="파일명 패턴 검증 스크립트")
    parser.add_argument("domain", help="도메인명 (humanoid, factory, powerline_inspection)")
    parser.add_argument("feature", help="기능명 (face_recognition, defect_detection, inspection)")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로 (기본값: data/domains)")
    parser.add_argument("--suggestions", action="store_true", help="수정 제안 표시")
    parser.add_argument("--output", help="검증 리포트 출력 파일")
    
    args = parser.parse_args()
    
    try:
        success = validate_filenames(args.domain, args.feature, args.base_path, 
                                   args.suggestions, args.output)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f" 검증 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
