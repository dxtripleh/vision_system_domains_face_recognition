#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
추적성 관리 스크립트 (manage_traceability.py)

파일 계보 추적 및 추적성 정보를 관리합니다.
"""

import os
import sys
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

def calculate_file_hash(file_path: Path) -> str:
    """파일의 SHA256 해시를 계산합니다."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"  파일 해시 계산 오류 ({file_path}): {e}")
        return ""

def load_traceability_data(domain_path: Path) -> Dict:
    """추적성 데이터를 로드합니다."""
    trace_file = domain_path / "traceability" / "trace.json"
    
    if trace_file.exists():
        try:
            return json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  추적성 파일 로드 오류: {e}")
            return create_empty_trace_data()
    else:
        return create_empty_trace_data()

def create_empty_trace_data() -> Dict:
    """빈 추적성 데이터 구조를 생성합니다."""
    return {
        "stage_1_to_2": {},
        "stage_2_to_3": {},
        "stage_3_to_4": {},
        "stage_4_to_5": {},
        "file_hashes": {},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    }

def save_traceability_data(domain_path: Path, trace_data: Dict):
    """추적성 데이터를 저장합니다."""
    trace_file = domain_path / "traceability" / "trace.json"
    trace_file.parent.mkdir(exist_ok=True)
    
    trace_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    try:
        trace_file.write_text(json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f" 추적성 파일 저장 오류: {e}")

def update_stage_mapping(domain_path: Path, stage_key: str, mapping_data: Dict):
    """단계 간 매핑 정보를 업데이트합니다."""
    trace_data = load_traceability_data(domain_path)
    trace_data[stage_key].update(mapping_data)
    save_traceability_data(domain_path, trace_data)
    print(f" {stage_key} 매핑 정보 업데이트 완료")

def add_file_hash(domain_path: Path, file_path: Path, file_hash: str):
    """파일 해시 정보를 추가합니다."""
    trace_data = load_traceability_data(domain_path)
    
    if "file_hashes" not in trace_data:
        trace_data["file_hashes"] = {}
    
    trace_data["file_hashes"][str(file_path)] = {
        "hash": file_hash,
        "added_at": datetime.now().isoformat()
    }
    
    save_traceability_data(domain_path, trace_data)

def scan_and_update_hashes(domain_path: Path):
    """도메인 내 모든 파일의 해시를 스캔하고 업데이트합니다."""
    print(" 파일 해시 스캔 중...")
    
    trace_data = load_traceability_data(domain_path)
    if "file_hashes" not in trace_data:
        trace_data["file_hashes"] = {}
    
    updated_count = 0
    
    # 모든 단계 폴더 스캔
    for stage in ["1_raw", "2_extracted", "3_clustered", "4_labeled", "5_embeddings"]:
        stage_path = domain_path / stage
        if not stage_path.exists():
            continue
        
        for file_path in stage_path.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith((".json", ".md")):
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    # 크로스플랫폼 호환성을 위해 경로 구분자 통일
                    file_key = str(file_path.relative_to(domain_path)).replace("\\", "/")
                    trace_data["file_hashes"][file_key] = {
                        "hash": file_hash,
                        "stage": stage,
                        "updated_at": datetime.now().isoformat()
                    }
                    updated_count += 1
    
    save_traceability_data(domain_path, trace_data)
    print(f" {updated_count}개 파일 해시 업데이트 완료")

def verify_file_integrity(domain_path: Path) -> Dict[str, List[str]]:
    """파일 무결성을 검증합니다."""
    print(" 파일 무결성 검증 중...")
    
    trace_data = load_traceability_data(domain_path)
    file_hashes = trace_data.get("file_hashes", {})
    
    integrity_issues = {
        "missing_files": [],
        "hash_mismatch": [],
        "new_files": []
    }
    
    # 현재 파일들과 해시 비교
    for stage in ["1_raw", "2_extracted", "3_clustered", "4_labeled", "5_embeddings"]:
        stage_path = domain_path / stage
        if not stage_path.exists():
            continue
        
        for file_path in stage_path.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith((".json", ".md")):
                # 크로스플랫폼 호환성을 위해 경로 구분자 통일
                file_key = str(file_path.relative_to(domain_path)).replace("\\", "/")
                current_hash = calculate_file_hash(file_path)
                
                if file_key not in file_hashes:
                    integrity_issues["new_files"].append(file_key)
                else:
                    stored_hash = file_hashes[file_key]["hash"]
                    if current_hash != stored_hash:
                        integrity_issues["hash_mismatch"].append(file_key)
    
    # 저장된 해시는 있지만 파일이 없는 경우
    for file_key in file_hashes:
        file_path = domain_path / file_key
        if not file_path.exists():
            integrity_issues["missing_files"].append(file_key)
    
    return integrity_issues

def generate_lineage_report(domain_path: Path, source_file: str) -> Dict:
    """특정 파일의 계보 리포트를 생성합니다."""
    trace_data = load_traceability_data(domain_path)
    
    lineage = {
        "source_file": source_file,
        "lineage": [],
        "metadata": {
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # 1단계  2단계 추적
    stage_1_to_2 = trace_data.get("stage_1_to_2", {})
    if source_file in stage_1_to_2:
        extracted_files = stage_1_to_2[source_file]
        if isinstance(extracted_files, list):
            for extracted_file in extracted_files:
                lineage["lineage"].append({
                    "stage": "1_to_2",
                    "source": source_file,
                    "target": extracted_file,
                    "type": "feature_extraction"
                })
                
                # 2단계  3단계 추적
                stage_2_to_3 = trace_data.get("stage_2_to_3", {})
                if extracted_file in stage_2_to_3:
                    clustered_file = stage_2_to_3[extracted_file]
                    lineage["lineage"].append({
                        "stage": "2_to_3",
                        "source": extracted_file,
                        "target": clustered_file,
                        "type": "clustering"
                    })
    
    return lineage

def print_lineage_report(lineage: Dict):
    """계보 리포트를 출력합니다."""
    print(f" 파일 계보 리포트")
    print(f"원본 파일: {lineage['source_file']}")
    print(f"생성 시간: {lineage['metadata']['generated_at']}")
    print()
    
    if not lineage["lineage"]:
        print(" 계보 정보가 없습니다.")
        return
    
    print(" 계보 정보:")
    for i, step in enumerate(lineage["lineage"], 1):
        print(f"  {i}. {step['stage']}: {step['source']}  {step['target']} ({step['type']})")

def export_traceability_data(domain_path: Path, output_file: str):
    """추적성 데이터를 외부 파일로 내보냅니다."""
    trace_data = load_traceability_data(domain_path)
    
    export_data = {
        "domain_info": {
            "domain_path": str(domain_path),
            "exported_at": datetime.now().isoformat()
        },
        "traceability_data": trace_data
    }
    
    output_path = Path(output_file)
    output_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" 추적성 데이터가 {output_file}에 내보내졌습니다.")

def import_traceability_data(domain_path: Path, input_file: str):
    """외부 파일에서 추적성 데이터를 가져옵니다."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f" 입력 파일이 존재하지 않습니다: {input_file}")
        return False
    
    try:
        import_data = json.loads(input_path.read_text(encoding="utf-8"))
        trace_data = import_data.get("traceability_data", {})
        
        save_traceability_data(domain_path, trace_data)
        print(f" 추적성 데이터가 {input_file}에서 가져와졌습니다.")
        return True
        
    except Exception as e:
        print(f" 추적성 데이터 가져오기 오류: {e}")
        return False

def cleanup_old_hashes(domain_path: Path, days: int = 30):
    """오래된 해시 정보를 정리합니다."""
    print(f" {days}일 이상 된 해시 정보 정리 중...")
    
    trace_data = load_traceability_data(domain_path)
    file_hashes = trace_data.get("file_hashes", {})
    
    cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
    removed_count = 0
    
    keys_to_remove = []
    for file_key, hash_info in file_hashes.items():
        if "updated_at" in hash_info:
            try:
                updated_time = datetime.fromisoformat(hash_info["updated_at"]).timestamp()
                if updated_time < cutoff_date:
                    keys_to_remove.append(file_key)
            except:
                pass
    
    for key in keys_to_remove:
        del file_hashes[key]
        removed_count += 1
    
    trace_data["file_hashes"] = file_hashes
    save_traceability_data(domain_path, trace_data)
    print(f" {removed_count}개 오래된 해시 정보 정리 완료")

def main():
    parser = argparse.ArgumentParser(description="추적성 관리 스크립트")
    parser.add_argument("domain", help="도메인명 (humanoid, factory, powerline_inspection)")
    parser.add_argument("feature", help="기능명 (face_recognition, defect_detection, inspection)")
    
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령")
    
    # 해시 스캔 명령
    scan_parser = subparsers.add_parser("scan", help="파일 해시 스캔 및 업데이트")
    
    # 무결성 검증 명령
    verify_parser = subparsers.add_parser("verify", help="파일 무결성 검증")
    
    # 계보 리포트 명령
    lineage_parser = subparsers.add_parser("lineage", help="파일 계보 리포트 생성")
    lineage_parser.add_argument("source_file", help="원본 파일명")
    
    # 내보내기 명령
    export_parser = subparsers.add_parser("export", help="추적성 데이터 내보내기")
    export_parser.add_argument("output_file", help="출력 파일 경로")
    
    # 가져오기 명령
    import_parser = subparsers.add_parser("import", help="추적성 데이터 가져오기")
    import_parser.add_argument("input_file", help="입력 파일 경로")
    
    # 정리 명령
    cleanup_parser = subparsers.add_parser("cleanup", help="오래된 해시 정보 정리")
    cleanup_parser.add_argument("--days", type=int, default=30, help="보관 기간 (일)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    domain_path = Path("data/domains") / args.domain / args.feature
    
    if not domain_path.exists():
        print(f" 도메인이 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    try:
        if args.command == "scan":
            scan_and_update_hashes(domain_path)
        elif args.command == "verify":
            issues = verify_file_integrity(domain_path)
            print(f" 무결성 검증 결과:")
            print(f"  - 새 파일: {len(issues['new_files'])}개")
            print(f"  - 해시 불일치: {len(issues['hash_mismatch'])}개")
            print(f"  - 누락 파일: {len(issues['missing_files'])}개")
        elif args.command == "lineage":
            lineage = generate_lineage_report(domain_path, args.source_file)
            print_lineage_report(lineage)
        elif args.command == "export":
            export_traceability_data(domain_path, args.output_file)
        elif args.command == "import":
            import_traceability_data(domain_path, args.input_file)
        elif args.command == "cleanup":
            cleanup_old_hashes(domain_path, args.days)
            
    except Exception as e:
        print(f" 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
