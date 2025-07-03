#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
긴급 얼굴 재그룹핑 스크립트

잘못된 그룹핑 결과를 삭제하고 개선된 알고리즘으로 재그룹핑합니다.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger
from run_unified_ai_grouping_processor import UnifiedAIGroupingProcessor

def emergency_regroup():
    """긴급 재그룹핑 실행"""
    
    # 로깅 설정
    setup_logging()
    logger = get_logger(__name__)
    
    print("🚨 긴급 얼굴 재그룹핑 시작")
    print("=" * 60)
    
    # 경로 설정
    grouped_dir = project_root / 'data' / 'domains' / 'face_recognition' / 'staging' / 'grouped'
    backup_dir = project_root / 'data' / 'domains' / 'face_recognition' / 'staging' / 'grouped_backup'
    
    # 1. 기존 그룹 백업
    if grouped_dir.exists():
        print("📁 기존 그룹 백업 중...")
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        shutil.copytree(grouped_dir, backup_dir)
        print(f"✅ 백업 완료: {backup_dir}")
        
        # 기존 그룹 삭제
        print("🗑️ 잘못된 그룹 삭제 중...")
        for item in grouped_dir.iterdir():
            if item.is_dir() and item.name.startswith('group_'):
                shutil.rmtree(item)
                print(f"   삭제됨: {item.name}")
    else:
        print("⚠️ 기존 그룹 폴더가 없습니다.")
    
    # 2. 개선된 설정으로 재그룹핑
    print("\n🤖 개선된 AI 알고리즘으로 재그룹핑 시작...")
    
    # 매우 엄격한 설정
    config = {
        'similarity_threshold': 0.90,
        'debug': True,
        'verbose': True,
        'dry_run': False,
        'strict_mode': True
    }
    
    try:
        processor = UnifiedAIGroupingProcessor(config)
        processor.process_grouping('all')
        
        print("\n✅ 재그룹핑 완료!")
        
        # 결과 확인
        if grouped_dir.exists():
            groups = [d for d in grouped_dir.iterdir() if d.is_dir() and d.name.startswith('group_')]
            print(f"📊 생성된 그룹 수: {len(groups)}개")
            
            for group_dir in groups:
                faces = [f for f in group_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                print(f"   {group_dir.name}: {len(faces)}개 얼굴")
                
                if len(faces) > 20:
                    print(f"   ⚠️ 경고: 그룹이 여전히 너무 큽니다!")
        
    except Exception as e:
        logger.error(f"재그룹핑 실패: {e}")
        print(f"❌ 재그룹핑 실패: {e}")
        
        # 백업 복원 옵션 제공
        restore = input("\n백업을 복원하시겠습니까? (y/N): ").lower().strip()
        if restore in ['y', 'yes']:
            if backup_dir.exists():
                if grouped_dir.exists():
                    shutil.rmtree(grouped_dir)
                shutil.copytree(backup_dir, grouped_dir)
                print("✅ 백업이 복원되었습니다.")
            else:
                print("❌ 백업 폴더를 찾을 수 없습니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="긴급 얼굴 재그룹핑")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="확인 없이 바로 실행"
    )
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("⚠️ 이 스크립트는 기존 그룹핑 결과를 삭제하고 재그룹핑합니다.")
        print("   계속하시겠습니까?")
        
        confirm = input("Continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("취소되었습니다.")
            return
    
    emergency_regroup()

if __name__ == "__main__":
    main() 