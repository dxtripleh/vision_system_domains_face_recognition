#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
올인원 얼굴 인식 시스템 마스터 런처.

모든 얼굴 인식 관련 기능을 하나의 메뉴에서 실행할 수 있습니다.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from common.logging import setup_logging

logger = logging.getLogger(__name__)


class FaceSystemLauncher:
    """얼굴 인식 시스템 통합 런처"""
    
    def __init__(self):
        """초기화"""
        self.project_root = Path(__file__).parent
        
        # 실행 가능한 스크립트들
        self.scripts = {
            # 데이터 수집
            'data_collection': {
                'enhanced_collector': 'scripts/data_collection/enhanced_data_collector.py',
                'realtime_capture': 'domains/face_recognition/runners/data_collection/run_capture_and_register.py',
                'unified_capture': 'domains/face_recognition/runners/data_collection/run_unified_face_capture.py',  # 🎯 사용자 제안 통합 시스템
                'auto_collector': 'domains/face_recognition/runners/data_collection/run_auto_face_collector.py',
                'batch_processor': 'domains/face_recognition/runners/data_collection/run_batch_face_processor.py',  # 📁 사진/동영상 업로드 처리
                'smart_batch_processor': 'domains/face_recognition/runners/data_collection/run_smart_batch_processor.py'  # 🧠 스마트 배치 처리 (자동 그룹핑)
            },
            
            # 얼굴 인식 실행
            'recognition': {
                'web_interface': 'scripts/interfaces/web/app.py',
                'realtime_demo': 'domains/face_recognition/runners/recognition/run_demo.py',
                'realtime_recognition': 'domains/face_recognition/runners/recognition/run_realtime_recognition.py',
                'advanced_recognition': 'domains/face_recognition/runners/recognition/run_advanced_recognition.py',
                'cli_interface': 'scripts/interfaces/cli/run_face_recognition_cli.py',
                'main_script': 'main.py'
            },
            
            # 시스템 관리
            'management': {
                'system_health': 'scripts/core/test/test_system_health.py',
                'hardware_validation': 'scripts/core/validation/validate_hardware_connection.py',
                'performance_monitor': 'scripts/core/monitoring/performance_monitor.py',
                'group_manager': 'domains/face_recognition/runners/management/run_group_manager.py'
            },
            
            # 훈련 및 모델 관리
            'training': {
                'model_training_pipeline': 'domains/face_recognition/runners/training/run_model_training_pipeline.py'
            }
        }
    
    def display_main_menu(self):
        """메인 메뉴 표시"""
        print("🎯 얼굴 인식 시스템 - 올인원 런처")
        print("=" * 60)
        print("📊 데이터 수집 및 학습")
        print("  1. 향상된 데이터 수집 (카메라)")
        print("  2. 실시간 캡처 & 등록")
        print("  3. 🎯 통합 캡처 시스템 (사용자 제안)")
        print("  4. 자동 얼굴 수집기")
        print("  5. 📁 배치 얼굴 처리 (업로드 파일용)")
        print("  6. 🧠 스마트 배치 처리 (자동 그룹핑)")
        print()
        print("🔍 얼굴 인식 실행")
        print("  7. 웹 인터페이스 (추천)")
        print("  8. 실시간 데모 (카메라)")
        print("  9. 실시간 인식 시스템")
        print("  10. 고급 인식 시스템")
        print("  11. CLI 인터페이스")
        print("  12. 메인 스크립트")
        print()
        print("⚙️ 시스템 관리")
        print("  13. 시스템 상태 점검")
        print("  14. 하드웨어 연결 확인")
        print("  15. 성능 모니터링")
        print("  16. 얼굴 그룹 관리")
        print()
        print("🤖 모델 훈련")
        print("  17. 모델 훈련 파이프라인")
        print()
        print("📚 도움말 & 가이드")
        print("  18. 사용법 가이드")
        print("  19. 데이터 흐름 설명")
        print("  20. 개발 상태 확인")
        print()
        print("  0. 종료")
        print("=" * 60)
    
    def run_script(self, script_path: str, args: list = None):
        """스크립트 실행"""
        full_path = self.project_root / script_path
        
        if not full_path.exists():
            print(f"❌ 스크립트를 찾을 수 없습니다: {script_path}")
            return False
        
        try:
            cmd = [sys.executable, str(full_path)]
            if args:
                cmd.extend(args)
            
            print(f"🚀 실행 중: {script_path}")
            print(f"📁 명령어: {' '.join(cmd)}")
            print("-" * 50)
            
            subprocess.run(cmd, cwd=str(self.project_root))
            return True
            
        except Exception as e:
            print(f"❌ 실행 실패: {str(e)}")
            return False
    
    def show_data_collection_guide(self):
        """데이터 수집 가이드"""
        print("\n📊 데이터 수집 및 학습 가이드")
        print("=" * 50)
        print("🎯 목적: 향후 새로운 모델 훈련을 위한 데이터 수집")
        print()
        print("📂 저장 위치:")
        print("  • 원본 이미지: datasets/face_recognition/raw/original_images/")
        print("  • 얼굴 크롭: datasets/face_recognition/raw/face_crops/")
        print("  • 메타데이터: datasets/face_recognition/raw/metadata/")
        print()
        print("🔄 데이터 흐름:")
        print("  1. 카메라/업로드로 원본 수집")
        print("  2. 얼굴 검출 & 품질 평가")
        print("  3. 원본 + 크롭 + 메타데이터 저장")
        print("  4. 향후 새 모델 훈련에 사용")
        print()
        print("⚡ 현재 vs 향후:")
        print("  • 현재: 기존 모델 (RetinaFace, ArcFace) 사용")
        print("  • 향후: 수집된 데이터로 자체 모델 훈련")
        print("  • 목표: 앙상블 또는 모델 교체로 인식률 향상")
    
    def show_data_flow_explanation(self):
        """데이터 흐름 설명"""
        print("\n📋 데이터 흐름 상세 설명")
        print("=" * 50)
        print()
        print("🗂️ 폴더별 역할:")
        print("┌─ data/storage/     : 현재 운영 중인 등록된 인물 데이터")
        print("│  ├─ persons/       : 인물 정보 (이름, ID)")
        print("│  └─ faces/         : 얼굴 임베딩 (512차원 벡터)")
        print("│")
        print("├─ datasets/         : 새 모델 훈련용 데이터 (중요!)")
        print("│  ├─ raw/           : 원본 수집 데이터")
        print("│  ├─ processed/     : 전처리된 데이터")
        print("│  ├─ augmented/     : 증강된 데이터")
        print("│  └─ splits/        : train/val/test 분할")
        print("│")
        print("└─ data/temp/        : 임시 파일 (자동 정리)")
        print("   └─ data/output/   : 처리 결과물")
        print()
        print("🔄 지속적 학습 전략:")
        print("  1️⃣ 운영 중 수집: data/storage/ (실시간 사용)")
        print("  2️⃣ 훈련용 변환: datasets/ (모델 개발)")
        print("  3️⃣ 새 모델 훈련: 수집된 데이터 활용")
        print("  4️⃣ 성능 비교: 기존 vs 신규 모델")
        print("  5️⃣ 모델 교체/앙상블: 인식률 향상")
    
    def show_usage_guide(self):
        """사용법 가이드"""
        print("\n📚 단계별 사용 가이드")
        print("=" * 50)
        print()
        print("🚀 첫 사용자를 위한 순서:")
        print("  1. 하드웨어 연결 확인 (메뉴 14)")
        print("  2. 시스템 상태 점검 (메뉴 13)")
        print("  3. 웹 인터페이스 실행 (메뉴 7)")
        print("  4. 브라우저에서 http://localhost:5000 접속")
        print("  5. 인물 등록 → 이미지 업로드 → 얼굴 등록")
        print("  6. 실시간 인식 테스트 (메뉴 8 또는 9)")
        print()
        print("📊 데이터 수집하고 싶다면:")
        print("  1. 향상된 데이터 수집 실행 (메뉴 1)")
        print("  2. 인물 이름 입력 후 's' 키로 다양한 각도 수집")
        print("  3. datasets/ 폴더에 체계적으로 저장됨")
        print("  4. 향후 새 모델 훈련에 활용 가능")
        print()
        print("📁 보유한 사진/동영상이 있다면:")
        print("  1. 배치 얼굴 처리 실행 (메뉴 5)")
        print("  2. 사진/동영상 파일 경로 입력")
        print("  3. 자동 얼굴 검출 → 이름 지정")
        print("  4. 기존 시스템과 동일한 분기 처리")
        print()
        print("🧠 많은 파일의 같은 사람들을 효율적으로 처리하려면:")
        print("  1. 스마트 배치 처리 실행 (메뉴 6)")
        print("  2. data/temp/uploads/ 폴더에 모든 파일 넣기")
        print("  3. AI가 자동으로 같은 사람끼리 그룹핑")
        print("  4. 그룹별로 한 번만 이름 입력하면 끝!")
        print()
        print("🔍 간단한 테스트만 원한다면:")
        print("  • 웹 인터페이스 (메뉴 7) - 가장 사용하기 쉬움")
        print("  • 실시간 데모 (메뉴 8) - 카메라로 바로 확인")
    
    def check_development_status(self):
        """개발 상태 확인"""
        print("\n📈 현재 개발 상태")
        print("=" * 50)
        
        # 파일 존재 여부 확인
        status_checks = [
            ("웹 인터페이스", "scripts/interfaces/web/app.py"),
            ("실시간 인식", "scripts/core/run/run_realtime_face_recognition.py"),
            ("CLI 도구", "scripts/interfaces/cli/run_face_recognition_cli.py"),
            ("데이터 수집기", "scripts/data_collection/enhanced_data_collector.py"),
            ("메인 스크립트", "main.py"),
            ("도메인 API", "domains/face_recognition/interfaces/api/face_recognition_api.py")
        ]
        
        print("✅ 구현 완료:")
        for name, path in status_checks:
            if (self.project_root / path).exists():
                print(f"  • {name}")
        
        print("\n📊 데이터 상태:")
        
        # 등록된 인물 수 확인
        persons_dir = self.project_root / "data/storage/persons"
        if persons_dir.exists():
            person_files = list(persons_dir.glob("*.json"))
            print(f"  • 등록된 인물: {len(person_files) - 1}명")  # index 파일 제외
        
        # 수집된 훈련 데이터 확인
        datasets_dir = self.project_root / "datasets/face_recognition/raw/metadata"
        if datasets_dir.exists():
            dataset_files = list(datasets_dir.glob("*.json"))
            print(f"  • 수집된 훈련 데이터: {len(dataset_files)}개")
        else:
            print("  • 수집된 훈련 데이터: 0개 (메뉴 1로 수집 시작)")
        
        print("\n🎯 추천 다음 단계:")
        if not datasets_dir.exists() or len(list(datasets_dir.glob("*.json"))) == 0:
            print("  1. 데이터 수집부터 시작 (메뉴 1)")
            print("  2. 다양한 각도로 50-100개 얼굴 수집")
            print("  3. 향후 모델 훈련 기반 구축")
        else:
            print("  1. 더 많은 데이터 수집")
            print("  2. 데이터 품질 분석")
            print("  3. 모델 훈련 파이프라인 구축")
    
    def run(self):
        """메인 실행 루프"""
        setup_logging()
        
        while True:
            self.display_main_menu()
            
            try:
                choice = input("\n선택하세요 (0-20): ").strip()
                
                if choice == '0':
                    print("👋 시스템을 종료합니다.")
                    break
                
                elif choice == '1':
                    self.run_script(self.scripts['data_collection']['enhanced_collector'])
                
                elif choice == '2':
                    self.run_script(self.scripts['data_collection']['realtime_capture'])
                
                elif choice == '3':
                    self.run_script(self.scripts['data_collection']['unified_capture'])
                
                elif choice == '4':
                    self.run_script(self.scripts['data_collection']['auto_collector'])
                
                elif choice == '5':
                    self.run_script(self.scripts['data_collection']['batch_processor'])
                
                elif choice == '6':
                    self.run_script(self.scripts['data_collection']['smart_batch_processor'])
                
                elif choice == '7':
                    print("\n🌐 웹 서버 시작 후 브라우저에서 http://localhost:5000 접속")
                    self.run_script(self.scripts['recognition']['web_interface'])
                
                elif choice == '8':
                    self.run_script(self.scripts['recognition']['realtime_demo'])
                
                elif choice == '9':
                    self.run_script(self.scripts['recognition']['realtime_recognition'])
                
                elif choice == '10':
                    self.run_script(self.scripts['recognition']['advanced_recognition'])
                
                elif choice == '11':
                    self.run_script(self.scripts['recognition']['cli_interface'])
                
                elif choice == '12':
                    self.run_script(self.scripts['recognition']['main_script'], ['--mode', 'realtime'])
                
                elif choice == '13':
                    self.run_script(self.scripts['management']['system_health'])
                
                elif choice == '14':
                    self.run_script(self.scripts['management']['hardware_validation'])
                
                elif choice == '15':
                    self.run_script(self.scripts['management']['performance_monitor'])
                
                elif choice == '16':
                    self.run_script(self.scripts['management']['group_manager'])
                
                elif choice == '17':
                    self.run_script(self.scripts['training']['model_training_pipeline'])
                
                elif choice == '18':
                    self.show_usage_guide()
                
                elif choice == '19':
                    self.show_data_flow_explanation()
                
                elif choice == '20':
                    self.check_development_status()
                
                else:
                    print("❌ 잘못된 선택입니다. 0-20 사이의 숫자를 입력하세요.")
                
                if choice not in ['0', '18', '19', '20']:
                    input("\n✅ 완료되었습니다. Enter를 눌러 메뉴로 돌아가세요...")
                
            except KeyboardInterrupt:
                print("\n\n👋 사용자가 중단했습니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                input("Enter를 눌러 계속...")


if __name__ == "__main__":
    launcher = FaceSystemLauncher()
    launcher.run() 