#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
하드웨어 연결 검증 스크립트.

실제 하드웨어(카메라) 연결 상태를 검증하고 시뮬레이션 모드를 방지합니다.
"""

import cv2
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging, get_logger

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


class HardwareValidator:
    """하드웨어 연결 검증기"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_all(self) -> Dict[str, bool]:
        """모든 하드웨어 검증 수행"""
        logger.info("🔍 하드웨어 연결 검증 시작")
        
        # 시뮬레이션 방지 검사
        self.validation_results['simulation_check'] = self._check_simulation_prevention()
        
        # 카메라 연결 검사
        self.validation_results['camera_check'] = self._check_camera_connection()
        
        # 시스템 리소스 검사
        self.validation_results['system_check'] = self._check_system_resources()
        
        # GPU 검사 (선택적)
        self.validation_results['gpu_check'] = self._check_gpu_availability()
        
        # 전체 결과 판정
        all_passed = all(self.validation_results.values())
        
        if all_passed:
            logger.info("✅ 모든 하드웨어 검증 통과")
        else:
            logger.error("❌ 하드웨어 검증 실패")
            self._print_failure_details()
        
        return self.validation_results
    
    def _check_simulation_prevention(self) -> bool:
        """시뮬레이션 방지 검사"""
        logger.info("🚫 시뮬레이션 방지 검사")
        
        # 환경 변수 검사
        use_simulation = os.environ.get("USE_SIMULATION", "False").lower()
        if use_simulation in ["true", "1", "yes", "on"]:
            logger.error("❌ USE_SIMULATION 환경 변수가 활성화됨")
            return False
        
        # Mock 모드 검사
        use_mock = os.environ.get("USE_MOCK", "False").lower()
        if use_mock in ["true", "1", "yes", "on"]:
            logger.error("❌ USE_MOCK 환경 변수가 활성화됨")
            return False
        
        logger.info("✅ 시뮬레이션 방지 검사 통과")
        return True
    
    def _check_camera_connection(self) -> bool:
        """카메라 연결 검사"""
        logger.info("📹 카메라 연결 검사")
        
        connected_cameras = []
        
        # 카메라 ID 0~5까지 테스트
        for camera_id in range(6):
            if self._test_camera(camera_id):
                connected_cameras.append(camera_id)
        
        if not connected_cameras:
            logger.error("❌ 연결된 카메라가 없습니다")
            return False
        
        logger.info(f"✅ 카메라 연결 확인: {connected_cameras}")
        return True
    
    def _test_camera(self, camera_id: int) -> bool:
        """개별 카메라 테스트"""
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                return False
            
            # 프레임 읽기 테스트
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False
            
            # 프레임 유효성 검사
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                logger.warning(f"카메라 {camera_id}: 해상도가 너무 낮음 ({frame.shape})")
                return False
            
            logger.debug(f"카메라 {camera_id}: 정상 ({frame.shape})")
            return True
            
        except Exception as e:
            logger.debug(f"카메라 {camera_id} 테스트 실패: {str(e)}")
            return False
    
    def _check_system_resources(self) -> bool:
        """시스템 리소스 검사"""
        logger.info("💻 시스템 리소스 검사")
        
        # CPU 사용률 검사
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.warning(f"⚠️ CPU 사용률이 높음: {cpu_percent:.1f}%")
        
        # 메모리 검사
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        if memory_percent > 90:
            logger.warning(f"⚠️ 메모리 사용률이 높음: {memory_percent:.1f}%")
        
        # 디스크 공간 검사
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 90:
            logger.warning(f"⚠️ 디스크 사용률이 높음: {disk_percent:.1f}%")
        
        # 최소 요구사항 검사
        min_memory_gb = 4  # 4GB 최소 메모리
        available_memory_gb = memory.available / (1024**3)
        
        if available_memory_gb < min_memory_gb:
            logger.error(f"❌ 사용 가능한 메모리 부족: {available_memory_gb:.1f}GB < {min_memory_gb}GB")
            return False
        
        logger.info(f"✅ 시스템 리소스 정상 - CPU: {cpu_percent:.1f}%, 메모리: {memory_percent:.1f}%, 사용 가능: {available_memory_gb:.1f}GB")
        return True
    
    def _check_gpu_availability(self) -> bool:
        """GPU 사용 가능성 검사"""
        logger.info("🎮 GPU 검사")
        
        try:
            # OpenCV CUDA 지원 확인
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            
            if cuda_devices > 0:
                logger.info(f"✅ CUDA GPU 발견: {cuda_devices}개 디바이스")
                return True
            else:
                logger.info("ℹ️ CUDA GPU 없음 (CPU 모드로 동작)")
                return True  # GPU는 선택사항이므로 True 반환
                
        except Exception as e:
            logger.debug(f"GPU 검사 중 오류: {str(e)}")
            logger.info("ℹ️ GPU 검사 실패 (CPU 모드로 동작)")
            return True  # GPU는 선택사항이므로 True 반환
    
    def _print_failure_details(self):
        """검증 실패 상세 정보 출력"""
        logger.error("🔍 검증 실패 상세:")
        
        for check_name, result in self.validation_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            logger.error(f"  - {check_name}: {status}")
    
    def generate_hardware_report(self) -> Dict[str, any]:
        """하드웨어 상태 보고서 생성"""
        report = {
            'timestamp': time.time(),
            'validation_results': self.validation_results,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            },
            'camera_info': self._get_camera_details(),
            'gpu_info': self._get_gpu_details()
        }
        
        return report
    
    def _get_camera_details(self) -> List[Dict]:
        """카메라 상세 정보 수집"""
        cameras = []
        
        for camera_id in range(6):
            try:
                cap = cv2.VideoCapture(camera_id)
                
                if cap.isOpened():
                    # 카메라 속성 수집
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    cameras.append({
                        'id': camera_id,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'status': 'connected'
                    })
                    
                    cap.release()
                    
            except Exception as e:
                cameras.append({
                    'id': camera_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        return cameras
    
    def _get_gpu_details(self) -> Dict:
        """GPU 상세 정보 수집"""
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            
            return {
                'cuda_available': cuda_devices > 0,
                'cuda_device_count': cuda_devices,
                'opencv_cuda_support': True
            }
            
        except Exception as e:
            return {
                'cuda_available': False,
                'cuda_device_count': 0,
                'opencv_cuda_support': False,
                'error': str(e)
            }


def validate_hardware_for_runtime() -> bool:
    """
    런타임 실행 전 하드웨어 검증
    
    Returns:
        bool: 검증 통과 여부
    """
    validator = HardwareValidator()
    results = validator.validate_all()
    
    # 필수 검사만 확인 (GPU는 선택적)
    required_checks = ['simulation_check', 'camera_check', 'system_check']
    
    for check in required_checks:
        if not results.get(check, False):
            logger.error(f"❌ 필수 검사 실패: {check}")
            return False
    
    return True


def main():
    """메인 함수"""
    print("=" * 60)
    print("🔧 하드웨어 연결 검증")
    print("=" * 60)
    
    validator = HardwareValidator()
    results = validator.validate_all()
    
    # 보고서 생성
    report = validator.generate_hardware_report()
    
    # 결과 출력
    print("\n📊 검증 결과:")
    for check_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  - {check_name}: {status}")
    
    print(f"\n💻 시스템 정보:")
    print(f"  - CPU 코어: {report['system_info']['cpu_count']}개")
    print(f"  - 메모리: {report['system_info']['memory_total_gb']:.1f}GB")
    print(f"  - GPU: {'사용 가능' if report['gpu_info']['cuda_available'] else '사용 불가'}")
    
    print(f"\n📹 카메라 정보:")
    for camera in report['camera_info']:
        if camera['status'] == 'connected':
            print(f"  - 카메라 {camera['id']}: {camera['resolution']} @ {camera['fps']:.1f}fps")
    
    # 전체 결과
    overall_result = all(results.values())
    if overall_result:
        print("\n🎉 하드웨어 검증 완료! 시스템이 준비되었습니다.")
        return 0
    else:
        print("\n⚠️ 하드웨어 검증 실패! 문제를 해결한 후 다시 시도하세요.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 