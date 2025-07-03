#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
환경 분석 서비스

하드웨어 환경을 분석하여 최적의 모델과 설정을 선택합니다.
"""

import os
import sys
import logging
import platform
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging

class EnvironmentAnalyzer:
    """환경 분석기"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 하드웨어 정보
        self.hardware_info = {}
        
        # 모델 선택 기준
        self.model_selection_criteria = {
            'detection': {
                'high_performance': {
                    'model': 'retinaface',
                    'config': {
                        'confidence_threshold': 0.7,
                        'nms_threshold': 0.4,
                        'device': 'cuda'
                    }
                },
                'balanced': {
                    'model': 'opencv_dnn',
                    'config': {
                        'confidence_threshold': 0.6,
                        'nms_threshold': 0.4,
                        'device': 'cpu'
                    }
                },
                'lightweight': {
                    'model': 'opencv_cascade',
                    'config': {
                        'scale_factor': 1.1,
                        'min_neighbors': 5,
                        'min_size': (30, 30),
                        'device': 'cpu'
                    }
                }
            },
            'recognition': {
                'high_performance': {
                    'model': 'arcface',
                    'config': {
                        'embedding_size': 512,
                        'device': 'cuda'
                    }
                },
                'balanced': {
                    'model': 'facenet',
                    'config': {
                        'embedding_size': 128,
                        'device': 'cpu'
                    }
                },
                'lightweight': {
                    'model': 'simple_cnn',
                    'config': {
                        'embedding_size': 64,
                        'device': 'cpu'
                    }
                }
            }
        }
        
        self.logger.info("환경 분석기 초기화 완료")
    
    def analyze_environment(self) -> Dict:
        """전체 환경 분석"""
        self.logger.info("환경 분석 시작")
        
        # 하드웨어 정보 수집
        self.hardware_info = {
            'system': self._get_system_info(),
            'cpu': self._get_cpu_info(),
            'memory': self._get_memory_info(),
            'gpu': self._get_gpu_info(),
            'storage': self._get_storage_info()
        }
        
        # 성능 등급 결정
        performance_tier = self._determine_performance_tier()
        
        # 최적 모델 선택
        optimal_models = self._select_optimal_models(performance_tier)
        
        # 권장 설정 생성
        recommended_config = self._generate_recommended_config(performance_tier)
        
        analysis_result = {
            'hardware_info': self.hardware_info,
            'performance_tier': performance_tier,
            'optimal_models': optimal_models,
            'recommended_config': recommended_config,
            'analysis_timestamp': self._get_timestamp()
        }
        
        self.logger.info(f"환경 분석 완료 - 성능 등급: {performance_tier}")
        return analysis_result
    
    def _get_system_info(self) -> Dict:
        """시스템 정보 수집"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def _get_cpu_info(self) -> Dict:
        """CPU 정보 수집"""
        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_max': 0,
            'frequency_current': 0
        }
        
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['frequency_max'] = cpu_freq.max
                cpu_info['frequency_current'] = cpu_freq.current
        except:
            pass
        
        return cpu_info
    
    def _get_memory_info(self) -> Dict:
        """메모리 정보 수집"""
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percent': memory.percent
        }
    
    def _get_gpu_info(self) -> Dict:
        """GPU 정보 수집"""
        gpu_info = {
            'available': False,
            'cuda_available': False,
            'devices': []
        }
        
        # CUDA 확인
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['available'] = True
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'id': i,
                        'name': device_props.name,
                        'memory_total_gb': round(device_props.total_memory / (1024**3), 2),
                        'compute_capability': f"{device_props.major}.{device_props.minor}"
                    })
        except ImportError:
            self.logger.warning("PyTorch가 설치되지 않음 - GPU 정보 수집 제한")
        
        # OpenCL 확인 (추가적으로)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info['nvidia_smi_available'] = True
        except:
            gpu_info['nvidia_smi_available'] = False
        
        return gpu_info
    
    def _get_storage_info(self) -> Dict:
        """저장소 정보 수집"""
        try:
            # Windows에서는 절대 경로 사용
            current_drive = os.getcwd()[:3]  # C:\ 형태
            disk_usage = psutil.disk_usage(current_drive)
            
            return {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
            }
        except Exception as e:
            self.logger.warning(f"저장소 정보 수집 실패: {str(e)}")
            return {
                'total_gb': 0,
                'free_gb': 0,
                'used_percent': 0
            }
    
    def _determine_performance_tier(self) -> str:
        """성능 등급 결정"""
        score = 0
        
        # GPU 점수 (가장 중요)
        if self.hardware_info['gpu']['cuda_available']:
            if self.hardware_info['gpu']['devices']:
                gpu_memory = self.hardware_info['gpu']['devices'][0]['memory_total_gb']
                if gpu_memory >= 8:
                    score += 40  # 고성능 GPU
                elif gpu_memory >= 4:
                    score += 25  # 중급 GPU
                else:
                    score += 15  # 저급 GPU
        
        # CPU 점수
        cpu_cores = self.hardware_info['cpu']['cores_logical']
        if cpu_cores >= 8:
            score += 20
        elif cpu_cores >= 4:
            score += 15
        else:
            score += 10
        
        # 메모리 점수
        memory_gb = self.hardware_info['memory']['total_gb']
        if memory_gb >= 16:
            score += 20
        elif memory_gb >= 8:
            score += 15
        else:
            score += 10
        
        # CPU 주파수 점수
        cpu_freq = self.hardware_info['cpu']['frequency_max']
        if cpu_freq >= 3000:  # 3GHz 이상
            score += 10
        elif cpu_freq >= 2000:  # 2GHz 이상
            score += 5
        
        # 성능 등급 결정
        if score >= 70:
            return 'high_performance'
        elif score >= 40:
            return 'balanced'
        else:
            return 'lightweight'
    
    def _select_optimal_models(self, performance_tier: str) -> Dict:
        """최적 모델 선택"""
        return {
            'detection': self.model_selection_criteria['detection'][performance_tier],
            'recognition': self.model_selection_criteria['recognition'][performance_tier]
        }
    
    def _generate_recommended_config(self, performance_tier: str) -> Dict:
        """권장 설정 생성"""
        base_config = {
            'lightweight': {
                'batch_size': 1,
                'num_workers': 2,
                'image_size': (224, 224),
                'max_concurrent_streams': 1
            },
            'balanced': {
                'batch_size': 2,
                'num_workers': 4,
                'image_size': (320, 320),
                'max_concurrent_streams': 2
            },
            'high_performance': {
                'batch_size': 4,
                'num_workers': 8,
                'image_size': (640, 640),
                'max_concurrent_streams': 4
            }
        }
        
        config = base_config[performance_tier].copy()
        
        # 하드웨어별 세부 조정
        if performance_tier == 'high_performance' and self.hardware_info['gpu']['cuda_available']:
            config['use_mixed_precision'] = True
            config['tensorrt_optimization'] = True
        
        return config
    
    def _get_timestamp(self) -> str:
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_optimal_detection_engine(self) -> str:
        """최적 검출 엔진 반환"""
        analysis = self.analyze_environment()
        return analysis['optimal_models']['detection']['model']
    
    def get_optimal_recognition_engine(self) -> str:
        """최적 인식 엔진 반환"""
        analysis = self.analyze_environment()
        return analysis['optimal_models']['recognition']['model']
    
    def get_detection_config(self) -> Dict:
        """검출 설정 반환"""
        analysis = self.analyze_environment()
        return analysis['optimal_models']['detection']['config']
    
    def get_recognition_config(self) -> Dict:
        """인식 설정 반환"""
        analysis = self.analyze_environment()
        return analysis['optimal_models']['recognition']['config']
    
    def print_analysis_report(self):
        """분석 결과 출력"""
        analysis = self.analyze_environment()
        
        print("\n" + "="*60)
        print("🔍 환경 분석 보고서")
        print("="*60)
        
        # 시스템 정보
        system = analysis['hardware_info']['system']
        print(f"\n💻 시스템 정보:")
        print(f"   플랫폼: {system['platform']} {system['architecture']}")
        print(f"   프로세서: {system['processor']}")
        
        # CPU 정보
        cpu = analysis['hardware_info']['cpu']
        print(f"\n🔧 CPU 정보:")
        print(f"   코어: {cpu['cores_physical']}물리 / {cpu['cores_logical']}논리")
        if cpu['frequency_max'] > 0:
            print(f"   주파수: {cpu['frequency_max']:.0f}MHz (최대)")
        
        # 메모리 정보
        memory = analysis['hardware_info']['memory']
        print(f"\n💾 메모리 정보:")
        print(f"   총 용량: {memory['total_gb']:.1f}GB")
        print(f"   사용 가능: {memory['available_gb']:.1f}GB")
        
        # GPU 정보
        gpu = analysis['hardware_info']['gpu']
        print(f"\n🎮 GPU 정보:")
        if gpu['cuda_available']:
            print(f"   CUDA 사용 가능: ✅")
            for device in gpu['devices']:
                print(f"   - {device['name']} ({device['memory_total_gb']:.1f}GB)")
        else:
            print(f"   CUDA 사용 가능: ❌")
        
        # 성능 등급
        tier = analysis['performance_tier']
        tier_names = {
            'high_performance': '고성능 🚀',
            'balanced': '균형형 ⚖️',
            'lightweight': '경량형 🪶'
        }
        print(f"\n📊 성능 등급: {tier_names.get(tier, tier)}")
        
        # 권장 모델
        models = analysis['optimal_models']
        print(f"\n🎯 권장 모델:")
        print(f"   얼굴 검출: {models['detection']['model']}")
        print(f"   얼굴 인식: {models['recognition']['model']}")
        
        # 권장 설정
        config = analysis['recommended_config']
        print(f"\n⚙️  권장 설정:")
        print(f"   배치 크기: {config['batch_size']}")
        print(f"   이미지 크기: {config['image_size']}")
        print(f"   동시 스트림: {config['max_concurrent_streams']}")
        
        print("="*60)
        
        return analysis

def main():
    """테스트용 메인 함수"""
    analyzer = EnvironmentAnalyzer()
    analyzer.print_analysis_report()

if __name__ == "__main__":
    main() 