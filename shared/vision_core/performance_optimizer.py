#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vision System Performance Optimizer.

비전 시스템 성능 최적화를 위한 배치 처리 및 메모리 관리 시스템입니다.
"""

import time
import threading
import queue
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import deque
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """배치 처리 결과"""
    batch_id: str
    results: List[Any]
    processing_time: float
    success_count: int
    error_count: int
    batch_size: int


class AdaptiveBatchProcessor:
    """적응형 배치 처리기"""
    
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.1):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        # 성능 추적
        self.performance_history = deque(maxlen=100)
        self.current_batch_size = min_batch_size
        
        # 큐 및 워커
        self.input_queue = queue.Queue()
        self.is_running = False
        
        # 통계
        self.total_processed = 0
        self.total_errors = 0
        self.total_time = 0.0
    
    def submit(self, item: Any) -> str:
        """아이템 제출"""
        item_id = f"item_{int(time.time() * 1000000)}"
        self.input_queue.put((item_id, item))
        return item_id
    
    def get_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        throughput = self.total_processed / self.total_time if self.total_time > 0 else 0
        
        return {
            'current_batch_size': self.current_batch_size,
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'total_time': self.total_time,
            'throughput': throughput,
            'queue_size': self.input_queue.qsize()
        }


class MemoryOptimizer:
    """메모리 최적화"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_stats = deque(maxlen=60)  # 1분간 통계
        self.monitoring_thread = None
        self.is_monitoring = False
    
    def start_monitoring(self):
        """메모리 모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent
                self.memory_stats.append(memory_percent)
                
                if memory_percent > self.max_memory_percent:
                    self._trigger_cleanup()
                
                time.sleep(1)  # 1초마다 체크
                
            except Exception as e:
                logger.error(f"메모리 모니터링 오류: {str(e)}")
    
    def _trigger_cleanup(self):
        """메모리 정리 트리거"""
        logger.warning(f"메모리 사용률 높음: {psutil.virtual_memory().percent:.1f}%")
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        logger.info(f"가비지 컬렉션 완료: {collected}개 객체 정리")
        
        # 추가 정리 로직
        self._cleanup_caches()
    
    def _cleanup_caches(self):
        """캐시 정리"""
        # 여기에 시스템별 캐시 정리 로직 구현
        pass
    
    def get_memory_stats(self) -> Dict[str, float]:
        """메모리 통계 반환"""
        current_memory = psutil.virtual_memory()
        
        return {
            'current_percent': current_memory.percent,
            'available_gb': current_memory.available / 1024**3,
            'used_gb': current_memory.used / 1024**3,
            'total_gb': current_memory.total / 1024**3,
            'avg_percent_1min': np.mean(self.memory_stats) if self.memory_stats else 0,
            'max_percent_1min': np.max(self.memory_stats) if self.memory_stats else 0
        }


class GPUResourceManager:
    """GPU 리소스 관리"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.gpu_stats = deque(maxlen=60)
    
    def _check_gpu_availability(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_optimal_batch_size(self, base_batch_size: int = 8) -> int:
        """최적 배치 크기 계산"""
        if not self.gpu_available:
            return base_batch_size
        
        try:
            import torch
            
            # GPU 메모리 사용률 기반 배치 크기 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated(0)
            free_memory = gpu_memory - used_memory
            
            memory_utilization = used_memory / gpu_memory
            
            if memory_utilization < 0.5:  # 50% 미만
                return min(base_batch_size * 2, 32)
            elif memory_utilization < 0.7:  # 70% 미만
                return base_batch_size
            else:  # 70% 이상
                return max(base_batch_size // 2, 1)
                
        except Exception as e:
            logger.warning(f"GPU 메모리 확인 실패: {str(e)}")
            return base_batch_size
    
    def clear_cache(self):
        """GPU 캐시 정리"""
        if not self.gpu_available:
            return
        
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("GPU 캐시 정리 완료")
        except Exception as e:
            logger.error(f"GPU 캐시 정리 실패: {str(e)}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """GPU 통계 반환"""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            import torch
            
            gpu_props = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_cached = torch.cuda.memory_reserved(0)
            
            return {
                'available': True,
                'name': gpu_props.name,
                'total_memory_gb': gpu_props.total_memory / 1024**3,
                'allocated_memory_gb': memory_allocated / 1024**3,
                'cached_memory_gb': memory_cached / 1024**3,
                'utilization_percent': (memory_allocated / gpu_props.total_memory) * 100
            }
            
        except Exception as e:
            logger.error(f"GPU 통계 수집 실패: {str(e)}")
            return {'available': False, 'error': str(e)}


class ParallelProcessor:
    """병렬 처리기"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.executor = None
    
    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def process_parallel(self, 
                        items: List[Any], 
                        processor_func: Callable,
                        chunk_size: Optional[int] = None) -> List[Any]:
        """병렬 처리"""
        if not self.executor:
            raise RuntimeError("ParallelProcessor를 with문으로 사용해야 합니다")
        
        if not items:
            return []
        
        chunk_size = chunk_size or max(1, len(items) // self.max_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, chunk, processor_func)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"청크 처리 실패: {str(e)}")
        
        return results
    
    def _process_chunk(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """청크 처리"""
        return [processor_func(item) for item in chunk]


# 전역 인스턴스들
memory_optimizer = MemoryOptimizer()
gpu_manager = GPUResourceManager()

# 컨텍스트 매니저로 사용할 수 있는 유틸리티
class PerformanceContext:
    """성능 최적화 컨텍스트"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        
        # GPU 캐시 정리
        gpu_manager.clear_cache()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        processing_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        logger.info(f"처리 완료 - 시간: {processing_time:.3f}초, 메모리 변화: {memory_delta/1024**2:.1f}MB")


if __name__ == "__main__":
    # 사용 예시
    def sample_processor(items):
        """샘플 처리 함수"""
        # 시뮬레이션: 각 아이템 처리에 약간의 시간 소요
        time.sleep(0.01)
        return [f"processed_{item}" for item in items]
    
    # 적응형 배치 처리 테스트
    processor = AdaptiveBatchProcessor(min_batch_size=2, max_batch_size=8)
    processor.start(sample_processor)
    
    # 아이템 제출
    for i in range(20):
        processor.submit(f"item_{i}")
    
    time.sleep(2)  # 처리 대기
    
    stats = processor.get_stats()
    print(f"처리 통계: {stats}")
    
    processor.stop() 