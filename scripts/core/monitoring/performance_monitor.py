#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Monitoring System.

비전 시스템의 실시간 성능 모니터링 및 메트릭 수집 시스템입니다.
"""

import time
import threading
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int


@dataclass
class VisionMetrics:
    """비전 처리 메트릭"""
    timestamp: float
    fps: float
    processing_time_ms: float
    detection_count: int
    recognition_accuracy: float
    model_inference_time_ms: float
    queue_size: int


@dataclass
class ErrorMetrics:
    """에러 메트릭"""
    timestamp: float
    error_type: str
    error_count: int
    component: str
    error_rate: float


class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 history_size: int = 1000,
                 save_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.save_interval = save_interval
        
        # 메트릭 저장소
        self.system_metrics = deque(maxlen=history_size)
        self.vision_metrics = deque(maxlen=history_size)
        self.error_metrics = deque(maxlen=history_size)
        
        # 스레드 관리
        self.is_monitoring = False
        self.monitor_thread = None
        self.save_thread = None
        
        # 메트릭 수집기들
        self.fps_calculator = FPSCalculator()
        self.error_tracker = ErrorTracker()
        
        # 저장 경로
        self.metrics_dir = Path("data/logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 네트워크 베이스라인
        self._network_baseline = self._get_network_baseline()
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("모니터링이 이미 실행 중입니다")
            return
        
        self.is_monitoring = True
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 저장 스레드 시작
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
        
        logger.info("성능 모니터링 시작됨")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if self.save_thread:
            self.save_thread.join(timeout=5)
        
        # 최종 저장
        self._save_metrics()
        
        logger.info("성능 모니터링 중지됨")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 시스템 메트릭 수집
                system_metric = self._collect_system_metrics()
                self.system_metrics.append(system_metric)
                
                # 실시간 분석
                self._analyze_real_time(system_metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {str(e)}")
                time.sleep(1)
    
    def _save_loop(self):
        """저장 루프"""
        while self.is_monitoring:
            try:
                time.sleep(self.save_interval)
                self._save_metrics()
                
            except Exception as e:
                logger.error(f"저장 루프 오류: {str(e)}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        
        # 네트워크 정보
        net_io = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_bytes_sent=net_io.bytes_sent - self._network_baseline['sent'],
            network_bytes_recv=net_io.bytes_recv - self._network_baseline['recv']
        )
    
    def _get_network_baseline(self) -> Dict[str, int]:
        """네트워크 베이스라인 설정"""
        net_io = psutil.net_io_counters()
        return {
            'sent': net_io.bytes_sent,
            'recv': net_io.bytes_recv
        }
    
    def _analyze_real_time(self, metric: SystemMetrics):
        """실시간 분석"""
        # CPU 사용률 경고
        if metric.cpu_percent > 90:
            logger.warning(f"높은 CPU 사용률: {metric.cpu_percent:.1f}%")
        
        # 메모리 사용률 경고
        if metric.memory_percent > 85:
            logger.warning(f"높은 메모리 사용률: {metric.memory_percent:.1f}%")
        
        # 디스크 사용률 경고
        if metric.disk_usage_percent > 95:
            logger.error(f"디스크 공간 부족: {metric.disk_usage_percent:.1f}%")
    
    def record_vision_metrics(self, 
                            fps: float,
                            processing_time_ms: float,
                            detection_count: int,
                            recognition_accuracy: float = 0.0,
                            model_inference_time_ms: float = 0.0,
                            queue_size: int = 0):
        """비전 메트릭 기록"""
        metric = VisionMetrics(
            timestamp=time.time(),
            fps=fps,
            processing_time_ms=processing_time_ms,
            detection_count=detection_count,
            recognition_accuracy=recognition_accuracy,
            model_inference_time_ms=model_inference_time_ms,
            queue_size=queue_size
        )
        
        self.vision_metrics.append(metric)
    
    def record_error(self, error_type: str, component: str):
        """에러 기록"""
        self.error_tracker.record_error(error_type, component)
        
        # 에러 메트릭 생성
        error_rate = self.error_tracker.get_error_rate(component)
        error_count = self.error_tracker.get_error_count(component)
        
        metric = ErrorMetrics(
            timestamp=time.time(),
            error_type=error_type,
            error_count=error_count,
            component=component,
            error_rate=error_rate
        )
        
        self.error_metrics.append(metric)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 통계 조회"""
        stats = {
            'system': {},
            'vision': {},
            'errors': {}
        }
        
        # 시스템 통계
        if self.system_metrics:
            latest_system = self.system_metrics[-1]
            stats['system'] = {
                'cpu_percent': latest_system.cpu_percent,
                'memory_percent': latest_system.memory_percent,
                'memory_used_gb': latest_system.memory_used_gb,
                'disk_usage_percent': latest_system.disk_usage_percent,
                'uptime_minutes': (time.time() - self.system_metrics[0].timestamp) / 60
            }
        
        # 비전 통계
        if self.vision_metrics:
            recent_vision = list(self.vision_metrics)[-10:]  # 최근 10개
            avg_fps = sum(v.fps for v in recent_vision) / len(recent_vision)
            avg_processing_time = sum(v.processing_time_ms for v in recent_vision) / len(recent_vision)
            
            stats['vision'] = {
                'current_fps': recent_vision[-1].fps,
                'avg_fps_10s': avg_fps,
                'avg_processing_time_ms': avg_processing_time,
                'total_detections': sum(v.detection_count for v in self.vision_metrics)
            }
        
        # 에러 통계
        stats['errors'] = self.error_tracker.get_summary()
        
        return stats
    
    def _save_metrics(self):
        """메트릭 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 시스템 메트릭 저장
        if self.system_metrics:
            system_file = self.metrics_dir / f"system_metrics_{timestamp}.json"
            self._save_to_file(list(self.system_metrics), system_file)
        
        # 비전 메트릭 저장
        if self.vision_metrics:
            vision_file = self.metrics_dir / f"vision_metrics_{timestamp}.json"
            self._save_to_file(list(self.vision_metrics), vision_file)
        
        # 에러 메트릭 저장
        if self.error_metrics:
            error_file = self.metrics_dir / f"error_metrics_{timestamp}.json"
            self._save_to_file(list(self.error_metrics), error_file)
    
    def _save_to_file(self, data: List, file_path: Path):
        """데이터를 파일에 저장"""
        try:
            serializable_data = [asdict(item) for item in data]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"파일 저장 실패 {file_path}: {str(e)}")


class FPSCalculator:
    """FPS 계산기"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
    
    def tick(self) -> float:
        """프레임 처리 완료 시 호출"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0.0


class ErrorTracker:
    """에러 추적기"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = deque(maxlen=1000)
    
    def record_error(self, error_type: str, component: str):
        """에러 기록"""
        key = f"{component}:{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self.error_history.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'component': component
        })
    
    def get_error_count(self, component: str) -> int:
        """특정 컴포넌트의 총 에러 수"""
        return sum(
            count for key, count in self.error_counts.items()
            if key.startswith(f"{component}:")
        )
    
    def get_error_rate(self, component: str, time_window: int = 3600) -> float:
        """에러율 계산 (시간당)"""
        cutoff_time = time.time() - time_window
        recent_errors = [
            e for e in self.error_history
            if e['timestamp'] > cutoff_time and e['component'] == component
        ]
        return len(recent_errors) / (time_window / 3600)  # 시간당 에러 수
    
    def get_summary(self) -> Dict[str, Any]:
        """에러 요약"""
        return {
            'total_errors': len(self.error_history),
            'error_types': len(set(e['error_type'] for e in self.error_history)),
            'components': len(set(e['component'] for e in self.error_history)),
            'error_counts': dict(self.error_counts)
        }


# 전역 인스턴스
performance_monitor = PerformanceMonitor()


if __name__ == "__main__":
    # 테스트 코드
    monitor = PerformanceMonitor(collection_interval=0.5)
    monitor.start_monitoring()
    
    try:
        # 5초간 모니터링
        for i in range(10):
            # 가짜 비전 메트릭 기록
            monitor.record_vision_metrics(
                fps=30.0,
                processing_time_ms=33.3,
                detection_count=2,
                recognition_accuracy=0.95
            )
            
            time.sleep(0.5)
        
        # 통계 출력
        stats = monitor.get_current_stats()
        print("현재 통계:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    finally:
        monitor.stop_monitoring() 