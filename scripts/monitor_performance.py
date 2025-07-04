#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
성능 모니터링 자동화 스크립트 (monitor_performance.py)

파이프라인 단계별 성능을 측정하고 모니터링합니다.
"""

import os
import sys
import argparse
import json
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.metrics_file = domain_path / "performance_metrics.json"
        self.metrics = self.load_metrics()
        
        # 성능 임계값 설정
        self.thresholds = {
            'max_processing_time': 30.0,  # 30초
            'max_memory_usage_mb': 512,   # 512MB
            'max_cpu_usage': 80.0,        # 80%
            'min_success_rate': 0.95      # 95%
        }
        
        # 모니터링 상태
        self.monitoring_active = False
        self.monitor_thread = None
    
    def load_metrics(self) -> Dict:
        """성능 메트릭 로드"""
        if self.metrics_file.exists():
            try:
                return json.loads(self.metrics_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"메트릭 파일 로드 실패: {e}")
        
        return {
            "stages": {},
            "overall": {
                "total_runs": 0,
                "successful_runs": 0,
                "average_processing_time": 0.0,
                "peak_memory_usage": 0.0,
                "peak_cpu_usage": 0.0
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def save_metrics(self):
        """성능 메트릭 저장"""
        try:
            self.metrics["metadata"]["last_updated"] = datetime.now().isoformat()
            self.metrics_file.write_text(
                json.dumps(self.metrics, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
    
    def get_current_memory_usage(self) -> int:
        """현재 메모리 사용량 반환 (bytes)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception as e:
            logger.warning(f"메모리 사용량 측정 실패: {e}")
            return 0
    
    def get_current_cpu_usage(self) -> float:
        """현재 CPU 사용률 반환 (%)"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"CPU 사용률 측정 실패: {e}")
            return 0.0
    
    def measure_stage_performance(self, stage: str, func: Callable, *args, **kwargs) -> Dict:
        """단계별 성능 측정"""
        logger.info(f"{stage} 성능 측정 시작...")
        
        # 시작 전 시스템 상태
        start_time = time.time()
        start_memory = self.get_current_memory_usage()
        start_cpu = self.get_current_cpu_usage()
        
        # 메모리 모니터링 스레드 시작
        memory_samples = []
        cpu_samples = []
        self.monitoring_active = True
        
        def monitor_resources():
            """리소스 모니터링 스레드"""
            try:
                while self.monitoring_active:
                    try:
                        memory_samples.append(self.get_current_memory_usage())
                        cpu_samples.append(self.get_current_cpu_usage())
                        time.sleep(0.1)  # 100ms 간격
                    except Exception as e:
                        logger.warning(f"리소스 모니터링 오류: {e}")
                        break
            except Exception as e:
                logger.error(f"모니터링 스레드 오류: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        try:
            # 함수 실행
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"{stage} 실행 중 오류: {e}")
        finally:
            # 모니터링 중지
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2)
        
        # 종료 후 시스템 상태
        end_time = time.time()
        end_memory = self.get_current_memory_usage()
        
        # 성능 메트릭 계산
        duration = end_time - start_time
        
        # 메모리 사용량 계산 (정확한 방법)
        if memory_samples:
            peak_memory = max(memory_samples)
            memory_used = peak_memory - start_memory
        else:
            memory_used = end_memory - start_memory
        
        # CPU 사용률 계산
        peak_cpu = max(cpu_samples) if cpu_samples else start_cpu
        
        # MB 단위로 변환
        memory_used_mb = max(0, memory_used) / (1024 * 1024)
        
        # 성능 메트릭 저장
        stage_metrics = {
            "duration_seconds": duration,
            "memory_used_mb": memory_used_mb,
            "peak_cpu_percent": peak_cpu,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "memory_samples": memory_samples[:10] if memory_samples else [],  # 처음 10개 샘플만 저장
            "cpu_samples": cpu_samples[:10] if cpu_samples else []
        }
        
        # 단계별 메트릭 업데이트
        if stage not in self.metrics["stages"]:
            self.metrics["stages"][stage] = {
                "runs": [],
                "total_runs": 0,
                "successful_runs": 0,
                "average_duration": 0.0,
                "average_memory": 0.0,
                "average_cpu": 0.0
            }
        
        self.metrics["stages"][stage]["runs"].append(stage_metrics)
        self.metrics["stages"][stage]["total_runs"] += 1
        
        if success:
            self.metrics["stages"][stage]["successful_runs"] += 1
        
        # 평균값 계산
        runs = self.metrics["stages"][stage]["runs"]
        if runs:
            self.metrics["stages"][stage]["average_duration"] = sum(r["duration_seconds"] for r in runs) / len(runs)
            self.metrics["stages"][stage]["average_memory"] = sum(r["memory_used_mb"] for r in runs) / len(runs)
            self.metrics["stages"][stage]["average_cpu"] = sum(r["peak_cpu_percent"] for r in runs) / len(runs)
        
        # 전체 메트릭 업데이트
        self.metrics["overall"]["total_runs"] += 1
        if success:
            self.metrics["overall"]["successful_runs"] += 1
        
        # 피크 값 업데이트
        if memory_used_mb > self.metrics["overall"]["peak_memory_usage"]:
            self.metrics["overall"]["peak_memory_usage"] = memory_used_mb
        
        if peak_cpu > self.metrics["overall"]["peak_cpu_usage"]:
            self.metrics["overall"]["peak_cpu_usage"] = peak_cpu
        
        # 성능 검증
        performance_issues = self.check_performance_thresholds(stage, stage_metrics)
        
        # 메트릭 저장
        self.save_metrics()
        
        # 결과 출력
        logger.info(f"{stage} 성능 측정 완료:")
        logger.info(f"  - 처리 시간: {duration:.2f}초")
        logger.info(f"  - 메모리 사용: {memory_used_mb:.1f}MB")
        logger.info(f"  - 피크 CPU: {peak_cpu:.1f}%")
        logger.info(f"  - 성공 여부: {'성공' if success else '실패'}")
        
        if performance_issues:
            logger.warning(f"  - 성능 이슈: {', '.join(performance_issues)}")
        
        return {
            "result": result,
            "success": success,
            "metrics": stage_metrics,
            "issues": performance_issues
        }
    
    def check_performance_thresholds(self, stage: str, metrics: Dict) -> List[str]:
        """성능 임계값 검증"""
        issues = []
        
        if metrics["duration_seconds"] > self.thresholds["max_processing_time"]:
            issues.append(f"처리 시간 초과 ({metrics['duration_seconds']:.1f}s > {self.thresholds['max_processing_time']}s)")
        
        if metrics["memory_used_mb"] > self.thresholds["max_memory_usage_mb"]:
            issues.append(f"메모리 사용량 초과 ({metrics['memory_used_mb']:.1f}MB > {self.thresholds['max_memory_usage_mb']}MB)")
        
        if metrics["peak_cpu_percent"] > self.thresholds["max_cpu_usage"]:
            issues.append(f"CPU 사용량 초과 ({metrics['peak_cpu_percent']:.1f}% > {self.thresholds['max_cpu_usage']}%)")
        
        # 성공률 검증 추가
        if stage in self.metrics["stages"]:
            stage_data = self.metrics["stages"][stage]
            if stage_data["total_runs"] > 0:
                success_rate = stage_data["successful_runs"] / stage_data["total_runs"]
                if success_rate < self.thresholds["min_success_rate"]:
                    issues.append(f"성공률 낮음 ({success_rate:.1%} < {self.thresholds['min_success_rate']:.1%})")
        
        return issues
    
    def generate_performance_report(self) -> Dict:
        """성능 리포트 생성"""
        report = {
            "domain_path": str(self.domain_path),
            "generated_at": datetime.now().isoformat(),
            "overall_summary": self.metrics["overall"].copy(),
            "stage_summaries": {},
            "recommendations": []
        }
        
        # 단계별 요약
        for stage, stage_data in self.metrics["stages"].items():
            if stage_data["total_runs"] > 0:
                success_rate = stage_data["successful_runs"] / stage_data["total_runs"]
                report["stage_summaries"][stage] = {
                    "total_runs": stage_data["total_runs"],
                    "success_rate": success_rate,
                    "average_duration": stage_data["average_duration"],
                    "average_memory": stage_data["average_memory"],
                    "average_cpu": stage_data["average_cpu"]
                }
                
                # 권장사항 생성
                if success_rate < self.thresholds["min_success_rate"]:
                    report["recommendations"].append(f"{stage}: 성공률이 낮습니다 ({success_rate:.1%})")
                
                if stage_data["average_duration"] > self.thresholds["max_processing_time"] * 0.8:
                    report["recommendations"].append(f"{stage}: 평균 처리 시간이 임계값에 근접합니다")
                
                if stage_data["average_memory"] > self.thresholds["max_memory_usage_mb"] * 0.8:
                    report["recommendations"].append(f"{stage}: 평균 메모리 사용량이 임계값에 근접합니다")
        
        return report
    
    def print_performance_report(self, report: Dict):
        """성능 리포트 출력"""
        print(f" 성능 모니터링 리포트")
        print(f"도메인: {report['domain_path']}")
        print(f"생성 시간: {report['generated_at']}")
        print()
        
        print(f" 전체 요약:")
        overall = report["overall_summary"]
        print(f"  - 총 실행: {overall['total_runs']}회")
        print(f"  - 성공: {overall['successful_runs']}회")
        if overall['total_runs'] > 0:
            success_rate = overall['successful_runs'] / overall['total_runs']
            print(f"  - 성공률: {success_rate:.1%}")
        print(f"  - 피크 메모리: {overall['peak_memory_usage']:.1f}MB")
        print(f"  - 피크 CPU: {overall['peak_cpu_usage']:.1f}%")
        print()
        
        print(f" 단계별 성능:")
        for stage, summary in report["stage_summaries"].items():
            print(f"  {stage}:")
            print(f"    - 실행: {summary['total_runs']}회")
            print(f"    - 성공률: {summary['success_rate']:.1%}")
            print(f"    - 평균 처리시간: {summary['average_duration']:.2f}초")
            print(f"    - 평균 메모리: {summary['average_memory']:.1f}MB")
            print(f"    - 평균 CPU: {summary['average_cpu']:.1f}%")
        
        if report["recommendations"]:
            print(f" 권장사항:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")

def main():
    parser = argparse.ArgumentParser(description="성능 모니터링 스크립트")
    parser.add_argument("domain", help="도메인명")
    parser.add_argument("feature", help="기능명")
    parser.add_argument("--report", action="store_true", help="성능 리포트 생성")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로")
    
    args = parser.parse_args()
    
    domain_path = Path(args.base_path) / args.domain / args.feature
    
    if not domain_path.exists():
        logger.error(f"도메인 경로가 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    monitor = PerformanceMonitor(domain_path)
    
    if args.report:
        report = monitor.generate_performance_report()
        monitor.print_performance_report(report)
    else:
        logger.info(f"성능 모니터링 시스템이 초기화되었습니다.")
        logger.info(f"도메인: {args.domain}/{args.feature}")
        logger.info(f"메트릭 파일: {monitor.metrics_file}")

if __name__ == "__main__":
    main() 