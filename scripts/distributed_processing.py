#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
분산 처리 지원 스크립트 (distributed_processing.py)

멀티 노드 처리, 작업 분산, 병렬 처리 최적화를 제공합니다.
"""

import os
import sys
import argparse
import json
import time
import threading
import multiprocessing
import platform
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import queue
import pickle
import hashlib
import socket
import subprocess

def get_optimal_worker_config():
    """플랫폼별 최적 워커 설정"""
    system = platform.system().lower()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    if system == "windows":
        # Windows에서는 프로세스 생성 오버헤드가 높음
        max_workers = min(cpu_count, 4)
        chunk_size = 5
    elif system == "linux":
        # Linux에서는 더 많은 워커 사용 가능
        max_workers = min(cpu_count, 8)
        chunk_size = 10
    else:
        # 기타 플랫폼
        max_workers = min(cpu_count, 4)
        chunk_size = 5
    
    return {
        "max_workers": max_workers,
        "chunk_size": chunk_size,
        "timeout_seconds": 300,
        "retry_attempts": 3
    }

def get_platform_socket_config():
    """플랫폼별 소켓 설정"""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "family": socket.AF_INET,
            "type": socket.SOCK_DGRAM,
            "broadcast": False
        }
    else:
        return {
            "family": socket.AF_INET,
            "type": socket.SOCK_DGRAM,
            "broadcast": True
        }

def create_platform_process(target, *args, **kwargs):
    """플랫폼별 프로세스 생성"""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows에서는 spawn 방식 사용
        kwargs["method"] = "spawn"
    else:
        # Unix 계열에서는 fork 방식 사용
        kwargs["method"] = "fork"
    
    return multiprocessing.Process(target=target, *args, **kwargs)

class DistributedProcessor:
    """분산 처리 관리자"""
    
    def __init__(self, domain_path: Path, config: Dict = None):
        self.domain_path = domain_path
        self.config = config or self.load_default_config()
        self.workers = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_processes = []
        self.is_running = False
        
        # 로깅 설정
        self.setup_logging()
        
        # 분산 처리 상태 파일
        self.status_file = domain_path / "distributed_status.json"
        self.task_history_file = domain_path / "task_history.json"
        
        # 작업 이력 로드
        self.task_history = self.load_task_history()
    
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.domain_path / "distributed_processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_default_config(self) -> Dict:
        """기본 분산 처리 설정"""
        return {
            "max_workers": multiprocessing.cpu_count(),
            "chunk_size": 10,
            "timeout_seconds": 300,
            "retry_attempts": 3,
            "load_balancing": "round_robin",
            "node_discovery": {
                "enabled": True,
                "broadcast_port": 5000,
                "discovery_interval": 30
            },
            "task_distribution": {
                "strategy": "dynamic",
                "priority_queue": True,
                "work_stealing": True
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 10,
                "health_check_interval": 60
            }
        }
    
    def load_task_history(self) -> Dict:
        """작업 이력 로드"""
        if self.task_history_file.exists():
            try:
                return json.loads(self.task_history_file.read_text(encoding="utf-8"))
            except:
                pass
        
        return {
            "tasks": [],
            "completed": 0,
            "failed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def save_task_history(self):
        """작업 이력 저장"""
        self.task_history["metadata"]["last_updated"] = datetime.now().isoformat()
        self.task_history_file.write_text(
            json.dumps(self.task_history, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
    
    def discover_nodes(self) -> List[Dict]:
        """네트워크 노드 자동 발견"""
        if not self.config["node_discovery"]["enabled"]:
            return []
        
        discovered_nodes = []
        broadcast_port = self.config["node_discovery"]["broadcast_port"]
        
        try:
            # UDP 브로드캐스트로 노드 검색
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            
            # 브로드캐스트 메시지 전송
            broadcast_message = {
                "type": "discovery",
                "source": socket.gethostname(),
                "timestamp": datetime.now().isoformat()
            }
            
            sock.sendto(
                pickle.dumps(broadcast_message), 
                ('<broadcast>', broadcast_port)
            )
            
            # 응답 수신
            start_time = time.time()
            while time.time() - start_time < 10:  # 10초 대기
                try:
                    data, addr = sock.recvfrom(1024)
                    response = pickle.loads(data)
                    
                    if response.get("type") == "discovery_response":
                        discovered_nodes.append({
                            "host": addr[0],
                            "port": response.get("port", broadcast_port),
                            "hostname": response.get("hostname", "unknown"),
                            "cpu_count": response.get("cpu_count", 1),
                            "memory_gb": response.get("memory_gb", 1),
                            "load": response.get("load", 0.0),
                            "discovered_at": datetime.now().isoformat()
                        })
                except socket.timeout:
                    continue
            
            sock.close()
            
        except Exception as e:
            self.logger.warning(f"노드 발견 실패: {e}")
        
        self.logger.info(f"발견된 노드: {len(discovered_nodes)}개")
        return discovered_nodes
    
    def create_worker_process(self, worker_id: int, task_queue: multiprocessing.Queue, 
                            result_queue: multiprocessing.Queue) -> multiprocessing.Process:
        """워커 프로세스 생성"""
        def worker_function(worker_id, task_queue, result_queue):
            """워커 함수"""
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(f"Worker-{worker_id}")
            
            logger.info(f"워커 {worker_id} 시작")
            
            while True:
                try:
                    # 작업 가져오기
                    task = task_queue.get(timeout=1)
                    
                    if task is None:  # 종료 신호
                        break
                    
                    logger.info(f"작업 처리 시작: {task['task_id']}")
                    
                    # 작업 실행
                    start_time = time.time()
                    result = self.execute_task(task)
                    processing_time = time.time() - start_time
                    
                    # 결과 반환
                    result.update({
                        "worker_id": worker_id,
                        "processing_time": processing_time,
                        "completed_at": datetime.now().isoformat()
                    })
                    
                    result_queue.put(result)
                    logger.info(f"작업 완료: {task['task_id']} ({processing_time:.2f}초)")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"작업 처리 오류: {e}")
                    result_queue.put({
                        "task_id": task.get("task_id", "unknown"),
                        "status": "failed",
                        "error": str(e),
                        "worker_id": worker_id,
                        "completed_at": datetime.now().isoformat()
                    })
            
            logger.info(f"워커 {worker_id} 종료")
        
        return multiprocessing.Process(
            target=worker_function,
            args=(worker_id, task_queue, result_queue)
        )
    
    def execute_task(self, task: Dict) -> Dict:
        """작업 실행 (시뮬레이션)"""
        task_id = task["task_id"]
        task_type = task["type"]
        data = task.get("data", {})
        
        # 시뮬레이션된 작업 처리
        if task_type == "file_processing":
            # 파일 처리 시뮬레이션
            time.sleep(0.1)  # 100ms 처리 시간
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {
                    "processed_files": len(data.get("files", [])),
                    "output_path": f"processed_{task_id}.json"
                }
            }
        
        elif task_type == "feature_extraction":
            # 특성 추출 시뮬레이션
            time.sleep(0.2)  # 200ms 처리 시간
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {
                    "features_extracted": data.get("count", 0),
                    "feature_vector": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
        
        elif task_type == "clustering":
            # 클러스터링 시뮬레이션
            time.sleep(0.3)  # 300ms 처리 시간
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {
                    "clusters_found": data.get("expected_clusters", 3),
                    "cluster_labels": [0, 1, 2, 0, 1]
                }
            }
        
        else:
            # 기본 작업
            time.sleep(0.1)
            return {
                "task_id": task_id,
                "status": "completed",
                "result": {"message": "Task completed successfully"}
            }
    
    def distribute_tasks(self, tasks: List[Dict]) -> Dict:
        """작업 분산 처리"""
        self.logger.info(f"작업 분산 처리 시작: {len(tasks)}개 작업")
        
        # 멀티프로세싱 큐 생성
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        
        # 작업을 큐에 추가
        for task in tasks:
            task_queue.put(task)
        
        # 종료 신호 추가
        for _ in range(self.config["max_workers"]):
            task_queue.put(None)
        
        # 워커 프로세스 시작
        worker_processes = []
        for i in range(self.config["max_workers"]):
            worker = self.create_worker_process(i, task_queue, result_queue)
            worker.start()
            worker_processes.append(worker)
        
        # 결과 수집
        results = []
        completed_count = 0
        failed_count = 0
        
        start_time = time.time()
        
        while completed_count + failed_count < len(tasks):
            try:
                result = result_queue.get(timeout=1)
                results.append(result)
                
                if result["status"] == "completed":
                    completed_count += 1
                else:
                    failed_count += 1
                
                # 진행 상황 로깅
                if (completed_count + failed_count) % 10 == 0:
                    progress = (completed_count + failed_count) / len(tasks) * 100
                    self.logger.info(f"진행률: {progress:.1f}% ({completed_count + failed_count}/{len(tasks)})")
                
            except queue.Empty:
                continue
        
        total_time = time.time() - start_time
        
        # 워커 프로세스 종료
        for worker in worker_processes:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        # 결과 요약
        summary = {
            "total_tasks": len(tasks),
            "completed": completed_count,
            "failed": failed_count,
            "success_rate": completed_count / len(tasks) * 100,
            "total_time": total_time,
            "average_time_per_task": total_time / len(tasks),
            "results": results
        }
        
        # 작업 이력 업데이트
        self.task_history["tasks"].extend(results)
        self.task_history["completed"] += completed_count
        self.task_history["failed"] += failed_count
        self.task_history["total_processing_time"] += total_time
        self.task_history["average_processing_time"] = (
            self.task_history["total_processing_time"] / 
            (self.task_history["completed"] + self.task_history["failed"])
        )
        
        self.save_task_history()
        
        self.logger.info(f"작업 분산 처리 완료: {completed_count}개 성공, {failed_count}개 실패")
        return summary
    
    def create_pipeline_tasks(self, stage: str, files: List[Path]) -> List[Dict]:
        """파이프라인 단계별 작업 생성"""
        tasks = []
        
        # 파일을 청크로 분할
        chunk_size = self.config["chunk_size"]
        file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        
        for chunk_idx, file_chunk in enumerate(file_chunks):
            task_id = f"{stage}_{chunk_idx}_{int(time.time())}"
            
            task = {
                "task_id": task_id,
                "type": self.get_task_type_for_stage(stage),
                "stage": stage,
                "data": {
                    "files": [str(f) for f in file_chunk],
                    "chunk_index": chunk_idx,
                    "total_chunks": len(file_chunks)
                },
                "priority": self.get_priority_for_stage(stage),
                "created_at": datetime.now().isoformat()
            }
            
            tasks.append(task)
        
        return tasks
    
    def get_task_type_for_stage(self, stage: str) -> str:
        """단계별 작업 타입 반환"""
        task_types = {
            "1_raw": "file_processing",
            "2_extracted": "feature_extraction",
            "3_clustered": "clustering",
            "4_labeled": "classification",
            "5_embeddings": "embedding_generation"
        }
        return task_types.get(stage, "file_processing")
    
    def get_priority_for_stage(self, stage: str) -> int:
        """단계별 우선순위 반환"""
        priorities = {
            "1_raw": 1,
            "2_extracted": 2,
            "3_clustered": 3,
            "4_labeled": 4,
            "5_embeddings": 5
        }
        return priorities.get(stage, 1)
    
    def run_distributed_pipeline(self, stages: List[str] = None) -> Dict:
        """분산 파이프라인 실행"""
        if stages is None:
            stages = ["1_raw", "2_extracted", "3_clustered", "4_labeled", "5_embeddings"]
        
        self.logger.info(f"분산 파이프라인 실행 시작: {stages}")
        
        # 노드 발견
        discovered_nodes = self.discover_nodes()
        
        # 전체 결과
        pipeline_results = {}
        
        for stage in stages:
            self.logger.info(f"단계 {stage} 처리 시작")
            
            # 단계별 파일 목록 생성
            stage_path = self.domain_path / stage
            if not stage_path.exists():
                self.logger.warning(f"단계 경로가 존재하지 않습니다: {stage_path}")
                continue
            
            # 파일 목록 수집
            files = []
            for file_path in stage_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    files.append(file_path)
            
            if not files:
                self.logger.info(f"단계 {stage}에 처리할 파일이 없습니다")
                continue
            
            # 작업 생성
            tasks = self.create_pipeline_tasks(stage, files)
            
            # 분산 처리 실행
            stage_result = self.distribute_tasks(tasks)
            pipeline_results[stage] = stage_result
            
            self.logger.info(f"단계 {stage} 완료: {stage_result['completed']}개 성공")
        
        # 전체 요약
        total_summary = {
            "pipeline_stages": list(pipeline_results.keys()),
            "total_tasks": sum(r["total_tasks"] for r in pipeline_results.values()),
            "total_completed": sum(r["completed"] for r in pipeline_results.values()),
            "total_failed": sum(r["failed"] for r in pipeline_results.values()),
            "total_time": sum(r["total_time"] for r in pipeline_results.values()),
            "overall_success_rate": sum(r["completed"] for r in pipeline_results.values()) / 
                                  sum(r["total_tasks"] for r in pipeline_results.values()) * 100,
            "discovered_nodes": len(discovered_nodes),
            "stage_results": pipeline_results
        }
        
        self.logger.info(f"분산 파이프라인 완료: 전체 성공률 {total_summary['overall_success_rate']:.1f}%")
        return total_summary
    
    def generate_distributed_report(self) -> Dict:
        """분산 처리 리포트 생성"""
        report = {
            "domain_path": str(self.domain_path),
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": multiprocessing.cpu_count(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname()
            },
            "configuration": self.config,
            "task_history": {
                "total_tasks": self.task_history["completed"] + self.task_history["failed"],
                "completed": self.task_history["completed"],
                "failed": self.task_history["failed"],
                "success_rate": (self.task_history["completed"] / 
                               (self.task_history["completed"] + self.task_history["failed"]) * 100) 
                               if (self.task_history["completed"] + self.task_history["failed"]) > 0 else 0,
                "average_processing_time": self.task_history["average_processing_time"]
            },
            "performance_metrics": {
                "throughput_tasks_per_second": (self.task_history["completed"] / 
                                              self.task_history["total_processing_time"]) 
                                              if self.task_history["total_processing_time"] > 0 else 0,
                "efficiency_score": self.calculate_efficiency_score()
            }
        }
        
        return report
    
    def calculate_efficiency_score(self) -> float:
        """효율성 점수 계산"""
        if not self.task_history["tasks"]:
            return 0.0
        
        # 성공률 가중치
        success_rate = self.task_history["completed"] / (self.task_history["completed"] + self.task_history["failed"])
        
        # 처리 시간 효율성 (짧을수록 좋음)
        avg_time = self.task_history["average_processing_time"]
        time_efficiency = max(0, 1 - (avg_time / 10))  # 10초 이상이면 효율성 감소
        
        # 병렬화 효율성
        parallel_efficiency = min(1.0, self.config["max_workers"] / multiprocessing.cpu_count())
        
        # 종합 점수
        efficiency_score = (success_rate * 0.4 + time_efficiency * 0.3 + parallel_efficiency * 0.3) * 100
        
        return round(efficiency_score, 2)
    
    def print_distributed_report(self, report: Dict):
        """분산 처리 리포트 출력"""
        print(f" 분산 처리 리포트")
        print(f"도메인: {report['domain_path']}")
        print(f"생성 시간: {report['generated_at']}")
        print()
        
        sys_info = report["system_info"]
        print(f" 시스템 정보:")
        print(f"  - CPU 코어: {sys_info['cpu_count']}개")
        print(f"  - 플랫폼: {sys_info['platform']}")
        print(f"  - Python: {sys_info['python_version']}")
        print(f"  - 호스트: {sys_info['hostname']}")
        
        task_history = report["task_history"]
        print(f" 작업 이력:")
        print(f"  - 총 작업: {task_history['total_tasks']}개")
        print(f"  - 성공: {task_history['completed']}개")
        print(f"  - 실패: {task_history['failed']}개")
        print(f"  - 성공률: {task_history['success_rate']:.1f}%")
        print(f"  - 평균 처리 시간: {task_history['average_processing_time']:.2f}초")
        
        perf_metrics = report["performance_metrics"]
        print(f" 성능 메트릭:")
        print(f"  - 처리량: {perf_metrics['throughput_tasks_per_second']:.2f} 작업/초")
        print(f"  - 효율성 점수: {perf_metrics['efficiency_score']:.1f}/100")

def main():
    parser = argparse.ArgumentParser(description="분산 처리 시스템")
    parser.add_argument("domain", help="도메인명")
    parser.add_argument("feature", help="기능명")
    parser.add_argument("--run-pipeline", action="store_true", help="분산 파이프라인 실행")
    parser.add_argument("--stages", nargs="+", help="실행할 단계들")
    parser.add_argument("--report", action="store_true", help="분산 처리 리포트 생성")
    parser.add_argument("--discover-nodes", action="store_true", help="네트워크 노드 발견")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로")
    
    args = parser.parse_args()
    
    domain_path = Path(args.base_path) / args.domain / args.feature
    
    if not domain_path.exists():
        print(f" 도메인 경로가 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    processor = DistributedProcessor(domain_path)
    
    if args.discover_nodes:
        print("네트워크 노드 발견 중...")
        nodes = processor.discover_nodes()
        print(f"발견된 노드: {len(nodes)}개")
        for node in nodes:
            print(f"  - {node['hostname']} ({node['host']}:{node['port']})")
    
    elif args.run_pipeline:
        print("분산 파이프라인 실행 중...")
        stages = args.stages if args.stages else None
        result = processor.run_distributed_pipeline(stages)
        print(f"파이프라인 완료: {result['total_completed']}개 성공, {result['total_failed']}개 실패")
    
    elif args.report:
        report = processor.generate_distributed_report()
        processor.print_distributed_report(report)
    
    else:
        print(f"분산 처리 시스템이 초기화되었습니다.")
        print(f"도메인: {args.domain}/{args.feature}")
        print(f"최대 워커: {processor.config['max_workers']}개")
        print(f"청크 크기: {processor.config['chunk_size']}개")

if __name__ == "__main__":
    main()
