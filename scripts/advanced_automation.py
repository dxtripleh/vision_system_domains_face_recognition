#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
고급 자동화 통합 관리 스크립트 (advanced_automation.py)

지속적 학습, 웹 대시보드, 분산 처리를 통합 관리합니다.
"""

import os
import sys
import argparse
import json
import time
import threading
import subprocess
import platform
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import signal
import socket

def get_optimal_process_config():
    """플랫폼별 최적 프로세스 설정"""
    system = platform.system().lower()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    if system == "windows":
        return {
            "max_processes": min(cpu_count, 4),
            "startup_timeout": 30,
            "kill_timeout": 10,
            "use_shell": True
        }
    elif system == "linux":
        return {
            "max_processes": min(cpu_count, 8),
            "startup_timeout": 60,
            "kill_timeout": 15,
            "use_shell": False
        }
    else:
        return {
            "max_processes": min(cpu_count, 4),
            "startup_timeout": 30,
            "kill_timeout": 10,
            "use_shell": True
        }

def get_platform_signal_config():
    """플랫폼별 시그널 설정"""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "terminate_signal": signal.CTRL_C_EVENT,
            "kill_signal": signal.CTRL_BREAK_EVENT
        }
    else:
        return {
            "terminate_signal": signal.SIGTERM,
            "kill_signal": signal.SIGKILL
        }

def create_platform_subprocess(cmd, **kwargs):
    """플랫폼별 서브프로세스 생성"""
    system = platform.system().lower()
    
    if system == "windows":
        kwargs["shell"] = True
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["shell"] = False
        kwargs["preexec_fn"] = os.setsid
    
    return subprocess.Popen(cmd, **kwargs)

def kill_platform_process(pid: int):
    """플랫폼별 프로세스 종료"""
    system = platform.system().lower()
    
    try:
        if system == "windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                         capture_output=True, check=True)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception as e:
        logging.warning(f"프로세스 종료 실패 (PID: {pid}): {e}")

class AdvancedAutomationManager:
    """고급 자동화 통합 관리자"""
    
    def __init__(self, domain_path: Path):
        self.domain_path = domain_path
        self.processes = {}
        self.is_running = False
        
        # 로깅 설정
        self.setup_logging()
        
        # 설정 파일
        self.config_file = domain_path / "advanced_automation_config.json"
        self.status_file = domain_path / "automation_status.json"
        
        # 설정 로드
        self.config = self.load_config()
        
        # 상태 초기화
        self.status = {
            "continuous_learning": {"status": "stopped", "pid": None, "started_at": None},
            "web_dashboard": {"status": "stopped", "pid": None, "started_at": None, "port": 8080},
            "distributed_processing": {"status": "stopped", "pid": None, "started_at": None},
            "monitoring": {"status": "stopped", "pid": None, "started_at": None},
            "last_updated": datetime.now().isoformat()
        }
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.domain_path / "advanced_automation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict:
        """설정 로드"""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text(encoding="utf-8"))
            except:
                pass
        
        # 기본 설정
        default_config = {
            "continuous_learning": {
                "enabled": True,
                "interval_hours": 24,
                "auto_retrain": True,
                "performance_threshold": 0.85
            },
            "web_dashboard": {
                "enabled": True,
                "port": 8080,
                "auto_start": True,
                "refresh_interval": 30
            },
            "distributed_processing": {
                "enabled": True,
                "max_workers": 4,
                "auto_scale": True,
                "load_threshold": 0.8
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 10,
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "disk_usage": 90
                }
            },
            "scheduling": {
                "pipeline_auto_run": True,
                "backup_interval_hours": 12,
                "cleanup_interval_hours": 24
            }
        }
        
        # 설정 저장
        self.config_file.write_text(
            json.dumps(default_config, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
        
        return default_config
    
    def save_status(self):
        """상태 저장"""
        self.status["last_updated"] = datetime.now().isoformat()
        self.status_file.write_text(
            json.dumps(self.status, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 종료 중...")
        self.stop_all_services()
        sys.exit(0)
    
    def start_continuous_learning(self) -> bool:
        """지속적 학습 서비스 시작"""
        if not self.config["continuous_learning"]["enabled"]:
            return False
        
        try:
            # 기존 프로세스 종료
            if self.status["continuous_learning"]["pid"]:
                self.stop_process(self.status["continuous_learning"]["pid"])
            
            # 새 프로세스 시작
            cmd = [
                sys.executable, "scripts/continuous_learning.py",
                str(self.domain_path.parent.name),  # domain
                str(self.domain_path.name),         # feature
                "--daemon"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            self.status["continuous_learning"].update({
                "status": "running",
                "pid": process.pid,
                "started_at": datetime.now().isoformat()
            })
            
            self.processes["continuous_learning"] = process
            self.save_status()
            
            self.logger.info(f"지속적 학습 서비스 시작 (PID: {process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"지속적 학습 서비스 시작 실패: {e}")
            return False
    
    def start_web_dashboard(self) -> bool:
        """웹 대시보드 서비스 시작"""
        if not self.config["web_dashboard"]["enabled"]:
            return False
        
        try:
            # 기존 프로세스 종료
            if self.status["web_dashboard"]["pid"]:
                self.stop_process(self.status["web_dashboard"]["pid"])
            
            # 포트 확인
            port = self.config["web_dashboard"]["port"]
            
            # 새 프로세스 시작
            cmd = [
                sys.executable, "scripts/web_dashboard.py",
                str(self.domain_path.parent.name),  # domain
                str(self.domain_path.name),         # feature
                "--port", str(port)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            self.status["web_dashboard"].update({
                "status": "running",
                "pid": process.pid,
                "started_at": datetime.now().isoformat(),
                "port": port
            })
            
            self.processes["web_dashboard"] = process
            self.save_status()
            
            self.logger.info(f"웹 대시보드 서비스 시작 (PID: {process.pid}, Port: {port})")
            return True
            
        except Exception as e:
            self.logger.error(f"웹 대시보드 서비스 시작 실패: {e}")
            return False
    
    def start_distributed_processing(self) -> bool:
        """분산 처리 서비스 시작"""
        if not self.config["distributed_processing"]["enabled"]:
            return False
        
        try:
            # 기존 프로세스 종료
            if self.status["distributed_processing"]["pid"]:
                self.stop_process(self.status["distributed_processing"]["pid"])
            
            # 새 프로세스 시작
            cmd = [
                sys.executable, "scripts/distributed_processing.py",
                str(self.domain_path.parent.name),  # domain
                str(self.domain_path.name),         # feature
                "--daemon"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            self.status["distributed_processing"].update({
                "status": "running",
                "pid": process.pid,
                "started_at": datetime.now().isoformat()
            })
            
            self.processes["distributed_processing"] = process
            self.save_status()
            
            self.logger.info(f"분산 처리 서비스 시작 (PID: {process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"분산 처리 서비스 시작 실패: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """모니터링 서비스 시작"""
        if not self.config["monitoring"]["enabled"]:
            return False
        
        try:
            # 기존 프로세스 종료
            if self.status["monitoring"]["pid"]:
                self.stop_process(self.status["monitoring"]["pid"])
            
            # 새 프로세스 시작
            cmd = [
                sys.executable, "scripts/monitor_performance.py",
                str(self.domain_path.parent.name),  # domain
                str(self.domain_path.name),         # feature
                "--daemon"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            self.status["monitoring"].update({
                "status": "running",
                "pid": process.pid,
                "started_at": datetime.now().isoformat()
            })
            
            self.processes["monitoring"] = process
            self.save_status()
            
            self.logger.info(f"모니터링 서비스 시작 (PID: {process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"모니터링 서비스 시작 실패: {e}")
            return False
    
    def stop_process(self, pid: int):
        """프로세스 종료"""
        try:
            if pid and psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)
        except:
            pass
    
    def stop_service(self, service_name: str):
        """서비스 종료"""
        if service_name in self.processes:
            process = self.processes[service_name]
            if process.poll() is None:  # 프로세스가 실행 중
                process.terminate()
                process.wait(timeout=5)
            
            self.status[service_name].update({
                "status": "stopped",
                "pid": None,
                "started_at": None
            })
            
            del self.processes[service_name]
            self.save_status()
            
            self.logger.info(f"{service_name} 서비스 종료")
    
    def stop_all_services(self):
        """모든 서비스 종료"""
        self.logger.info("모든 서비스 종료 중...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        self.is_running = False
        self.logger.info("모든 서비스가 종료되었습니다.")
    
    def start_all_services(self):
        """모든 서비스 시작"""
        self.logger.info("고급 자동화 서비스 시작 중...")
        
        services_started = 0
        
        # 지속적 학습 시작
        if self.start_continuous_learning():
            services_started += 1
        
        # 웹 대시보드 시작
        if self.start_web_dashboard():
            services_started += 1
        
        # 분산 처리 시작
        if self.start_distributed_processing():
            services_started += 1
        
        # 모니터링 시작
        if self.start_monitoring():
            services_started += 1
        
        self.is_running = True
        self.logger.info(f"{services_started}개 서비스가 시작되었습니다.")
        
        return services_started
    
    def check_service_health(self):
        """서비스 상태 확인"""
        for service_name, process in self.processes.items():
            if process.poll() is not None:  # 프로세스가 종료됨
                self.logger.warning(f"{service_name} 서비스가 비정상 종료됨")
                self.status[service_name]["status"] = "crashed"
                self.save_status()
                
                # 자동 재시작
                if self.config.get("auto_restart", True):
                    self.logger.info(f"{service_name} 서비스 재시작 중...")
                    if service_name == "continuous_learning":
                        self.start_continuous_learning()
                    elif service_name == "web_dashboard":
                        self.start_web_dashboard()
                    elif service_name == "distributed_processing":
                        self.start_distributed_processing()
                    elif service_name == "monitoring":
                        self.start_monitoring()
    
    def run_scheduled_tasks(self):
        """예약된 작업 실행"""
        try:
            # 파이프라인 자동 실행
            if self.config["scheduling"]["pipeline_auto_run"]:
                self.run_pipeline_if_needed()
            
            # 백업 실행
            if self.config["scheduling"]["backup_interval_hours"] > 0:
                self.run_backup_if_needed()
            
            # 정리 작업 실행
            if self.config["scheduling"]["cleanup_interval_hours"] > 0:
                self.run_cleanup_if_needed()
                
        except Exception as e:
            self.logger.error(f"예약 작업 실행 오류: {e}")
    
    def run_pipeline_if_needed(self):
        """필요시 파이프라인 실행"""
        # 마지막 실행 시간 확인
        pipeline_status_file = self.domain_path / "pipeline_progress.json"
        if not pipeline_status_file.exists():
            return
        
        try:
            progress_data = json.loads(pipeline_status_file.read_text(encoding="utf-8"))
            last_updated = progress_data.get("metadata", {}).get("last_updated")
            
            if last_updated:
                last_time = datetime.fromisoformat(last_updated)
                hours_since_last_run = (datetime.now() - last_time).total_seconds() / 3600
                
                # 24시간 이상 지났으면 재실행
                if hours_since_last_run > 24:
                    self.logger.info("파이프라인 자동 실행 시작")
                    subprocess.run([
                        sys.executable, "scripts/run_pipeline.py",
                        str(self.domain_path.parent.name),
                        str(self.domain_path.name)
                    ], cwd=Path.cwd())
                    
        except Exception as e:
            self.logger.error(f"파이프라인 자동 실행 오류: {e}")
    
    def run_backup_if_needed(self):
        """필요시 백업 실행"""
        backup_interval = self.config["scheduling"]["backup_interval_hours"]
        backup_file = self.domain_path / "last_backup.txt"
        
        should_backup = True
        if backup_file.exists():
            try:
                last_backup = datetime.fromisoformat(backup_file.read_text().strip())
                hours_since_backup = (datetime.now() - last_backup).total_seconds() / 3600
                should_backup = hours_since_backup >= backup_interval
            except:
                pass
        
        if should_backup:
            self.logger.info("자동 백업 실행")
            self.create_backup()
            backup_file.write_text(datetime.now().isoformat(), encoding="utf-8")
    
    def create_backup(self):
        """백업 생성"""
        try:
            backup_dir = self.domain_path / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 중요 파일들 백업
            important_files = [
                "pipeline_progress.json",
                "performance_metrics.json",
                "traceability_data.json"
            ]
            
            for file_name in important_files:
                source_file = self.domain_path / file_name
                if source_file.exists():
                    import shutil
                    shutil.copy2(source_file, backup_dir / file_name)
            
            self.logger.info(f"백업 생성 완료: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패: {e}")
    
    def run_cleanup_if_needed(self):
        """필요시 정리 작업 실행"""
        cleanup_interval = self.config["scheduling"]["cleanup_interval_hours"]
        cleanup_file = self.domain_path / "last_cleanup.txt"
        
        should_cleanup = True
        if cleanup_file.exists():
            try:
                last_cleanup = datetime.fromisoformat(cleanup_file.read_text().strip())
                hours_since_cleanup = (datetime.now() - last_cleanup).total_seconds() / 3600
                should_cleanup = hours_since_cleanup >= cleanup_interval
            except:
                pass
        
        if should_cleanup:
            self.logger.info("자동 정리 작업 실행")
            self.cleanup_old_files()
            cleanup_file.write_text(datetime.now().isoformat(), encoding="utf-8")
    
    def cleanup_old_files(self):
        """오래된 파일 정리"""
        try:
            # 7일 이상 된 로그 파일 삭제
            log_files = list(self.domain_path.glob("*.log"))
            cutoff_time = datetime.now() - timedelta(days=7)
            
            for log_file in log_files:
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    self.logger.info(f"오래된 로그 파일 삭제: {log_file.name}")
            
            # 30일 이상 된 백업 삭제
            backup_dir = self.domain_path / "backups"
            if backup_dir.exists():
                backup_dirs = [d for d in backup_dir.iterdir() if d.is_dir()]
                cutoff_time = datetime.now() - timedelta(days=30)
                
                for backup in backup_dirs:
                    if backup.stat().st_mtime < cutoff_time.timestamp():
                        import shutil
                        shutil.rmtree(backup)
                        self.logger.info(f"오래된 백업 삭제: {backup.name}")
            
        except Exception as e:
            self.logger.error(f"정리 작업 실패: {e}")
    
    def monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 서비스 상태 확인
                self.check_service_health()
                
                # 예약된 작업 실행
                self.run_scheduled_tasks()
                
                # 상태 저장
                self.save_status()
                
                time.sleep(60)  # 1분마다 확인
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(60)
    
    def print_status(self):
        """상태 출력"""
        print(f" 고급 자동화 시스템 상태")
        print(f"도메인: {self.domain_path}")
        print(f"마지막 업데이트: {self.status['last_updated']}")
        print()
        
        for service_name, service_status in self.status.items():
            if service_name == "last_updated":
                continue
            
            status_icon = "🟢" if service_status["status"] == "running" else "🔴"
            print(f"{status_icon} {service_name}: {service_status['status']}")
            
            if service_status["pid"]:
                print(f"   PID: {service_status['pid']}")
            
            if service_status["started_at"]:
                print(f"   시작 시간: {service_status['started_at']}")
            
            if service_name == "web_dashboard" and service_status["port"]:
                print(f"   포트: {service_status['port']}")
            
            print()
    
    def run(self):
        """메인 실행 루프"""
        try:
            # 모든 서비스 시작
            services_started = self.start_all_services()
            
            if services_started == 0:
                self.logger.warning("시작된 서비스가 없습니다.")
                return
            
            print(f"🚀 고급 자동화 시스템이 시작되었습니다!")
            print(f"   시작된 서비스: {services_started}개")
            print(f"   도메인: {self.domain_path}")
            print(f"   웹 대시보드: http://localhost:{self.config['web_dashboard']['port']}")
            print(f"   중지하려면 Ctrl+C를 누르세요.")
            print()
            
            # 모니터링 루프 시작
            self.monitor_loop()
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"실행 오류: {e}")
        finally:
            self.stop_all_services()

def main():
    parser = argparse.ArgumentParser(description="고급 자동화 통합 관리")
    parser.add_argument("domain", help="도메인명")
    parser.add_argument("feature", help="기능명")
    parser.add_argument("--start", action="store_true", help="모든 서비스 시작")
    parser.add_argument("--stop", action="store_true", help="모든 서비스 중지")
    parser.add_argument("--status", action="store_true", help="서비스 상태 확인")
    parser.add_argument("--config", action="store_true", help="설정 파일 편집")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로")
    
    args = parser.parse_args()
    
    domain_path = Path(args.base_path) / args.domain / args.feature
    
    if not domain_path.exists():
        print(f" 도메인 경로가 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    manager = AdvancedAutomationManager(domain_path)
    
    if args.stop:
        print("모든 서비스 중지 중...")
        manager.stop_all_services()
        print("모든 서비스가 중지되었습니다.")
    
    elif args.status:
        manager.print_status()
    
    elif args.config:
        print(f"설정 파일: {manager.config_file}")
        print("설정을 수정하려면 파일을 직접 편집하세요.")
    
    elif args.start:
        manager.run()
    
    else:
        print(f"고급 자동화 시스템이 초기화되었습니다.")
        print(f"도메인: {args.domain}/{args.feature}")
        print()
        print("사용 가능한 명령:")
        print("  --start   : 모든 서비스 시작")
        print("  --stop    : 모든 서비스 중지")
        print("  --status  : 서비스 상태 확인")
        print("  --config  : 설정 파일 정보")

if __name__ == "__main__":
    main()
