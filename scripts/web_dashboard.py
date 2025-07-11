#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
웹 대시보드 스크립트 (web_dashboard.py)

실시간 모니터링 UI, 성능 차트, 알림 시스템을 제공합니다.
"""

import os
import sys
import argparse
import json
import time
import threading
import platform
import socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

def get_optimal_network_config():
    """플랫폼별 최적 네트워크 설정"""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "host": "localhost",
            "port": 8080,
            "backlog": 5,
            "timeout": 30
        }
    elif system == "linux":
        return {
            "host": "0.0.0.0",  # 모든 인터페이스에서 접근 허용
            "port": 8080,
            "backlog": 10,
            "timeout": 60
        }
    else:
        return {
            "host": "localhost",
            "port": 8080,
            "backlog": 5,
            "timeout": 30
        }

def is_port_available(port: int) -> bool:
    """포트 사용 가능 여부 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """사용 가능한 포트 찾기"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    raise RuntimeError(f"사용 가능한 포트를 찾을 수 없습니다 (시작: {start_port})")

class DashboardHandler(BaseHTTPRequestHandler):
    """대시보드 HTTP 핸들러"""
    
    def __init__(self, *args, dashboard_data=None, **kwargs):
        self.dashboard_data = dashboard_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """GET 요청 처리"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == "/" or path == "/index.html":
            self.send_dashboard_page()
        elif path == "/api/status":
            self.send_json_response(self.get_system_status())
        elif path == "/api/performance":
            self.send_json_response(self.get_performance_data())
        elif path == "/api/progress":
            self.send_json_response(self.get_progress_data())
        elif path == "/api/alerts":
            self.send_json_response(self.get_alerts())
        elif path.startswith("/static/"):
            self.send_static_file(path)
        else:
            self.send_error(404, "Not Found")
    
    def send_dashboard_page(self):
        """대시보드 HTML 페이지 전송"""
        html_content = self.generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def generate_dashboard_html(self) -> str:
        """대시보드 HTML 생성"""
        return f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>비전 시스템 대시보드</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }}
        .card h3 {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.2em;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: 500;
            color: #666;
        }}
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-active {{
            background: #28a745;
        }}
        .status-inactive {{
            background: #dc3545;
        }}
        .status-warning {{
            background: #ffc107;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 15px;
        }}
        .alert {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}
        .alert-danger {{
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}
        .refresh-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .refresh-btn:hover {{
            background: #0056b3;
        }}
        .auto-refresh {{
            text-align: center;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 비전 시스템 대시보드</h1>
            <p>실시간 모니터링 및 성능 추적</p>
        </div>
        
        <div class="auto-refresh">
            <button class="refresh-btn" onclick="refreshData()">새로고침</button>
            <label><input type="checkbox" id="autoRefresh" checked> 자동 새로고침 (30초)</label>
        </div>
        
        <div class="dashboard-grid">
            <!-- 시스템 상태 -->
            <div class="card">
                <h3>📊 시스템 상태</h3>
                <div id="systemStatus">
                    <div class="metric">
                        <span class="metric-label">파이프라인 상태</span>
                        <span class="metric-value" id="pipelineStatus">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">현재 단계</span>
                        <span class="metric-value" id="currentStage">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">진행률</span>
                        <span class="metric-value" id="progressPercent">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">처리된 파일</span>
                        <span class="metric-value" id="processedFiles">로딩 중...</span>
                    </div>
                </div>
            </div>
            
            <!-- 성능 메트릭 -->
            <div class="card">
                <h3>⚡ 성능 메트릭</h3>
                <div id="performanceMetrics">
                    <div class="metric">
                        <span class="metric-label">CPU 사용률</span>
                        <span class="metric-value" id="cpuUsage">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">메모리 사용률</span>
                        <span class="metric-value" id="memoryUsage">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">처리 시간</span>
                        <span class="metric-value" id="processingTime">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">성공률</span>
                        <span class="metric-value" id="successRate">로딩 중...</span>
                    </div>
                </div>
            </div>
            
            <!-- 성능 차트 -->
            <div class="card">
                <h3>📈 성능 추세</h3>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
            
            <!-- 진행 상황 차트 -->
            <div class="card">
                <h3>🎯 진행 상황</h3>
                <div class="chart-container">
                    <canvas id="progressChart"></canvas>
                </div>
            </div>
            
            <!-- 알림 -->
            <div class="card">
                <h3>🔔 알림</h3>
                <div id="alerts">
                    <div class="alert alert-info">
                        시스템이 정상적으로 작동 중입니다.
                    </div>
                </div>
            </div>
            
            <!-- 최근 활동 -->
            <div class="card">
                <h3>📝 최근 활동</h3>
                <div id="recentActivity">
                    <div class="metric">
                        <span class="metric-label">마지막 업데이트</span>
                        <span class="metric-value" id="lastUpdate">로딩 중...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">다음 예정</span>
                        <span class="metric-value" id="nextScheduled">로딩 중...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let performanceChart, progressChart;
        let autoRefreshInterval;
        
        // 차트 초기화
        function initCharts() {{
            const ctx1 = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx1, {{
                type: 'line',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'F1 점수',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }}
                }}
            }});
            
            const ctx2 = document.getElementById('progressChart').getContext('2d');
            progressChart = new Chart(ctx2, {{
                type: 'doughnut',
                data: {{
                    labels: ['완료', '진행 중', '대기'],
                    datasets: [{{
                        data: [0, 0, 9],
                        backgroundColor: ['#28a745', '#ffc107', '#6c757d']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});
        }}
        
        // 데이터 새로고침
        async function refreshData() {{
            try {{
                // 시스템 상태 업데이트
                const statusResponse = await fetch('/api/status');
                const statusData = await statusResponse.json();
                updateSystemStatus(statusData);
                
                // 성능 데이터 업데이트
                const perfResponse = await fetch('/api/performance');
                const perfData = await perfResponse.json();
                updatePerformanceMetrics(perfData);
                
                // 진행 상황 업데이트
                const progressResponse = await fetch('/api/progress');
                const progressData = await progressResponse.json();
                updateProgressData(progressData);
                
                // 알림 업데이트
                const alertsResponse = await fetch('/api/alerts');
                const alertsData = await alertsResponse.json();
                updateAlerts(alertsData);
                
            }} catch (error) {{
                console.error('데이터 새로고침 오류:', error);
            }}
        }}
        
        // 시스템 상태 업데이트
        function updateSystemStatus(data) {{
            document.getElementById('pipelineStatus').innerHTML = 
                `<span class="status-indicator status-${{data.status}}"></span>${{data.status_text}}`;
            document.getElementById('currentStage').textContent = data.current_stage || '대기 중';
            document.getElementById('progressPercent').textContent = `${{data.progress_percent}}%`;
            document.getElementById('processedFiles').textContent = data.processed_files;
            document.getElementById('lastUpdate').textContent = data.last_update;
        }}
        
        // 성능 메트릭 업데이트
        function updatePerformanceMetrics(data) {{
            document.getElementById('cpuUsage').textContent = `${{data.cpu_usage}}%`;
            document.getElementById('memoryUsage').textContent = `${{data.memory_usage}}%`;
            document.getElementById('processingTime').textContent = `${{data.processing_time}}초`;
            document.getElementById('successRate').textContent = `${{data.success_rate}}%`;
            
            // 차트 업데이트
            if (performanceChart && data.history) {{
                performanceChart.data.labels = data.history.labels;
                performanceChart.data.datasets[0].data = data.history.f1_scores;
                performanceChart.update();
            }}
        }}
        
        // 진행 상황 업데이트
        function updateProgressData(data) {{
            if (progressChart) {{
                progressChart.data.datasets[0].data = [
                    data.completed_stages,
                    data.active_stages,
                    data.pending_stages
                ];
                progressChart.update();
            }}
        }}
        
        // 알림 업데이트
        function updateAlerts(data) {{
            const alertsContainer = document.getElementById('alerts');
            alertsContainer.innerHTML = '';
            
            data.alerts.forEach(alert => {{
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${{alert.level}}`;
                alertDiv.textContent = alert.message;
                alertsContainer.appendChild(alertDiv);
            }});
        }}
        
        // 자동 새로고침 설정
        function setupAutoRefresh() {{
            const checkbox = document.getElementById('autoRefresh');
            
            checkbox.addEventListener('change', function() {{
                if (this.checked) {{
                    autoRefreshInterval = setInterval(refreshData, 30000);
                }} else {{
                    clearInterval(autoRefreshInterval);
                }}
            }});
            
            if (checkbox.checked) {{
                autoRefreshInterval = setInterval(refreshData, 30000);
            }}
        }}
        
        // 페이지 로드 시 초기화
        document.addEventListener('DOMContentLoaded', function() {{
            initCharts();
            refreshData();
            setupAutoRefresh();
        }});
    </script>
</body>
</html>
        """
    
    def send_json_response(self, data: Dict):
        """JSON 응답 전송"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def get_system_status(self) -> Dict:
        """시스템 상태 데이터 반환"""
        try:
            progress_file = Path(self.dashboard_data["domain_path"]) / "pipeline_progress.json"
            if progress_file.exists():
                progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
                
                completed_stages = sum(1 for stage in progress_data.values() 
                                     if isinstance(stage, dict) and stage.get("status") == "completed")
                total_stages = 5  # 1~5단계
                progress_percent = (completed_stages / total_stages) * 100
                
                current_stage = None
                for stage_name, stage_data in progress_data.items():
                    if isinstance(stage_data, dict) and stage_data.get("status") == "started":
                        current_stage = stage_name
                        break
                
                return {
                    "status": "active" if completed_stages < total_stages else "completed",
                    "status_text": "진행 중" if completed_stages < total_stages else "완료",
                    "current_stage": current_stage,
                    "progress_percent": round(progress_percent, 1),
                    "processed_files": sum(stage.get("files_count", 0) for stage in progress_data.values() 
                                         if isinstance(stage, dict)),
                    "last_update": progress_data.get("metadata", {}).get("last_updated", "알 수 없음")
                }
        except Exception as e:
            pass
        
        return {
            "status": "unknown",
            "status_text": "상태 불명",
            "current_stage": None,
            "progress_percent": 0,
            "processed_files": 0,
            "last_update": "알 수 없음"
        }
    
    def get_performance_data(self) -> Dict:
        """성능 데이터 반환"""
        try:
            metrics_file = Path(self.dashboard_data["domain_path"]) / "performance_metrics.json"
            if metrics_file.exists():
                metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                
                # 최근 성능 데이터 추출
                recent_metrics = []
                for stage_data in metrics_data.get("stages", {}).values():
                    if stage_data.get("runs"):
                        recent_metrics.extend(stage_data["runs"][-3:])  # 최근 3개
                
                if recent_metrics:
                    latest_metric = recent_metrics[-1]
                    return {
                        "cpu_usage": round(latest_metric.get("peak_cpu_percent", 0), 1),
                        "memory_usage": round(latest_metric.get("memory_used_mb", 0) / 1024, 1),  # GB로 변환
                        "processing_time": round(latest_metric.get("duration_seconds", 0), 1),
                        "success_rate": 100 if latest_metric.get("success", False) else 0,
                        "history": {
                            "labels": [f"T-{i}" for i in range(len(recent_metrics), 0, -1)],
                            "f1_scores": [m.get("peak_cpu_percent", 0) / 100 for m in recent_metrics]
                        }
                    }
        except Exception as e:
            pass
        
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "processing_time": 0,
            "success_rate": 0,
            "history": {"labels": [], "f1_scores": []}
        }
    
    def get_progress_data(self) -> Dict:
        """진행 상황 데이터 반환"""
        try:
            progress_file = Path(self.dashboard_data["domain_path"]) / "pipeline_progress.json"
            if progress_file.exists():
                progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
                
                completed = sum(1 for stage in progress_data.values() 
                              if isinstance(stage, dict) and stage.get("status") == "completed")
                active = sum(1 for stage in progress_data.values() 
                           if isinstance(stage, dict) and stage.get("status") == "started")
                pending = 5 - completed - active
                
                return {
                    "completed_stages": completed,
                    "active_stages": active,
                    "pending_stages": pending
                }
        except Exception as e:
            pass
        
        return {
            "completed_stages": 0,
            "active_stages": 0,
            "pending_stages": 5
        }
    
    def get_alerts(self) -> Dict:
        """알림 데이터 반환"""
        alerts = []
        
        try:
            # 성능 임계값 체크
            metrics_file = Path(self.dashboard_data["domain_path"]) / "performance_metrics.json"
            if metrics_file.exists():
                metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                
                for stage_name, stage_data in metrics_data.get("stages", {}).items():
                    if stage_data.get("runs"):
                        latest_run = stage_data["runs"][-1]
                        if latest_run.get("peak_cpu_percent", 0) > 80:
                            alerts.append({
                                "level": "warning",
                                "message": f"{stage_name} 단계 CPU 사용률이 높습니다 ({latest_run['peak_cpu_percent']:.1f}%)"
                            })
                        
                        if latest_run.get("memory_used_mb", 0) > 500:
                            alerts.append({
                                "level": "warning",
                                "message": f"{stage_name} 단계 메모리 사용량이 높습니다 ({latest_run['memory_used_mb']:.1f}MB)"
                            })
        except Exception as e:
            pass
        
        if not alerts:
            alerts.append({
                "level": "info",
                "message": "모든 시스템이 정상적으로 작동 중입니다."
            })
        
        return {"alerts": alerts}
    
    def send_static_file(self, path: str):
        """정적 파일 전송"""
        self.send_error(404, "Static files not implemented")

class WebDashboard:
    """웹 대시보드 서버"""
    
    def __init__(self, domain_path: Path, port: int = 8080):
        self.domain_path = domain_path
        self.port = port
        self.server = None
        self.dashboard_data = {"domain_path": str(domain_path)}
        
        # 로깅 설정
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.domain_path / "dashboard.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_server(self):
        """대시보드 서버 시작"""
        try:
            # 핸들러 팩토리 함수
            def handler_factory(*args, **kwargs):
                return DashboardHandler(*args, dashboard_data=self.dashboard_data, **kwargs)
            
            self.server = HTTPServer(('localhost', self.port), handler_factory)
            
            self.logger.info(f"대시보드 서버 시작: http://localhost:{self.port}")
            print(f"🌐 웹 대시보드가 시작되었습니다!")
            print(f"   URL: http://localhost:{self.port}")
            print(f"   도메인: {self.domain_path}")
            print(f"   중지하려면 Ctrl+C를 누르세요.")
            
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            self.stop_server()
        except Exception as e:
            self.logger.error(f"서버 시작 오류: {e}")
            print(f"❌ 서버 시작 실패: {e}")
    
    def stop_server(self):
        """대시보드 서버 중지"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.logger.info("대시보드 서버 중지")
            print("🛑 웹 대시보드가 중지되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="웹 대시보드 서버")
    parser.add_argument("domain", help="도메인명")
    parser.add_argument("feature", help="기능명")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트")
    parser.add_argument("--base-path", default="data/domains", help="기본 경로")
    
    args = parser.parse_args()
    
    domain_path = Path(args.base_path) / args.domain / args.feature
    
    if not domain_path.exists():
        print(f" 도메인 경로가 존재하지 않습니다: {domain_path}")
        sys.exit(1)
    
    dashboard = WebDashboard(domain_path, args.port)
    dashboard.start_server()

if __name__ == "__main__":
    main()
