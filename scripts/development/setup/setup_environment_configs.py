#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Configuration Management.

환경별 설정 파일 자동 생성 및 관리 시스템입니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class EnvironmentConfigManager:
    """환경별 설정 관리자"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "config"
        self.config_dir.mkdir(exist_ok=True)
    
    def setup_all_environments(self):
        """모든 환경 설정 파일 생성"""
        logger.info("환경별 설정 파일 생성 시작...")
        
        # 개발 환경 설정
        dev_config = {
            "environment": "development",
            "debug": True,
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "workers": 1
            },
            "face_recognition": {
                "use_mock": True,
                "similarity_threshold": 0.6
            },
            "logging": {
                "level": "DEBUG"
            }
        }
        
        # 프로덕션 환경 설정
        prod_config = {
            "environment": "production",
            "debug": False,
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "face_recognition": {
                "use_mock": False,
                "similarity_threshold": 0.8
            },
            "logging": {
                "level": "WARNING"
            }
        }
        
        # 설정 파일 저장
        configs = {
            'development': dev_config,
            'production': prod_config
        }
        
        for env_name, config in configs.items():
            config_file = self.config_dir / f"{env_name}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"생성: {config_file}")
        
        # .env 템플릿 생성
        env_content = """# Environment Variables
ENVIRONMENT=development
DEBUG=true
API_HOST=127.0.0.1
API_PORT=8000
USE_MOCK_RECOGNIZER=true
"""
        
        env_file = self.project_root / ".env.example"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"생성: {env_file}")
        logger.info("✅ 환경별 설정 파일 생성 완료!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = EnvironmentConfigManager()
    manager.setup_all_environments() 