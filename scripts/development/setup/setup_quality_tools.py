#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quality Tools Setup Script.

코드 품질 도구 자동 설정 및 초기화 스크립트입니다.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any
import shutil

logger = logging.getLogger(__name__)


class QualityToolsSetup:
    """코드 품질 도구 설정"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
    
    def setup_all_tools(self):
        """모든 품질 도구 설정"""
        logger.info("코드 품질 도구 설정 시작...")
        
        try:
            # 1. 도구 설치
            self._install_tools()
            
            # 2. 설정 파일 생성
            self._create_config_files()
            
            logger.info("✅ 코드 품질 도구 설정 완료!")
            
        except Exception as e:
            logger.error(f"❌ 설정 중 오류 발생: {str(e)}")
            raise
    
    def _install_tools(self):
        """도구 설치"""
        tools = [
            'black==23.3.0',
            'isort==5.12.0', 
            'flake8==6.0.0',
            'mypy==1.3.0',
            'pytest==7.3.0',
            'pytest-cov==4.0.0'
        ]
        
        for tool in tools:
            logger.info(f"설치 중: {tool}")
            subprocess.run(['pip', 'install', tool], check=True)
    
    def _create_config_files(self):
        """설정 파일 생성"""
        # pyproject.toml 생성
        pyproject_content = """[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.9"
warn_return_any = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["domains/face_recognition/tests"]
addopts = ["-v", "--cov=domains/face_recognition/core"]
"""
        
        config_file = self.project_root / "pyproject.toml"
        with open(config_file, 'w') as f:
            f.write(pyproject_content)
        
        logger.info("설정 파일 생성 완료")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup = QualityToolsSetup()
    setup.setup_all_tools() 