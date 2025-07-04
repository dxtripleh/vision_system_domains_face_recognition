#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파일 유틸리티 모듈.

파일 저장 시 올바른 위치를 강제하고, 잘못된 위치에 파일이 생성되는 것을 방지합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import yaml
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FileLocationEnforcer:
    """파일 위치 강제기"""
    
    # 허용된 파일 위치 매핑
    ALLOWED_LOCATIONS = {
        # 로그 파일
        '*.log': 'data/logs/',
        '*.txt': 'data/logs/',
        
        # 이미지 파일
        '*.jpg': 'data/output/',
        '*.jpeg': 'data/output/',
        '*.png': 'data/output/',
        '*.bmp': 'data/output/',
        '*.tiff': 'data/output/',
        '*.tif': 'data/output/',
        
        # 비디오 파일
        '*.mp4': 'data/output/',
        '*.avi': 'data/output/',
        '*.mov': 'data/output/',
        '*.mkv': 'data/output/',
        
        # 데이터 파일
        '*.json': 'data/temp/',
        '*.csv': 'data/temp/',
        '*.xml': 'data/temp/',
        '*.yaml': 'data/temp/',
        '*.yml': 'data/temp/',
        
        # 임시 파일
        '*.tmp': 'data/temp/',
        '*.temp': 'data/temp/',
        'temp_*': 'data/temp/',
        'debug_*': 'data/temp/',
        
        # 모델 파일
        '*.onnx': 'models/weights/',
        '*.pt': 'models/weights/',
        '*.pth': 'models/weights/',
        '*.h5': 'models/weights/',
        '*.pb': 'models/weights/',
        '*.tflite': 'models/weights/',
        
        # 데이터셋 파일
        'dataset_*': 'datasets/',
        'train_*': 'datasets/',
        'val_*': 'datasets/',
        'test_*': 'datasets/'
    }
    
    # 금지된 위치 (프로젝트 루트)
    FORBIDDEN_ROOT_PATTERNS = [
        '*.log', '*.tmp', '*.temp', '*.jpg', '*.png', '*.mp4', 
        '*.json', '*.csv', '*.txt', 'debug_*', 'temp_*', 'output_*',
        'result_*', 'capture_*', 'frame_*', 'image_*', 'video_*'
    ]
    
    def __init__(self, project_root: Optional[Path] = None):
        """초기화"""
        if project_root is None:
            # 현재 파일에서 프로젝트 루트 찾기
            current_file = Path(__file__)
            self.project_root = current_file.parent.parent
        else:
            self.project_root = project_root
    
    def get_correct_path(self, filename: str, file_type: Optional[str] = None) -> Path:
        """파일명에 따른 올바른 경로 반환"""
        import fnmatch
        
        # 파일 타입이 지정된 경우
        if file_type:
            if file_type in self.ALLOWED_LOCATIONS:
                return self.project_root / self.ALLOWED_LOCATIONS[file_type] / filename
        
        # 패턴 매칭으로 찾기
        for pattern, location in self.ALLOWED_LOCATIONS.items():
            if fnmatch.fnmatch(filename, pattern):
                return self.project_root / location / filename
        
        # 기본값: temp 폴더
        return self.project_root / 'data/temp' / filename
    
    def validate_path(self, file_path: Union[str, Path]) -> bool:
        """파일 경로가 올바른지 검증"""
        file_path = Path(file_path)
        
        # 프로젝트 루트에 직접 있는 파일인지 확인
        if file_path.parent == self.project_root:
            filename = file_path.name
            
            # 허용된 프로젝트 파일들
            allowed_project_files = {
                'README.md', 'requirements.txt', 'pyproject.toml', 
                'setup.py', 'LICENSE', '.gitignore', '.env.example',
                'pytest.ini', 'tox.ini', 'Makefile', 'Dockerfile',
                'docker-compose.yml', '.dockerignore'
            }
            
            if filename in allowed_project_files:
                return True
            
            # 금지된 패턴과 매치되는지 확인
            import fnmatch
            for pattern in self.FORBIDDEN_ROOT_PATTERNS:
                if fnmatch.fnmatch(filename, pattern):
                    return False
        
        return True
    
    def enforce_save_path(self, file_path: Union[str, Path], 
                         create_dirs: bool = True) -> Path:
        """저장 경로 강제 적용"""
        file_path = Path(file_path)
        
        # 이미 올바른 경로인 경우
        if self.validate_path(file_path):
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            return file_path
        
        # 올바른 경로로 변경
        correct_path = self.get_correct_path(file_path.name)
        
        if create_dirs:
            correct_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.warning(f"파일 경로가 잘못되었습니다: {file_path}")
        logger.info(f"올바른 경로로 변경: {correct_path}")
        
        return correct_path

# 전역 인스턴스
file_enforcer = FileLocationEnforcer()

def save_image(image: np.ndarray, filename: str, 
               output_dir: Optional[str] = None) -> Path:
    """이미지 저장 (올바른 위치 강제)"""
    if output_dir:
        # 사용자 지정 디렉토리가 있는 경우 검증
        save_path = Path(output_dir) / filename
        if not file_enforcer.validate_path(save_path):
            logger.warning(f"지정된 경로가 올바르지 않습니다: {save_path}")
            save_path = file_enforcer.get_correct_path(filename)
    else:
        # 자동으로 올바른 위치 결정
        save_path = file_enforcer.get_correct_path(filename)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 이미지 저장
    success = cv2.imwrite(str(save_path), image)
    if success:
        logger.info(f"이미지 저장 완료: {save_path}")
        return save_path
    else:
        raise RuntimeError(f"이미지 저장 실패: {save_path}")

def save_json(data: Dict[str, Any], filename: str, 
              output_dir: Optional[str] = None) -> Path:
    """JSON 파일 저장 (올바른 위치 강제)"""
    if output_dir:
        save_path = Path(output_dir) / filename
        if not file_enforcer.validate_path(save_path):
            save_path = file_enforcer.get_correct_path(filename)
    else:
        save_path = file_enforcer.get_correct_path(filename)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON 저장 완료: {save_path}")
    return save_path

def save_yaml(data: Dict[str, Any], filename: str, 
              output_dir: Optional[str] = None) -> Path:
    """YAML 파일 저장 (올바른 위치 강제)"""
    if output_dir:
        save_path = Path(output_dir) / filename
        if not file_enforcer.validate_path(save_path):
            save_path = file_enforcer.get_correct_path(filename)
    else:
        save_path = file_enforcer.get_correct_path(filename)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # YAML 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"YAML 저장 완료: {save_path}")
    return save_path

def save_csv(data: list, filename: str, 
             output_dir: Optional[str] = None) -> Path:
    """CSV 파일 저장 (올바른 위치 강제)"""
    import pandas as pd
    
    if output_dir:
        save_path = Path(output_dir) / filename
        if not file_enforcer.validate_path(save_path):
            save_path = file_enforcer.get_correct_path(filename)
    else:
        save_path = file_enforcer.get_correct_path(filename)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # DataFrame으로 변환하여 저장
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    logger.info(f"CSV 저장 완료: {save_path}")
    return save_path

def save_video(frames: list, filename: str, 
               fps: int = 30, output_dir: Optional[str] = None) -> Path:
    """비디오 저장 (올바른 위치 강제)"""
    if output_dir:
        save_path = Path(output_dir) / filename
        if not file_enforcer.validate_path(save_path):
            save_path = file_enforcer.get_correct_path(filename)
    else:
        save_path = file_enforcer.get_correct_path(filename)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not frames:
        raise ValueError("저장할 프레임이 없습니다.")
    
    # 비디오 작성기 생성
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    # 프레임 저장
    for frame in frames:
        out.write(frame)
    
    out.release()
    logger.info(f"비디오 저장 완료: {save_path}")
    return save_path

def create_temp_file(prefix: str = "temp", suffix: str = ".tmp") -> Path:
    """임시 파일 생성 (올바른 위치)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{prefix}_{timestamp}{suffix}"
    
    temp_path = file_enforcer.get_correct_path(filename)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 빈 파일 생성
    temp_path.touch()
    
    logger.info(f"임시 파일 생성: {temp_path}")
    return temp_path

def get_output_path(filename: str, file_type: Optional[str] = None) -> Path:
    """출력 파일 경로 생성 (올바른 위치)"""
    return file_enforcer.get_correct_path(filename, file_type)

def validate_and_fix_path(file_path: Union[str, Path]) -> Path:
    """파일 경로 검증 및 수정"""
    file_path = Path(file_path)
    
    if file_enforcer.validate_path(file_path):
        return file_path
    else:
        return file_enforcer.get_correct_path(file_path.name)

# 데코레이터: 파일 저장 함수에 자동으로 경로 검증 적용
def enforce_file_location(func):
    """파일 저장 함수에 경로 강제 적용 데코레이터"""
    def wrapper(*args, **kwargs):
        # 파일 경로가 첫 번째 인자인 경우
        if args and isinstance(args[0], (str, Path)):
            file_path = Path(args[0])
            if not file_enforcer.validate_path(file_path):
                # 올바른 경로로 변경
                correct_path = file_enforcer.get_correct_path(file_path.name)
                logger.warning(f"파일 경로 수정: {file_path} -> {correct_path}")
                args = (correct_path,) + args[1:]
        
        # 파일 경로가 kwargs에 있는 경우
        if 'file_path' in kwargs:
            file_path = Path(kwargs['file_path'])
            if not file_enforcer.validate_path(file_path):
                correct_path = file_enforcer.get_correct_path(file_path.name)
                logger.warning(f"파일 경로 수정: {file_path} -> {correct_path}")
                kwargs['file_path'] = correct_path
        
        return func(*args, **kwargs)
    
    return wrapper 