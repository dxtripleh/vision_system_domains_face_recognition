#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
detected_faces 폴더를 날짜별로 정리하는 스크립트

기존의 복잡한 하위 폴더 구조를 날짜별로 단순화합니다.
"""

import os
import shutil
import re
from datetime import datetime
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_date_from_filename(filename):
    """파일명에서 날짜 추출"""
    # 패턴: YYYYMMDD_HHMMSS
    pattern = r'(\d{8})_(\d{6})'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return f"{date_str}_{time_str[:4]}"  # YYYYMMDD_HHMM 형식
    return None

def reorganize_detected_faces():
    """detected_faces 폴더를 날짜별로 정리"""
    base_path = Path("data/domains/face_recognition/detected_faces")
    
    if not base_path.exists():
        logger.error("detected_faces 폴더가 존재하지 않습니다.")
        return
    
    # 기존 하위 폴더들
    old_folders = ['auto_collected', 'from_captured', 'from_manual', 'from_uploads']
    
    # 모든 파일을 수집
    all_files = []
    for folder in old_folders:
        folder_path = base_path / folder
        if folder_path.exists():
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.jpg', '.png', '.json']:
                    all_files.append(file_path)
    
    logger.info(f"총 {len(all_files)}개의 파일을 찾았습니다.")
    
    # 날짜별로 파일 분류
    date_groups = {}
    for file_path in all_files:
        date_key = extract_date_from_filename(file_path.name)
        if date_key:
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(file_path)
        else:
            # 날짜를 추출할 수 없는 파일은 'unknown' 폴더로
            if 'unknown' not in date_groups:
                date_groups['unknown'] = []
            date_groups['unknown'].append(file_path)
    
    # 기존 폴더 백업
    backup_path = base_path / "backup_old_structure"
    if backup_path.exists():
        shutil.rmtree(backup_path)
    backup_path.mkdir(exist_ok=True)
    
    for folder in old_folders:
        folder_path = base_path / folder
        if folder_path.exists():
            try:
                shutil.move(str(folder_path), str(backup_path / folder))
                logger.info(f"백업: {folder} -> backup_old_structure/")
            except Exception as e:
                logger.error(f"백업 실패 {folder}: {e}")
    
    # 날짜별 폴더 생성 및 파일 이동
    for date_key, files in date_groups.items():
        date_folder = base_path / date_key
        date_folder.mkdir(exist_ok=True)
        
        for file_path in files:
            try:
                # 파일이 실제로 존재하는지 확인
                if not file_path.exists():
                    logger.warning(f"파일이 존재하지 않음: {file_path}")
                    continue
                
                # 파일을 새 위치로 이동
                new_path = date_folder / file_path.name
                if new_path.exists():
                    # 중복 파일 처리
                    counter = 1
                    while new_path.exists():
                        name_parts = file_path.stem.split('_')
                        if len(name_parts) > 1 and name_parts[-1].isdigit():
                            name_parts[-1] = str(int(name_parts[-1]) + counter)
                        else:
                            name_parts.append(str(counter))
                        new_name = '_'.join(name_parts) + file_path.suffix
                        new_path = date_folder / new_name
                        counter += 1
                
                # 파일 복사 후 원본 삭제 (더 안전한 방식)
                shutil.copy2(str(file_path), str(new_path))
                file_path.unlink()  # 원본 파일 삭제
                logger.info(f"이동: {file_path.name} -> {date_key}/")
                
            except Exception as e:
                logger.error(f"파일 이동 실패 {file_path}: {e}")
    
    # README 파일 업데이트
    update_readme(base_path)
    
    logger.info("detected_faces 폴더 정리 완료!")

def update_readme(base_path):
    """README 파일 업데이트"""
    readme_content = """# Detected Faces

이 폴더는 얼굴 검출 결과를 저장합니다.

## 폴더 구조

- `YYYYMMDD_HHMM/` - 날짜별로 정리된 검출된 얼굴 이미지 및 메타데이터
- `unknown/` - 날짜를 추출할 수 없는 파일들
- `backup_old_structure/` - 기존 구조 백업 (삭제 가능)

## 파일 형식

- `*.jpg`, `*.png` - 검출된 얼굴 이미지
- `*.json` - 얼굴 검출 메타데이터 (바운딩 박스, 신뢰도 등)

## 메타데이터 구조

```json
{
    "bbox": [x, y, width, height],
    "confidence": 0.95,
    "landmarks": [[x1, y1], [x2, y2], ...],
    "timestamp": "20250630_110859",
    "source_file": "original_image.jpg"
}
```

## 정리 규칙

- 파일명에서 날짜(YYYYMMDD_HHMM)를 추출하여 해당 폴더로 분류
- 중복 파일명은 자동으로 번호를 추가하여 구분
- 날짜를 추출할 수 없는 파일은 'unknown' 폴더로 이동
"""
    
    readme_path = base_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

if __name__ == "__main__":
    reorganize_detected_faces() 