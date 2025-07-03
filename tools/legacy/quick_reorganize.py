import os
import shutil
from pathlib import Path

# 디렉토리 생성
dirs = [
    "domains/face_recognition/data/storage/faces",
    "domains/face_recognition/data/storage/persons", 
    "domains/face_recognition/runners/demos",
    "tools/setup"
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")

# 파일 이동
moves = [
    ("run_simple_demo.py", "domains/face_recognition/runners/demos/"),
    ("run_face_recognition_demo.py", "domains/face_recognition/runners/demos/"),
    ("run_face_registration.py", "domains/face_recognition/runners/data_collection/"),
    ("download_models.py", "tools/setup/")
]

for src, dst in moves:
    if Path(src).exists():
        Path(dst).mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst + Path(src).name)
        print(f"Moved: {src} -> {dst}")

# 데이터 저장소 이동
if Path("data/storage").exists():
    dst = Path("domains/face_recognition/data/storage")
    dst.mkdir(parents=True, exist_ok=True)
    
    for item in Path("data/storage").iterdir():
        if item.is_dir():
            dst_item = dst / item.name
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.move(str(item), str(dst_item))
            print(f"Moved data: {item.name}")

print("Reorganization complete!") 