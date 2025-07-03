#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
카메라 실시간 캡쳐 스크립트

- 실시간 카메라 프리뷰 및 프레임 캡쳐
- 토글키 지원: q(종료), s(저장), r(녹화), p(일시정지), h(도움말), i(정보표시 토글)
- 캡쳐 이미지는 data/domains/face_recognition/raw_input/captured/에 저장
"""

import cv2
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 범용 네이밍 시스템 import
try:
    from shared.vision_core.naming import UniversalNamingSystem
except ImportError:
    # 상대 경로로 import 시도
    sys.path.insert(0, str(project_root / "shared"))
    from vision_core.naming import UniversalNamingSystem

# 저장 경로
CAPTURE_DIR = Path("data/domains/face_recognition/raw_input/captured")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def save_frame(frame, cam_id=0, prefix="cap"):
    """프레임을 파일로 저장"""
    ts = get_timestamp()
    
    # 범용 네이밍 시스템 사용
    filename = UniversalNamingSystem.create_capture_filename(ts)
    
    # 날짜별 폴더 생성 (YYYYMMDD까지만)
    date_folder = ts[:8]  # YYYYMMDD
    date_dir = CAPTURE_DIR / date_folder
    date_dir.mkdir(exist_ok=True)
    
    img_path = date_dir / filename
    json_path = date_dir / filename.replace('.jpg', '.json')
    
    cv2.imwrite(str(img_path), frame)
    
    # 메타데이터 저장
    meta = {
        "source": "camera_capture",
        "timestamp": ts,
        "camera_id": cam_id,
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        "created_at": datetime.now().isoformat()
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    logger.info(f"프레임 저장: {img_path}")
    return img_path, json_path

def show_help():
    print("""
[키보드 단축키]
  q : 종료
  s : 현재 프레임 저장
  r : 녹화 시작/중지
  p : 일시정지/재생
  h : 도움말 표시
  i : 카메라 정보 표시 토글
""")

def draw_korean_text_on_image(image, text_lines, font_size=16, position='top'):
    """한글 텍스트를 이미지에 그리기"""
    try:
        # OpenCV 이미지를 PIL 이미지로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 폰트 설정 (한글 폰트 우선 시도)
        font = None
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",   # 굴림
            "C:/Windows/Fonts/batang.ttc",  # 바탕
            "malgun.ttf",
            "arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # 위치에 따른 y_offset 설정
        if position == 'top':
            y_offset = 10
        elif position == 'bottom':
            y_offset = image.shape[0] - len(text_lines) * (font_size + 5) - 10
        else:
            y_offset = 10
        
        # 텍스트 그리기
        for line in text_lines:
            draw.text((10, y_offset), line, font=font, fill=(0, 255, 0))
            y_offset += font_size + 5
        
        # PIL 이미지를 OpenCV 이미지로 변환
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image_bgr
        
    except Exception as e:
        logger.warning(f"한글 텍스트 그리기 실패: {e}")
        # 실패 시 영문으로 대체
        return draw_english_text_on_image(image, text_lines, position)

def draw_english_text_on_image(image, text_lines, position='top'):
    """영문 텍스트를 이미지에 그리기 (백업)"""
    font_size = 0.6
    line_height = 20
    
    if position == 'bottom':
        y_start = image.shape[0] - len(text_lines) * line_height - 10
    else:
        y_start = 10
    
    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,255,0), 1, cv2.LINE_AA)
    return image

def draw_camera_info(frame, info: dict, show_info: bool):
    if not show_info:
        return frame
    
    # 카메라 정보 (위쪽, 작게)
    camera_info_lines = [
        f"Cam:{info['cam_id']} {info['width']}x{info['height']} {info['fps']}fps",
        f"Save:{'ON' if info['saving'] else 'OFF'} Rec:{'ON' if info['recording'] else 'OFF'} Pause:{'ON' if info['paused'] else 'OFF'}"
    ]
    
    frame = draw_korean_text_on_image(frame, camera_info_lines, font_size=14, position='top')
    
    # 토글키 정보 (아래쪽, 작게)
    toggle_info_lines = [
        "i:정보 h:도움말 s:저장 r:녹화 p:일시정지 q:종료"
    ]
    
    frame = draw_korean_text_on_image(frame, toggle_info_lines, font_size=12, position='bottom')
    
    return frame

def main():
    cam_id = 0
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        logger.error("카메라를 열 수 없습니다.")
        sys.exit(1)

    # 카메라 초기화 시간 대기
    logger.info("카메라 초기화 중...")
    time.sleep(2)
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logger.info("카메라 캡쳐 시작 (q: 종료, s: 저장, r: 녹화, p: 일시정지, h: 도움말, i: 정보표시 토글)")
    show_help()

    recording = False
    paused = False
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = None
    show_info = True
    saving = False
    frame_count = 0
    max_retries = 10

    # 창 크기 설정
    cv2.namedWindow('Camera Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Capture', 1280, 720)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                frame_count += 1
                if frame_count <= max_retries:
                    logger.warning(f"프레임을 읽을 수 없습니다. 재시도 {frame_count}/{max_retries}")
                    time.sleep(0.1)
                    continue
                else:
                    logger.error("카메라에서 프레임을 읽을 수 없습니다. 종료합니다.")
                    break
            frame_count = 0  # 성공 시 카운터 리셋
        else:
            # 일시정지 상태에서는 이전 프레임 유지
            pass

        info = {
            'cam_id': cam_id,
            'width': width,
            'height': height,
            'fps': fps,
            'saving': saving,
            'recording': recording,
            'paused': paused
        }
        frame_disp = draw_camera_info(frame.copy(), info, show_info)
        cv2.imshow('Camera Capture', frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("종료합니다.")
            break
        elif key == ord('s'):
            save_frame(frame, cam_id)
            saving = True
        elif key == ord('r'):
            recording = not recording
            if recording:
                out_path = CAPTURE_DIR / f"record_{get_timestamp()}.avi"
                video_writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                logger.info(f"녹화 시작: {out_path}")
            else:
                if video_writer:
                    video_writer.release()
                    logger.info("녹화 중지")
        elif key == ord('p'):
            paused = not paused
            logger.info(f"일시정지: {paused}")
        elif key == ord('h'):
            show_help()
        elif key == ord('i'):
            show_info = not show_info
            logger.info(f"카메라 정보 표시: {show_info}")

        if recording and not paused and video_writer:
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 