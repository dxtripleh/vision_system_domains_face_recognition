#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition API 실행 스크립트.

얼굴인식 REST API 서버를 실행합니다.
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가 (scripts/domains/face_recognition -> vision_system)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.interfaces.api.face_recognition_api import FaceRecognitionAPI


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Face Recognition API 서버")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    parser.add_argument("--reload", action="store_true", help="자동 리로드 (개발용)")
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 로깅 설정
    setup_logging()
    
    try:
        print("=" * 60)
        print("🚀 Face Recognition API Server")
        print("=" * 60)
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Debug: {args.debug}")
        print(f"Config: {args.config or 'Default'}")
        print("=" * 60)
        
        # API 서버 생성 및 실행
        api = FaceRecognitionAPI(config_path=args.config)
        
        if args.reload:
            # 개발용 자동 리로드
            import uvicorn
            uvicorn.run(
                "domains.face_recognition.interfaces.api.face_recognition_api:create_app",
                host=args.host,
                port=args.port,
                reload=True,
                debug=args.debug
            )
        else:
            # 일반 실행
            api.run(host=args.host, port=args.port, debug=args.debug)
            
    except KeyboardInterrupt:
        print("\n🛑 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 