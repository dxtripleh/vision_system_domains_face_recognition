#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition CLI 실행 스크립트.

얼굴인식 명령줄 도구를 실행합니다.
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 경로 추가 (scripts/face_recognition -> vision_system)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.interfaces.cli.face_recognition_cli import FaceRecognitionCLI


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Face Recognition CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령")
    
    # 단일 이미지 검출
    detect_parser = subparsers.add_parser("detect", help="단일 이미지에서 얼굴 검출")
    detect_parser.add_argument("input", help="입력 이미지 경로")
    detect_parser.add_argument("--output", "-o", help="결과 이미지 저장 경로")
    detect_parser.add_argument("--show", action="store_true", help="결과 화면 표시")
    detect_parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    
    # 배치 처리
    batch_parser = subparsers.add_parser("batch", help="디렉토리의 모든 이미지 처리")
    batch_parser.add_argument("input", help="입력 디렉토리 경로")
    batch_parser.add_argument("--output", "-o", help="결과 디렉토리 경로")
    batch_parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    batch_parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
                             help="처리할 파일 확장자")
    
    # 비디오 처리
    video_parser = subparsers.add_parser("video", help="비디오에서 얼굴 검출")
    video_parser.add_argument("input", help="입력 비디오 경로")
    video_parser.add_argument("--output", "-o", help="결과 비디오 저장 경로")
    video_parser.add_argument("--show", action="store_true", help="실시간 결과 표시")
    video_parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    video_parser.add_argument("--frame-skip", type=int, default=1, help="프레임 건너뛰기 수")
    
    # 공통 옵션
    parser.add_argument("--config", "-c", help="설정 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    if not args.command:
        print("❌ 명령을 지정해주세요. --help를 참고하세요.")
        sys.exit(1)
    
    # 로깅 설정
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    try:
        print("=" * 60)
        print("🔍 Face Recognition CLI")
        print("=" * 60)
        print(f"Command: {args.command}")
        print(f"Input: {args.input}")
        print(f"Config: {args.config or 'Default'}")
        print("=" * 60)
        
        # CLI 초기화
        cli = FaceRecognitionCLI(config_path=args.config)
        
        # 명령 실행
        if args.command == "detect":
            result = cli.detect_faces_in_image(
                image_path=args.input,
                output_path=args.output,
                show_result=args.show
            )
            
        elif args.command == "batch":
            result = cli.detect_faces_in_directory(
                input_dir=args.input,
                output_dir=args.output,
                extensions=args.extensions
            )
            
        elif args.command == "video":
            result = cli.process_video(
                video_path=args.input,
                output_path=args.output,
                frame_skip=args.frame_skip,
                show_result=args.show
            )
        
        # 결과 출력
        if result.get("success"):
            print("\n✅ 처리 완료!")
            
            if args.command == "detect":
                print(f"   검출된 얼굴 수: {result.get('faces_count', 0)}")
                print(f"   처리 시간: {result.get('processing_time_ms', 0):.2f}ms")
                
            elif args.command == "batch":
                print(f"   처리된 파일 수: {result.get('total_files', 0)}")
                print(f"   총 검출된 얼굴 수: {result.get('total_faces', 0)}")
                
            elif args.command == "video":
                print(f"   처리된 프레임 수: {result.get('processed_frames', 0)}")
                print(f"   총 검출된 얼굴 수: {result.get('total_faces', 0)}")
        else:
            print(f"\n❌ 처리 실패: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # JSON 저장
        if hasattr(args, 'json') and args.json:
            cli.save_results_json(result, args.json)
            print(f"   결과 JSON 저장됨: {args.json}")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중지되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 