#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition CLI.

얼굴인식 시스템의 명령줄 인터페이스입니다.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import get_logger
from common.config_loader import load_config
from ...core.services.face_detection_service import FaceDetectionService
from ...core.services.face_recognition_service import FaceRecognitionService
from ...core.services.face_matching_service import FaceMatchingService

logger = get_logger(__name__)


class FaceRecognitionCLI:
    """얼굴인식 CLI 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        CLI 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path) if config_path else {}
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService(config=self.config.get('detection', {}))
        self.recognition_service = FaceRecognitionService(config=self.config.get('recognition', {}))
        self.matching_service = FaceMatchingService(config=self.config.get('matching', {}))
        
        logger.info("FaceRecognitionCLI 초기화 완료")
    
    def detect_faces_in_image(self, image_path: str, output_path: Optional[str] = None, 
                             show_result: bool = False) -> Dict[str, Any]:
        """
        이미지에서 얼굴을 검출합니다.
        
        Args:
            image_path: 입력 이미지 경로
            output_path: 결과 저장 경로 (선택사항)
            show_result: 결과 화면 표시 여부
            
        Returns:
            Dict[str, Any]: 검출 결과
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            logger.info(f"얼굴 검출 시작 - 이미지: {image_path}")
            
            # 얼굴 검출
            result = self.detection_service.detect_faces(image)
            
            # 결과 시각화
            if result.faces:
                annotated_image = self._draw_faces(image.copy(), result.faces)
                
                # 결과 저장
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, annotated_image)
                    logger.info(f"결과 저장됨: {output_path}")
                
                # 결과 표시
                if show_result:
                    cv2.imshow('Face Detection Result', annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            # 결과 정보
            result_info = {
                "success": True,
                "image_path": image_path,
                "faces_count": len(result.faces),
                "processing_time_ms": result.processing_time_ms,
                "model_name": result.model_name,
                "faces": [
                    {
                        "face_id": face.face_id,
                        "bbox": {
                            "x": face.bbox.x,
                            "y": face.bbox.y,
                            "width": face.bbox.width,
                            "height": face.bbox.height
                        },
                        "confidence": face.confidence,
                        "quality_score": face.quality_score
                    }
                    for face in result.faces
                ]
            }
            
            logger.info(f"얼굴 검출 완료 - {len(result.faces)}개 검출, {result.processing_time_ms:.2f}ms")
            
            return result_info
            
        except Exception as e:
            logger.error(f"얼굴 검출 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def detect_faces_in_directory(self, input_dir: str, output_dir: Optional[str] = None,
                                 extensions: List[str] = None) -> Dict[str, Any]:
        """
        디렉토리의 모든 이미지에서 얼굴을 검출합니다.
        
        Args:
            input_dir: 입력 디렉토리 경로
            output_dir: 출력 디렉토리 경로 (선택사항)
            extensions: 처리할 파일 확장자 리스트
            
        Returns:
            Dict[str, Any]: 배치 처리 결과
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            return {"success": False, "error": f"디렉토리가 존재하지 않습니다: {input_dir}"}
        
        # 이미지 파일 찾기
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            return {"success": False, "error": "처리할 이미지 파일이 없습니다"}
        
        logger.info(f"배치 처리 시작 - {len(image_files)}개 이미지")
        
        results = []
        total_faces = 0
        
        for image_file in image_files:
            try:
                # 출력 경로 설정
                if output_dir:
                    output_path = Path(output_dir) / f"detected_{image_file.name}"
                else:
                    output_path = None
                
                # 얼굴 검출
                result = self.detect_faces_in_image(
                    str(image_file), 
                    str(output_path) if output_path else None
                )
                
                results.append({
                    "file": str(image_file),
                    "result": result
                })
                
                if result.get("success"):
                    total_faces += result.get("faces_count", 0)
                
            except Exception as e:
                logger.error(f"파일 처리 중 오류 {image_file}: {str(e)}")
                results.append({
                    "file": str(image_file),
                    "result": {"success": False, "error": str(e)}
                })
        
        summary = {
            "success": True,
            "total_files": len(image_files),
            "total_faces": total_faces,
            "results": results
        }
        
        logger.info(f"배치 처리 완료 - {len(image_files)}개 파일, {total_faces}개 얼굴")
        
        return summary
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     frame_skip: int = 1, show_result: bool = False) -> Dict[str, Any]:
        """
        비디오에서 얼굴을 검출합니다.
        
        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로 (선택사항)
            frame_skip: 프레임 건너뛰기 수
            show_result: 결과 화면 표시 여부
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
            
            # 비디오 정보
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"비디오 처리 시작 - {video_path} ({width}x{height}, {fps}fps, {total_frames}프레임)")
            
            # 비디오 라이터 설정
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            total_faces = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 프레임 건너뛰기
                if frame_count % (frame_skip + 1) != 0:
                    if out:
                        out.write(frame)
                    continue
                
                processed_frames += 1
                
                try:
                    # 얼굴 검출
                    result = self.detection_service.detect_faces(frame)
                    
                    # 결과 그리기
                    if result.faces:
                        frame = self._draw_faces(frame, result.faces)
                        total_faces += len(result.faces)
                    
                    # 프레임 정보 표시
                    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Faces: {len(result.faces)}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    logger.warning(f"프레임 {frame_count} 처리 중 오류: {str(e)}")
                
                # 결과 저장
                if out:
                    out.write(frame)
                
                # 결과 표시
                if show_result:
                    cv2.imshow('Video Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # 리소스 정리
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            result_info = {
                "success": True,
                "video_path": video_path,
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "total_faces": total_faces,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
            logger.info(f"비디오 처리 완료 - {processed_frames}개 프레임 처리, {total_faces}개 얼굴 검출")
            
            return result_info
            
        except Exception as e:
            logger.error(f"비디오 처리 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _draw_faces(self, image, faces) -> any:
        """이미지에 얼굴 바운딩 박스 그리기"""
        for face in faces:
            x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 신뢰도 표시
            label = f"Conf: {face.confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 품질 점수 표시
            quality_label = f"Quality: {face.quality_score:.2f}"
            cv2.putText(image, quality_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return image
    
    def save_results_json(self, results: Dict[str, Any], output_path: str):
        """결과를 JSON 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"결과 JSON 저장됨: {output_path}")
        except Exception as e:
            logger.error(f"JSON 저장 중 오류: {str(e)}")


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description="Face Recognition CLI")
    parser.add_argument("command", choices=["detect", "batch", "video"], help="실행할 명령")
    parser.add_argument("--input", "-i", required=True, help="입력 파일/디렉토리 경로")
    parser.add_argument("--output", "-o", help="출력 파일/디렉토리 경로")
    parser.add_argument("--config", "-c", help="설정 파일 경로")
    parser.add_argument("--show", action="store_true", help="결과 화면 표시")
    parser.add_argument("--json", help="결과를 JSON으로 저장할 경로")
    parser.add_argument("--frame-skip", type=int, default=1, help="비디오 프레임 건너뛰기 수")
    
    args = parser.parse_args()
    
    try:
        # CLI 초기화
        cli = FaceRecognitionCLI(args.config)
        
        # 명령 실행
        if args.command == "detect":
            result = cli.detect_faces_in_image(args.input, args.output, args.show)
        elif args.command == "batch":
            result = cli.detect_faces_in_directory(args.input, args.output)
        elif args.command == "video":
            result = cli.process_video(args.input, args.output, args.frame_skip, args.show)
        else:
            raise ValueError(f"지원하지 않는 명령: {args.command}")
        
        # 결과 출력
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # JSON 저장
        if args.json:
            cli.save_results_json(result, args.json)
        
        # 성공 여부에 따른 종료 코드
        sys.exit(0 if result.get("success", False) else 1)
        
    except Exception as e:
        logger.error(f"CLI 실행 중 오류: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 