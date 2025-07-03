#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Recognition API.

얼굴인식 도메인의 메인 API 클래스입니다.
"""

import os
import time
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from common.logging import get_logger
from common.config_loader import load_config
from ...core.services.face_detection_service import FaceDetectionService
from ...core.services.face_recognition_service import FaceRecognitionService
from ...core.services.face_matching_service import FaceMatchingService

logger = get_logger(__name__)


class FaceRecognitionAPI:
    """얼굴인식 API 메인 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        API 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path) if config_path else {}
        self.api_config = self.config.get('api', {})
        
        # 서비스 초기화
        self.detection_service = FaceDetectionService(config=self.config.get('detection', {}))
        self.recognition_service = FaceRecognitionService(config=self.config.get('recognition', {}))
        self.matching_service = FaceMatchingService(config=self.config.get('matching', {}))
        
        # FastAPI 앱 생성
        self.app = FastAPI(
            title="Face Recognition API",
            description="얼굴인식 시스템 REST API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS 설정
        self._setup_cors()
        
        # 라우트 설정
        self._setup_routes()
        
        logger.info("FaceRecognitionAPI 초기화 완료")
    
    def _setup_cors(self):
        """CORS 설정"""
        allowed_origins = self.api_config.get('allowed_origins', ["*"])
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/")
        async def root():
            """API 루트 엔드포인트"""
            return {
                "message": "Face Recognition API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": time.time()
            }
        
        @self.app.get("/health")
        async def health_check():
            """헬스 체크 엔드포인트"""
            return {
                "status": "healthy",
                "services": {
                    "detection": "running",
                    "recognition": "running",
                    "matching": "running"
                },
                "timestamp": time.time()
            }
        
        @self.app.post("/detect")
        async def detect_faces(file: UploadFile = File(...)):
            """얼굴 검출 엔드포인트"""
            try:
                # 이미지 파일 읽기
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # 얼굴 검출
                result = self.detection_service.detect_faces(image)
                
                # 응답 형식 변환
                response = {
                    "success": True,
                    "image_id": result.image_id,
                    "faces_count": len(result.faces),
                    "processing_time_ms": result.processing_time_ms,
                    "model_name": result.model_name,
                    "faces": [self._face_to_dict(face) for face in result.faces]
                }
                
                logger.info(f"얼굴 검출 완료 - 파일: {file.filename}, 검출 수: {len(result.faces)}")
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logger.error(f"얼굴 검출 중 오류: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/recognize")
        async def recognize_face(file: UploadFile = File(...)):
            """얼굴 인식 엔드포인트"""
            try:
                # 이미지 파일 읽기
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # 얼굴 검출
                detection_result = self.detection_service.detect_faces(image)
                
                if not detection_result.faces:
                    return JSONResponse(content={
                        "success": True,
                        "message": "얼굴이 검출되지 않았습니다",
                        "faces": []
                    })
                
                # 각 얼굴에 대해 인식 수행
                recognition_results = []
                for face in detection_result.faces:
                    # 얼굴 영역 추출
                    face_image = self._extract_face_region(image, face)
                    
                    # 임베딩 추출
                    embedding = self.recognition_service.extract_embedding(face_image)
                    face.embedding = embedding
                    
                    # 인물 식별
                    identified_person = self.recognition_service.identify_face(face)
                    
                    face_result = {
                        "face_id": face.face_id,
                        "bbox": {
                            "x": face.bbox.x,
                            "y": face.bbox.y,
                            "width": face.bbox.width,
                            "height": face.bbox.height
                        },
                        "confidence": face.confidence,
                        "quality_score": face.quality_score,
                        "person": {
                            "person_id": identified_person.person_id if identified_person else None,
                            "name": identified_person.name if identified_person else "Unknown"
                        }
                    }
                    
                    recognition_results.append(face_result)
                
                response = {
                    "success": True,
                    "image_id": detection_result.image_id,
                    "processing_time_ms": detection_result.processing_time_ms,
                    "faces": recognition_results
                }
                
                logger.info(f"얼굴 인식 완료 - 파일: {file.filename}, 처리 수: {len(recognition_results)}")
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logger.error(f"얼굴 인식 중 오류: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/register")
        async def register_person(
            name: str = Form(...),
            files: List[UploadFile] = File(...)
        ):
            """인물 등록 엔드포인트"""
            try:
                if not name or not files:
                    raise HTTPException(status_code=400, detail="이름과 얼굴 이미지가 필요합니다")
                
                # 이미지 파일들 읽기
                face_images = []
                for file in files:
                    image_data = await file.read()
                    image = self._decode_image(image_data)
                    face_images.append(image)
                
                # 인물 등록
                person_id = self.recognition_service.register_person(
                    name=name,
                    face_images=face_images
                )
                
                response = {
                    "success": True,
                    "person_id": person_id,
                    "name": name,
                    "images_count": len(face_images),
                    "message": "인물이 성공적으로 등록되었습니다"
                }
                
                logger.info(f"인물 등록 완료 - 이름: {name}, ID: {person_id}")
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logger.error(f"인물 등록 중 오류: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/verify")
        async def verify_face(
            person_id: str = Form(...),
            file: UploadFile = File(...)
        ):
            """얼굴 검증 엔드포인트"""
            try:
                # 이미지 파일 읽기
                image_data = await file.read()
                image = self._decode_image(image_data)
                
                # 얼굴 검출
                detection_result = self.detection_service.detect_faces(image)
                
                if not detection_result.faces:
                    return JSONResponse(content={
                        "success": False,
                        "message": "얼굴이 검출되지 않았습니다",
                        "verified": False,
                        "similarity": 0.0
                    })
                
                # 가장 큰 얼굴 선택
                best_face = max(detection_result.faces, key=lambda f: f.bbox.area)
                
                # 얼굴 영역 추출 및 임베딩 생성
                face_image = self._extract_face_region(image, best_face)
                embedding = self.recognition_service.extract_embedding(face_image)
                best_face.embedding = embedding
                
                # 검증 수행
                is_verified, similarity = self.recognition_service.verify_face(best_face, person_id)
                
                response = {
                    "success": True,
                    "person_id": person_id,
                    "verified": is_verified,
                    "similarity": similarity,
                    "face_id": best_face.face_id,
                    "confidence": best_face.confidence
                }
                
                logger.info(f"얼굴 검증 완료 - 인물: {person_id}, 검증: {is_verified}, 유사도: {similarity:.3f}")
                
                return JSONResponse(content=response)
                
            except Exception as e:
                logger.error(f"얼굴 검증 중 오류: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_statistics():
            """통계 정보 엔드포인트"""
            try:
                stats = {
                    "detection_service": self.detection_service.get_statistics(),
                    "recognition_service": self.recognition_service.get_statistics(),
                    "matching_service": self.matching_service.get_statistics(),
                    "api_info": {
                        "version": "1.0.0",
                        "uptime": time.time(),
                        "config": self.api_config
                    }
                }
                
                return JSONResponse(content=stats)
                
            except Exception as e:
                logger.error(f"통계 조회 중 오류: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """이미지 데이터를 OpenCV 형식으로 디코딩"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("유효하지 않은 이미지 형식입니다")
        
        return image
    
    def _extract_face_region(self, image: np.ndarray, face) -> np.ndarray:
        """이미지에서 얼굴 영역 추출"""
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
        
        # 경계 확인
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        face_region = image[y:y+h, x:x+w]
        return face_region
    
    def _face_to_dict(self, face) -> Dict[str, Any]:
        """Face 엔티티를 딕셔너리로 변환"""
        return {
            "face_id": face.face_id,
            "bbox": {
                "x": face.bbox.x,
                "y": face.bbox.y,
                "width": face.bbox.width,
                "height": face.bbox.height
            },
            "confidence": face.confidence,
            "quality_score": face.quality_score,
            "landmarks": face.landmarks,
            "created_at": face.created_at
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """API 서버 실행"""
        logger.info(f"Face Recognition API 서버 시작 - {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            log_level="info"
        )


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """FastAPI 앱 팩토리 함수"""
    api = FaceRecognitionAPI(config_path)
    return api.app


if __name__ == "__main__":
    # 개발용 서버 실행
    api = FaceRecognitionAPI()
    api.run(debug=True) 