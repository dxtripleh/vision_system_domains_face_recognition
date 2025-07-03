#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
입력 폴더에서 얼굴 검출 및 저장

이 스크립트는 captured와 uploads 폴더의 모든 이미지에서 얼굴을 검출하고,
detected_faces 폴더에 저장합니다. 이미 처리된 파일이라도 결과물이 없으면 재처리합니다.
"""

import os
import sys
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Any
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from common.logging import setup_logging
from domains.face_recognition.infrastructure.detection_engines.opencv_detection_engine import OpenCVDetectionEngine
from shared.vision_core.naming import UniversalNamingSystem, UniversalMetadata

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 경로 설정
CAPTURED_DIR = Path("data/domains/face_recognition/raw_input/captured")
UPLOADS_DIR = Path("data/domains/face_recognition/raw_input/uploads")
DETECTED_DIR = Path("data/domains/face_recognition/detected_faces")

def get_timestamp() -> str:
    """현재 타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def load_face_detector():
    """얼굴 검출기 로딩 (딥러닝 우선, OpenCV 백업)"""
    # 1차 선택: RetinaFace 계열 ONNX
    try:
        retinaface_models = [
            "models/weights/face_detection_retinaface_resnet50.onnx",
            "models/weights/face_detection_scrfd_10g_20250628.onnx"
        ]
        for model_path in retinaface_models:
            if Path(model_path).exists():
                try:
                    print(f"[1차] RetinaFace ONNX 모델 시도: {model_path}")
                    logger.info(f"[1차] RetinaFace ONNX 모델 시도: {model_path}")
                    detector = ONNXFaceDetector(model_path, model_type="retinaface")
                    print(f"[1차] RetinaFace 모델 로딩 성공: {model_path}")
                    logger.info(f"[1차] RetinaFace 모델 로딩 성공: {model_path}")
                    return detector
                except Exception as e:
                    print(f"[1차] RetinaFace 모델 {model_path} 로딩 실패: {e}")
                    logger.warning(f"[1차] RetinaFace 모델 {model_path} 로딩 실패: {e}")
                    continue
        print("[1차] 사용 가능한 RetinaFace 계열 모델 없음")
        logger.warning("[1차] 사용 가능한 RetinaFace 계열 모델 없음")
    except Exception as e:
        print(f"[1차] RetinaFace 로딩 실패: {e}")
        logger.warning(f"[1차] RetinaFace 로딩 실패: {e}")

    # 2차 선택: MobileFaceNet 계열 ONNX
    try:
        mobilefacenet_models = [
            "models/weights/face_recognition_mobilefacenet_20250628.onnx",
            "models/weights/face_detection_ultraface_rfb_320.onnx"
        ]
        for model_path in mobilefacenet_models:
            if Path(model_path).exists():
                try:
                    print(f"[2차] MobileFaceNet ONNX 모델 시도: {model_path}")
                    logger.info(f"[2차] MobileFaceNet ONNX 모델 시도: {model_path}")
                    detector = ONNXFaceDetector(model_path, model_type="mobilenet")
                    print(f"[2차] MobileFaceNet 모델 로딩 성공: {model_path}")
                    logger.info(f"[2차] MobileFaceNet 모델 로딩 성공: {model_path}")
                    return detector
                except Exception as e:
                    print(f"[2차] MobileFaceNet 모델 {model_path} 로딩 실패: {e}")
                    logger.warning(f"[2차] MobileFaceNet 모델 {model_path} 로딩 실패: {e}")
                    continue
        print("[2차] 사용 가능한 MobileFaceNet 계열 모델 없음")
        logger.warning("[2차] 사용 가능한 MobileFaceNet 계열 모델 없음")
    except Exception as e:
        print(f"[2차] MobileFaceNet 로딩 실패: {e}")
        logger.warning(f"[2차] MobileFaceNet 로딩 실패: {e}")

    # 백업: OpenCV Haar Cascade
    try:
        config = {
            'cascade_path': 'models/weights/haarcascade_frontalface_default.xml',
            'confidence_threshold': 0.3,
            'scale_factor': 1.05,
            'min_neighbors': 2,
            'min_size': (20, 20)
        }
        detector = OpenCVDetectionEngine(config)
        print("[백업] OpenCV Haar Cascade 사용")
        logger.info("[백업] OpenCV Haar Cascade 사용")
        return detector
    except Exception as e:
        print(f"[백업] OpenCV 로딩 실패: {e}")
        logger.error(f"[백업] OpenCV 로딩 실패: {e}")
    return None

class ONNXFaceDetector:
    """ONNX Runtime 기반 범용 얼굴 검출기"""
    
    def __init__(self, model_path: str, model_type: str = "retinaface"):
        try:
            import onnxruntime as ort
            self.model_path = model_path
            self.model_type = model_type
            
            # ONNX Runtime 세션 생성 (CPU 전용으로 안정화)
            providers = ['CPUExecutionProvider']
            
            # 세션 옵션 설정 (로그 레벨 낮춤)
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 2  # 경고만 출력
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # 입력/출력 정보 가져오기
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"ONNX 모델 로드 성공: {model_path}")
            print(f"입력 형태: {self.input_shape}")
            print(f"입력 이름: {self.input_name}")
            
        except ImportError:
            raise ImportError("onnxruntime이 설치되지 않았습니다. pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"ONNX 모델 로딩 실패: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """ONNX 모델로 얼굴 검출"""
        try:
            h, w = image.shape[:2]
            
            # 모델 타입에 따른 전처리
            if self.model_type == "retinaface":
                input_tensor = self._preprocess_retinaface(image)
            elif self.model_type == "mobilenet":
                input_tensor = self._preprocess_mobilenet(image)
            else:
                input_tensor = self._preprocess_default(image)
            
            # 추론 실행
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # 모델 타입에 따른 후처리
            if self.model_type == "retinaface":
                detections = self._postprocess_retinaface(outputs, w, h)
            elif self.model_type == "mobilenet":
                detections = self._postprocess_mobilenet(outputs, w, h)
            else:
                detections = self._postprocess_default(outputs, w, h)
            
            return detections
            
        except Exception as e:
            print(f"ONNX 검출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _preprocess_retinaface(self, image: np.ndarray) -> np.ndarray:
        """RetinaFace 전처리"""
        # 입력 형태에서 동적 차원('?' 문자열) 처리
        try:
            # 기본값 사용 (대부분의 RetinaFace는 640x640)
            target_size = 640
            
            # 입력 형태가 고정된 경우에만 확인
            if len(self.input_shape) > 2:
                height_dim = self.input_shape[2]
                width_dim = self.input_shape[3] if len(self.input_shape) > 3 else height_dim
                
                # 동적 차원이 아닌 경우에만 사용
                if isinstance(height_dim, int) and isinstance(width_dim, int):
                    target_size = max(height_dim, width_dim)
                    print(f"입력 크기 감지: {target_size}x{target_size}")
                else:
                    print(f"동적 입력 차원 감지 ('{height_dim}', '{width_dim}'), 기본값 {target_size} 사용")
            
        except Exception as e:
            print(f"입력 형태 분석 실패, 기본값 640 사용: {e}")
            target_size = 640
        
        # 이미지 리사이즈
        resized = cv2.resize(image, (target_size, target_size))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-1)
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가
        return np.expand_dims(transposed, axis=0)
    
    def _preprocess_mobilenet(self, image: np.ndarray) -> np.ndarray:
        """MobileFaceNet 계열 전처리"""
        # 기본값 사용
        target_size = 320
        
        try:
            # 입력 형태가 고정된 경우에만 확인
            if len(self.input_shape) > 2:
                height_dim = self.input_shape[2]
                if isinstance(height_dim, int):
                    target_size = height_dim
                    print(f"MobileFaceNet 입력 크기 감지: {target_size}x{target_size}")
                else:
                    print(f"MobileFaceNet 동적 입력 차원, 기본값 {target_size} 사용")
        except Exception as e:
            print(f"MobileFaceNet 입력 형태 분석 실패, 기본값 320 사용: {e}")
        
        # 이미지 리사이즈
        resized = cv2.resize(image, (target_size, target_size))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (-1 to 1 또는 0-1)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        
        # HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가
        return np.expand_dims(transposed, axis=0)
    
    def _preprocess_default(self, image: np.ndarray) -> np.ndarray:
        """기본 전처리"""
        target_size = 640
        resized = cv2.resize(image, (target_size, target_size))
        blob = cv2.dnn.blobFromImage(resized, 1.0/255.0, (target_size, target_size), (0, 0, 0), swapRB=True)
        return blob
    
    def _postprocess_retinaface(self, outputs: List[np.ndarray], w: int, h: int) -> List[Dict[str, Any]]:
        """RetinaFace 후처리(가장 간단한 접근)"""
        try:
            print(f"RetinaFace 출력 처리: {len(outputs)}개 출력")
            
            # 가장 간단한 접근: 모델이 이미 디코딩된 좌표를 출력한다고 가정
            confidence_threshold = 0.5
            nms_threshold = 0.4
            
            all_detections = []
            
            # 9개 출력이면 다중 스케일, 3개 출력이면 단일 스케일
            if len(outputs) == 9:
                # 다중 스케일: [scores_3개, boxes_3개, landmarks_3개]
                for scale_idx in range(3):
                    scores = outputs[scale_idx][0]  # (N, 1)
                    boxes = outputs[scale_idx + 3][0]  # (N, 4)
                    landmarks = outputs[scale_idx + 6][0] if len(outputs) > scale_idx + 6 else None  # (N, 10)
                    
                    print(f"스케일 {scale_idx}: scores={scores.shape}, boxes={boxes.shape}")
                    
                    # sigmoid 적용
                    if len(scores.shape) == 2 and scores.shape[1] == 1:
                        confidences = 1 / (1 + np.exp(-scores.flatten()))
                    else:
                        confidences = scores.flatten()
                    
                    # 신뢰도 필터링
                    valid_indices = np.where(confidences > confidence_threshold)[0]
                    print(f"  스케일 {scale_idx}: {len(valid_indices)}개 유효 검출")
                    
                    for idx in valid_indices[:5]:  # 상위 5개만
                        confidence = float(confidences[idx])
                        
                        # 박스 좌표 (이미 디코딩되었다고 가정)
                        box = boxes[idx]
                        
                        # 좌표가 0-1 범위인지 0-640 범위인지 확인
                        if np.max(box) <= 1.0:
                            # 0-1 범위면 원본 크기로 변환
                            x1, y1, x2, y2 = box * [w, h, w, h]
                        else:
                            # 0-640 범위면 원본 크기로 스케일링
                            x1, y1, x2, y2 = box * [w/640, h/640, w/640, h/640]
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 경계 보정
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        if x2 > x1 and y2 > y1:
                            bbox = [x1, y1, x2-x1, y2-y1]
                            
                            # 랜드마크
                            lm = None
                            if landmarks is not None and len(landmarks[idx]) >= 10:
                                landmark_pred = landmarks[idx]
                                lm = []
                                for j in range(0, 10, 2):
                                    if np.max(landmark_pred) <= 1.0:
                                        lx = int(landmark_pred[j] * w)
                                        ly = int(landmark_pred[j+1] * h)
                                    else:
                                        lx = int(landmark_pred[j] * w / 640)
                                        ly = int(landmark_pred[j+1] * h / 640)
                                    lm.append((lx, ly))
                            
                            all_detections.append({
                                'bbox': bbox,
                                'confidence': confidence,
                                'landmarks': lm,
                                'quality_score': confidence
                            })
                            
                            print(f"  ✓ 스케일 {scale_idx} 검출: bbox={bbox}, conf={confidence:.3f}")
            
            elif len(outputs) >= 3:
                # 단일 스케일: [boxes, scores, landmarks]
                boxes = outputs[0][0]  # (N, 4)
                scores = outputs[1][0]  # (N,) 또는 (N, 1)
                landmarks = outputs[2][0] if len(outputs) > 2 else None  # (N, 10)
                
                if len(scores.shape) == 2:
                    scores = scores[:, 0]
                
                # sigmoid 적용
                confidences = 1 / (1 + np.exp(-scores))
                
                # 신뢰도 필터링
                valid_indices = np.where(confidences > confidence_threshold)[0]
                print(f"단일 스케일: {len(valid_indices)}개 유효 검출")
                
                for idx in valid_indices[:10]:  # 상위 10개
                    confidence = float(confidences[idx])
                    
                    # 박스 좌표
                    box = boxes[idx]
                    
                    # 좌표 범위 확인 및 변환
                    if np.max(box) <= 1.0:
                        x1, y1, x2, y2 = box * [w, h, w, h]
                    else:
                        x1, y1, x2, y2 = box * [w/640, h/640, w/640, h/640]
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 경계 보정
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    if x2 > x1 and y2 > y1:
                        bbox = [x1, y1, x2-x1, y2-y1]
                        
                        # 랜드마크
                        lm = None
                        if landmarks is not None and len(landmarks[idx]) >= 10:
                            landmark_pred = landmarks[idx]
                            lm = []
                            for j in range(0, 10, 2):
                                if np.max(landmark_pred) <= 1.0:
                                    lx = int(landmark_pred[j] * w)
                                    ly = int(landmark_pred[j+1] * h)
                                else:
                                    lx = int(landmark_pred[j] * w / 640)
                                    ly = int(landmark_pred[j+1] * h / 640)
                                lm.append((lx, ly))
                        
                        all_detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'landmarks': lm,
                            'quality_score': confidence
                        })
                        
                        print(f"  ✓ 단일 스케일 검출: bbox={bbox}, conf={confidence:.3f}")
            
            else:
                print(f"지원하지 않는 출력 형태: {len(outputs)}개")
                return []
            
            # NMS 적용
            if all_detections:
                final_detections = self._nms(all_detections, nms_threshold)
                print(f"RetinaFace 최종 검출: {len(final_detections)}개")
                return final_detections
            else:
                print("RetinaFace 검출 결과 없음")
                return []
                
        except Exception as e:
            print(f"RetinaFace 후처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _nms(self, detections: List[Dict[str, Any]], nms_threshold: float) -> List[Dict[str, Any]]:
        """Non-Maximum Suppression (NMS)"""
        if not detections:
            return []
        # 신뢰도 기준 내림차순 정렬
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self._iou(best['bbox'], d['bbox']) < nms_threshold]
        return keep

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        if xB <= xA or yB <= yA:
            return 0.0
        interArea = (xB - xA) * (yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
        return iou
    
    def _postprocess_mobilenet(self, outputs: List[np.ndarray], w: int, h: int) -> List[Dict[str, Any]]:
        """MobileFaceNet 계열 후처리"""
        detections = []
        
        try:
            # UltraFace 등의 출력 형식
            if len(outputs) >= 1:
                output = outputs[0]
                
                # 출력 형태에 따라 처리
                if len(output.shape) == 3:  # (1, N, 4+1+...)
                    for detection in output[0]:
                        if len(detection) >= 5:
                            # numpy 배열에서 안전하게 스칼라 추출
                            confidence = detection[4].item() if hasattr(detection[4], 'item') else float(detection[4])
                            if confidence > 0.5:
                                x1, y1, x2, y2 = detection[:4]
                                
                                # 좌표 변환
                                if x1 <= 1.0:  # 정규화된 좌표
                                    x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                                
                                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                                
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'landmarks': None,
                                    'quality_score': confidence
                                })
        except Exception as e:
            print(f"MobileFaceNet 후처리 오류: {e}")
        
        return detections
    
    def _postprocess_default(self, outputs: List[np.ndarray], w: int, h: int) -> List[Dict[str, Any]]:
        """기본 후처리"""
        detections = []
        
        try:
            if len(outputs) >= 1:
                output = outputs[0]
                print(f"기본 후처리 - 출력 형태: {output.shape}")
                
                # 간단한 형태로 파싱 시도
                if len(output.shape) >= 2:
                    for i in range(min(output.shape[1], 100)):  # 최대 100개까지
                        if len(output[0]) > i and len(output[0][i]) >= 5:
                            detection = output[0][i]
                            # numpy 배열에서 안전하게 스칼라 추출
                            confidence_val = detection[4] if len(detection) > 4 else detection[-1]
                            confidence = confidence_val.item() if hasattr(confidence_val, 'item') else float(confidence_val)
                            
                            if confidence > 0.3:
                                x1, y1, x2, y2 = detection[:4]
                                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                                
                                detections.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'landmarks': None,
                                    'quality_score': confidence
                                })
        except Exception as e:
            print(f"기본 후처리 오류: {e}")
        
        return detections

def get_all_input_files() -> List[Path]:
    """모든 입력 파일 목록 가져오기"""
    input_files = []
    
    # captured 폴더의 이미지 파일들 (하위 폴더 포함)
    if CAPTURED_DIR.exists():
        # 하위 폴더까지 모든 이미지 파일 검색
        for file_path in CAPTURED_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                input_files.append(file_path)
                print(f"발견된 이미지: {file_path}")
    
    # uploads 폴더의 이미지 파일들 (하위 폴더 포함)
    if UPLOADS_DIR.exists():
        for file_path in UPLOADS_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                input_files.append(file_path)
                print(f"발견된 이미지: {file_path}")
    
    return input_files

def get_existing_detections() -> Dict[str, List[Path]]:
    """기존 검출 결과 파일들 매핑 (원본 파일명 -> 검출 파일들)"""
    detection_map = {}
    
    if not DETECTED_DIR.exists():
        return detection_map
    
    # 모든 날짜 폴더 검색
    for date_dir in DETECTED_DIR.glob("*"):
        if date_dir.is_dir():
            for json_file in date_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    # 원본 파일명 추출
                    source_file = meta.get('source_file', '')
                    if source_file:
                        if source_file not in detection_map:
                            detection_map[source_file] = []
                        
                        # 대응하는 이미지 파일도 찾기
                        img_file = json_file.with_suffix('.jpg')
                        if img_file.exists():
                            detection_map[source_file].append(img_file)
                        
                except Exception as e:
                    logger.warning(f"메타데이터 읽기 실패: {json_file} - {e}")
    
    return detection_map

def should_process_file(file_path: Path, existing_detections: Dict[str, List[Path]]) -> bool:
    """파일을 처리해야 하는지 판단"""
    filename = file_path.name
    
    # 임시로 모든 파일을 처리 (기존 결과 무시)
    logger.info(f"처리 대상: {filename}")
    return True
    
    # 기존 검출 결과가 있는지 확인
    if filename in existing_detections:
        detection_files = existing_detections[filename]
        if detection_files:
            logger.debug(f"이미 처리됨 (검출 결과 {len(detection_files)}개): {filename}")
            return False
    
    logger.info(f"처리 대상: {filename}")
    return True

def save_detected_face(face_img, bbox, confidence, source_path, sequence):
    """검출된 얼굴 저장"""
    ts = get_timestamp()
    # 날짜별 폴더 생성
    date_folder = ts[:8]  # YYYYMMDD
    date_dir = DETECTED_DIR / date_folder
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # 범용 네이밍 시스템 사용
    filename = UniversalNamingSystem.create_detection_filename(
        domain='face_recognition',
        timestamp=ts,
        sequence=sequence,
        confidence=confidence,
        original_filename=source_path.name
    )
    
    jsonname = filename.replace('.jpg', '.json')
    img_path = date_dir / filename
    json_path = date_dir / jsonname
    
    # 얼굴 이미지 저장
    cv2.imwrite(str(img_path), face_img)
    
    # 메타데이터 생성 및 저장
    detection_meta = UniversalMetadata.create_detection_metadata(
        domain='face_recognition',
        object_id=f"{ts[:14]}_{sequence:02d}",
        confidence=confidence,
        bbox=bbox,
        original_capture=source_path.name
    )
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_meta, f, ensure_ascii=False, indent=2)
    
    logger.info(f"검출 얼굴 저장: {img_path}")
    return img_path, json_path

def process_image(detector, img_path: Path) -> int:
    """이미지에서 얼굴 검출 및 저장"""
    try:
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"이미지 로드 실패: {img_path}")
            return 0
        
        # 얼굴 검출
        faces = detector.detect(img)
        
        if not faces:
            logger.info(f"{img_path.name}: 얼굴 없음")
            return 0
        
        # 검출된 얼굴들 저장
        face_count = 0
        for i, face in enumerate(faces):
            # OpenCV 검출 결과 처리 (딕셔너리 형태)
            if isinstance(face, dict):
                bbox = face['bbox']
                confidence = face.get('confidence', 0.8)
            else:
                # 리스트 형태 (x, y, w, h)
                bbox = [int(x) for x in face]
                confidence = 0.8
            
            # numpy 타입을 일반 Python 타입으로 변환
            bbox = [int(x) for x in bbox]
            confidence = float(confidence)
            
            # 얼굴 영역 추출
            x, y, w, h = bbox
            if x >= 0 and y >= 0 and x+w <= img.shape[1] and y+h <= img.shape[0]:
                face_img = img[y:y+h, x:x+w]
                
                # 얼굴 이미지 저장
                save_detected_face(face_img, bbox, confidence, img_path, i + 1)
                face_count += 1
        
        logger.info(f"{img_path.name}: {face_count}개 얼굴 검출")
        return face_count
        
    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {img_path} - {e}")
        return 0

def main():
    """메인 함수"""
    print("=== 얼굴 검출 시작 ===")
    logger.info("=== 얼굴 검출 시작 ===")
    
    # 얼굴 검출기 로딩
    detector = load_face_detector()
    if detector is None:
        print("얼굴 검출기를 로딩할 수 없습니다.")
        logger.error("얼굴 검출기를 로딩할 수 없습니다.")
        return
    
    # 모든 입력 파일 가져오기
    all_input_files = get_all_input_files()
    print(f"총 입력 파일: {len(all_input_files)}개")
    logger.info(f"총 입력 파일: {len(all_input_files)}개")
    
    if not all_input_files:
        print("처리할 파일이 없습니다.")
        logger.info("처리할 파일이 없습니다.")
        return
    
    # 기존 검출 결과 확인
    existing_detections = get_existing_detections()
    print(f"기존 검출 결과: {len(existing_detections)}개 파일")
    logger.info(f"기존 검출 결과: {len(existing_detections)}개 파일")
    
    # 파일별 처리
    total_processed = 0
    total_faces = 0
    
    for img_path in all_input_files:
        if should_process_file(img_path, existing_detections):
            print(f"처리 중: {img_path.name}")
            face_count = process_image(detector, img_path)
            total_processed += 1
            total_faces += face_count
    
    print(f"=== 얼굴 검출 완료 ===")
    print(f"처리된 파일: {total_processed}개")
    print(f"검출된 얼굴: {total_faces}개")
    logger.info(f"=== 얼굴 검출 완료 ===")
    logger.info(f"처리된 파일: {total_processed}개")
    logger.info(f"검출된 얼굴: {total_faces}개")

if __name__ == "__main__":
    main() 