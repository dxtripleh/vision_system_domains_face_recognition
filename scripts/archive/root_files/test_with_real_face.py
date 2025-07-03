#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실제 얼굴 이미지로 모델 테스트
"""

import cv2
import numpy as np
from pathlib import Path

def test_with_real_face():
    """실제 얼굴 이미지로 테스트"""
    
    # 카메라에서 실제 얼굴 이미지 캡처
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    print("📷 카메라에서 얼굴을 캡처합니다. 3초 후 촬영됩니다...")
    
    # 3초 대기
    for i in range(3, 0, -1):
        print(f"{i}...")
        cv2.waitKey(1000)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 이미지를 캡처할 수 없습니다.")
        return
    
    print(f"✅ 이미지 캡처 완료: {frame.shape}")
    
    # 이미지 저장
    cv2.imwrite("test_face.jpg", frame)
    print("💾 테스트 이미지 저장: test_face.jpg")
    
    # 모델 테스트
    models = [
        ('YuNet', 'face_detection_yunet_2023mar.onnx', 'yunet'),
        ('UltraFace', 'ultraface_rfb_320_robust.onnx', 'ultraface'),
        ('RetinaFace', 'face_detection_retinaface_resnet50.onnx', 'retinaface'),
        ('SCRFD', 'face_detection_scrfd_10g_20250628.onnx', 'scrfd')
    ]
    
    for name, model_file, model_type in models:
        print(f"\n{'='*30}")
        print(f"테스트: {name}")
        
        model_path = Path('models/weights') / model_file
        
        if not model_path.exists():
            print(f"❌ 파일 없음: {model_file}")
            continue
        
        try:
            # 모델 로드
            net = cv2.dnn.readNetFromONNX(str(model_path))
            print(f"✅ 모델 로드 성공")
            
            # 전처리
            if model_type == 'yunet':
                blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (0, 0, 0), True, False)
            elif model_type == 'ultraface':
                resized = cv2.resize(frame, (320, 240))
                blob = resized.astype(np.float32)
                blob = (blob - 127.0) / 127.0
                blob = blob.transpose(2, 0, 1)[np.newaxis, :]
            else:  # retinaface, scrfd
                blob = cv2.dnn.blobFromImage(frame, 1.0, (640, 640), (0, 0, 0), True, False)
            
            # 추론
            net.setInput(blob)
            outputs = net.forward()
            
            print(f"출력 shape: {[out.shape for out in outputs]}")
            
            # 검출 결과 확인
            detections = []
            
            if model_type == 'yunet' and len(outputs) > 0:
                output = outputs[0]
                print(f"YuNet 출력 샘플: {output[0] if len(output) > 0 else 'None'}")
                
                for detection in output:
                    if len(detection) >= 10:  # YuNet은 10개 값
                        confidence = detection[4] if len(detection) > 4 else 0.5
                        if confidence > 0.1:
                            detections.append({
                                'bbox': [0, 0, 100, 100],  # 임시 값
                                'confidence': float(confidence),
                                'model_type': 'yunet'
                            })
            
            elif model_type == 'ultraface' and len(outputs) > 0:
                boxes = outputs[0]
                print(f"UltraFace 출력 샘플: {boxes[0] if len(boxes) > 0 else 'None'}")
                
                # 상위 5개만 사용
                for i, box in enumerate(boxes[:5]):
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[:4]
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:
                            detections.append({
                                'bbox': [int(x1 * 640 / 320), int(y1 * 480 / 240), 
                                        int(w * 640 / 320), int(h * 480 / 240)],
                                'confidence': 0.8,
                                'model_type': 'ultraface'
                            })
            
            else:  # retinaface, scrfd
                for output in outputs:
                    if len(output.shape) == 2 and output.shape[1] >= 5:
                        print(f"{name} 출력 샘플: {output[0] if len(output) > 0 else 'None'}")
                        
                        for detection in output:
                            if len(detection) >= 5:
                                confidence = detection[4]
                                if confidence > 0.1:
                                    detections.append({
                                        'bbox': [0, 0, 100, 100],  # 임시 값
                                        'confidence': float(confidence),
                                        'model_type': model_type
                                    })
            
            print(f"🔍 검출된 얼굴: {len(detections)}개")
            
            # 결과 시각화
            result_image = frame.copy()
            for i, det in enumerate(detections):
                bbox = det['bbox']
                confidence = det['confidence']
                model_type = det['model_type']
                
                x, y, w, h = bbox
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"{model_type}: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 결과 저장
            output_file = f"result_{model_type}.jpg"
            cv2.imwrite(output_file, result_image)
            print(f"💾 결과 저장: {output_file}")
            
        except Exception as e:
            print(f"❌ 오류: {str(e)}")

if __name__ == "__main__":
    test_with_real_face() 