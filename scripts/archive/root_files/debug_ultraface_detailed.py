#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UltraFace 모델 출력 상세 분석
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_ultraface_output():
    """UltraFace 모델 출력 상세 분석"""
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    model_path = Path('models/weights/ultraface_rfb_320_robust.onnx')
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 존재하지 않음: {model_path}")
        return
    
    try:
        # 모델 로드
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print(f"✅ UltraFace 모델 로드 성공")
        
        # UltraFace 전처리
        resized = cv2.resize(test_image, (320, 240))
        blob = resized.astype(np.float32)
        blob = (blob - 127.0) / 127.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]
        
        print(f"📊 입력 blob shape: {blob.shape}")
        
        # 추론
        net.setInput(blob)
        outputs = net.forward()
        
        print(f"📊 출력 개수: {len(outputs)}")
        
        # 각 출력 분석
        for i, output in enumerate(outputs):
            print(f"\n출력 {i}:")
            print(f"  shape: {output.shape}")
            print(f"  dtype: {output.dtype}")
            print(f"  min: {output.min()}")
            print(f"  max: {output.max()}")
            print(f"  mean: {output.mean()}")
            
            # 첫 몇 개 값 출력
            if len(output.shape) == 1:
                print(f"  첫 10개 값: {output[:10]}")
            elif len(output.shape) == 2:
                print(f"  첫 3행: {output[:3]}")
            elif len(output.shape) == 3:
                print(f"  첫 3개: {output[:3, :3] if output.shape[1] >= 3 and output.shape[2] >= 3 else output[:3]}")
        
        # UltraFace 출력 해석 시도
        print(f"\n{'='*50}")
        print("UltraFace 출력 해석:")
        
        if len(outputs) >= 2:
            # 첫 번째 출력이 박스, 두 번째 출력이 점수일 가능성
            boxes_output = outputs[0]
            scores_output = outputs[1]
            
            print(f"박스 출력 shape: {boxes_output.shape}")
            print(f"점수 출력 shape: {scores_output.shape}")
            
            # 차원 정규화
            if len(boxes_output.shape) == 3:
                boxes = boxes_output[0]  # [1, N, 4] -> [N, 4]
            else:
                boxes = boxes_output
                
            if len(scores_output.shape) == 3:
                scores = scores_output[0]  # [1, N, 2] -> [N, 2]
            else:
                scores = scores_output
            
            print(f"정규화된 박스 shape: {boxes.shape}")
            print(f"정규화된 점수 shape: {scores.shape}")
            
            # 높은 신뢰도 검출 찾기
            if len(scores.shape) == 2 and scores.shape[1] >= 2:
                face_scores = scores[:, 1]  # 얼굴 클래스 점수
                high_conf_indices = np.where(face_scores > 0.5)[0]
                
                print(f"높은 신뢰도 검출 개수: {len(high_conf_indices)}")
                
                if len(high_conf_indices) > 0:
                    for idx in high_conf_indices[:3]:  # 처음 3개만
                        print(f"  검출 {idx}: 점수={face_scores[idx]:.3f}, 박스={boxes[idx]}")
        
    except Exception as e:
        print(f"❌ UltraFace 분석 실패: {e}")

if __name__ == "__main__":
    analyze_ultraface_output() 