#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 얼굴검출 데모.

OpenCV Haar Cascade를 사용한 기본적인 얼굴검출 데모입니다.
"""

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

def create_required_directories():
    """필요한 디렉토리 생성"""
    directories = ["data/output", "data/logs", "data/temp"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """메인 함수"""
    print("🎯 간단한 얼굴검출 데모")
    print("=" * 50)
    print("OpenCV Haar Cascade를 사용합니다")
    print("키보드 조작:")
    print("  'q' - 종료")
    print("  's' - 스크린샷 저장")
    print("  'i' - 정보 표시 토글")
    print("=" * 50)
    
    # 필요한 디렉토리 생성
    create_required_directories()
    
    # Haar Cascade 분류기 초기화
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Haar Cascade 로드 실패")
            return 1
        print("✅ Haar Cascade 로드 성공")
    except Exception as e:
        print(f"❌ Haar Cascade 초기화 오류: {str(e)}")
        return 1
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다")
        print("카메라가 연결되어 있는지 확인하세요")
        return 1
    
    print("✅ 카메라 초기화 성공")
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 성능 측정 변수
    frame_count = 0
    start_time = time.time()
    show_info = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다")
                break
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 검출된 얼굴 그리기
            for i, (x, y, w, h) in enumerate(faces):
                # 바운딩 박스
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 얼굴 번호
                cv2.putText(frame, f"Face {i+1}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 중심점
                center_x, center_y = x + w//2, y + h//2
                cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)
            
            # FPS 계산
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 정보 표시
            if show_info:
                # 배경 박스
                cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
                
                # 텍스트 정보
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces: {len(faces)}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 프레임 표시
            cv2.imshow('Simple Face Detection Demo', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 종료 요청")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"data/output/simple_demo_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 스크린샷 저장: {filename}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"ℹ️ 정보 표시: {'ON' if show_info else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다")
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 데모 종료")
        
        # 최종 통계
        total_time = time.time() - start_time
        print(f"📊 총 실행 시간: {total_time:.1f}초")
        print(f"📊 총 프레임 수: {frame_count}")
        print(f"📊 평균 FPS: {frame_count/total_time:.1f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 