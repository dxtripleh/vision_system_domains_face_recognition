#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPS 카운터 유틸리티.

실시간 처리에서 FPS를 계산하고 추적합니다.
"""

import time
from collections import deque
from typing import Optional

class FPSCounter:
    """FPS 카운터."""
    
    def __init__(self, window_size: int = 30):
        """
        FPS 카운터 초기화.
        
        Args:
            window_size: 평균 계산을 위한 윈도우 크기
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.last_time = time.time()
        self.fps = 0.0
    
    def tick(self) -> float:
        """
        새 프레임 틱을 기록하고 현재 FPS를 반환.
        
        Returns:
            현재 FPS
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) >= 2:
            # 윈도우 내 평균 FPS 계산
            time_span = self.timestamps[-1] - self.timestamps[0]
            frame_count = len(self.timestamps) - 1
            
            if time_span > 0:
                self.fps = frame_count / time_span
            else:
                self.fps = 0.0
        
        return self.fps
    
    def get_fps(self) -> float:
        """현재 FPS 반환."""
        return self.fps
    
    def reset(self):
        """카운터 리셋."""
        self.timestamps.clear()
        self.fps = 0.0 