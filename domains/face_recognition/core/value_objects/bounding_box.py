#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bounding Box Value Object.

얼굴의 위치를 나타내는 바운딩 박스 값 객체입니다.
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """
    바운딩 박스 값 객체
    
    얼굴이나 객체의 위치를 나타내는 사각형 영역을 정의합니다.
    """
    
    x: int
    y: int  
    width: int
    height: int
    
    def __post_init__(self):
        """초기화 후 검증"""
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.x < 0:
            raise ValueError(f"X coordinate must be non-negative, got {self.x}")
        if self.y < 0:
            raise ValueError(f"Y coordinate must be non-negative, got {self.y}")
    
    @property
    def right(self) -> int:
        """오른쪽 경계 x 좌표"""
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        """아래쪽 경계 y 좌표"""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """중심점 x 좌표"""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """중심점 y 좌표"""
        return self.y + self.height / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        """중심점 좌표 (x, y)"""
        return (self.center_x, self.center_y)
    
    @property
    def area(self) -> int:
        """바운딩 박스의 면적"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """종횡비 (width / height)"""
        return self.width / self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        """
        주어진 점이 바운딩 박스 내부에 있는지 확인
        
        Args:
            x: 점의 x 좌표
            y: 점의 y 좌표
            
        Returns:
            bool: 점이 내부에 있으면 True
        """
        return (self.x <= x < self.right and 
                self.y <= y < self.bottom)
    
    def overlaps_with(self, other: 'BoundingBox') -> bool:
        """
        다른 바운딩 박스와 겹치는지 확인
        
        Args:
            other: 비교할 바운딩 박스
            
        Returns:
            bool: 겹치면 True
        """
        return not (self.right <= other.x or 
                   other.right <= self.x or
                   self.bottom <= other.y or 
                   other.bottom <= self.y)
    
    def intersection_with(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        다른 바운딩 박스와의 교집합 영역 계산
        
        Args:
            other: 교집합을 계산할 바운딩 박스
            
        Returns:
            BoundingBox: 교집합 영역 (겹치지 않으면 면적이 0인 박스)
        """
        if not self.overlaps_with(other):
            return BoundingBox(x=0, y=0, width=0, height=0)
        
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        
        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top
        )
    
    def union_with(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        다른 바운딩 박스와의 합집합 영역 계산
        
        Args:
            other: 합집합을 계산할 바운딩 박스
            
        Returns:
            BoundingBox: 두 박스를 모두 포함하는 최소 바운딩 박스
        """
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        
        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top
        )
    
    def iou_with(self, other: 'BoundingBox') -> float:
        """
        다른 바운딩 박스와의 IoU (Intersection over Union) 계산
        
        Args:
            other: IoU를 계산할 바운딩 박스
            
        Returns:
            float: IoU 값 (0.0 ~ 1.0)
        """
        intersection = self.intersection_with(other)
        if intersection.area == 0:
            return 0.0
        
        union_area = self.area + other.area - intersection.area
        return intersection.area / union_area if union_area > 0 else 0.0
    
    def scale(self, scale_factor: float) -> 'BoundingBox':
        """
        바운딩 박스를 스케일링
        
        Args:
            scale_factor: 스케일 팩터
            
        Returns:
            BoundingBox: 스케일링된 바운딩 박스
        """
        if scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {scale_factor}")
        
        return BoundingBox(
            x=int(self.x * scale_factor),
            y=int(self.y * scale_factor),
            width=int(self.width * scale_factor),
            height=int(self.height * scale_factor)
        )
    
    def expand(self, margin: int) -> 'BoundingBox':
        """
        바운딩 박스를 마진만큼 확장
        
        Args:
            margin: 확장할 마진 (픽셀)
            
        Returns:
            BoundingBox: 확장된 바운딩 박스
        """
        return BoundingBox(
            x=max(0, self.x - margin),
            y=max(0, self.y - margin),
            width=self.width + 2 * margin,
            height=self.height + 2 * margin
        )
    
    def fit_to_image(self, image_width: int, image_height: int) -> 'BoundingBox':
        """
        바운딩 박스를 이미지 경계에 맞춤
        
        Args:
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            BoundingBox: 이미지 경계에 맞춰진 바운딩 박스
        """
        # 경계 조정
        x = max(0, min(self.x, image_width - 1))
        y = max(0, min(self.y, image_height - 1))
        right = max(x + 1, min(self.right, image_width))
        bottom = max(y + 1, min(self.bottom, image_height))
        
        return BoundingBox(
            x=x,
            y=y,
            width=right - x,
            height=bottom - y
        )
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """
        (x1, y1, x2, y2) 형식으로 변환
        
        Returns:
            Tuple[int, int, int, int]: (x1, y1, x2, y2) 좌표
        """
        return (self.x, self.y, self.right, self.bottom)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """
        (x, y, width, height) 형식으로 변환
        
        Returns:
            Tuple[int, int, int, int]: (x, y, w, h) 좌표
        """
        return (self.x, self.y, self.width, self.height)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리 형식으로 변환
        
        Returns:
            Dict[str, Any]: 바운딩 박스 정보
        """
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'right': self.right,
            'bottom': self.bottom,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio
        }
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> 'BoundingBox':
        """
        (x1, y1, x2, y2) 좌표에서 바운딩 박스 생성
        
        Args:
            x1: 왼쪽 위 x 좌표
            y1: 왼쪽 위 y 좌표  
            x2: 오른쪽 아래 x 좌표
            y2: 오른쪽 아래 y 좌표
            
        Returns:
            BoundingBox: 생성된 바운딩 박스
        """
        return cls(
            x=min(x1, x2),
            y=min(y1, y2),
            width=abs(x2 - x1),
            height=abs(y2 - y1)
        )
    
    @classmethod
    def from_center(cls, center_x: float, center_y: float, 
                   width: int, height: int) -> 'BoundingBox':
        """
        중심점과 크기로부터 바운딩 박스 생성
        
        Args:
            center_x: 중심점 x 좌표
            center_y: 중심점 y 좌표
            width: 너비
            height: 높이
            
        Returns:
            BoundingBox: 생성된 바운딩 박스
        """
        x = int(center_x - width / 2)
        y = int(center_y - height / 2)
        
        return cls(x=x, y=y, width=width, height=height)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현"""
        return (f"BoundingBox(x={self.x}, y={self.y}, "
               f"width={self.width}, height={self.height}, "
               f"area={self.area})") 