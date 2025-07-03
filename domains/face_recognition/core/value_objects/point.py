#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point Value Object.

2D 점을 나타내는 값 객체입니다.
"""

from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Point:
    """
    2D 점 값 객체
    
    x, y 좌표를 가지는 2차원 점을 안전하게 관리합니다.
    """
    
    x: float
    y: float
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not isinstance(self.x, (int, float)):
            raise TypeError(f"X coordinate must be numeric, got {type(self.x)}")
        if not isinstance(self.y, (int, float)):
            raise TypeError(f"Y coordinate must be numeric, got {type(self.y)}")
        
        if math.isnan(self.x) or math.isinf(self.x):
            raise ValueError(f"X coordinate must be finite, got {self.x}")
        if math.isnan(self.y) or math.isinf(self.y):
            raise ValueError(f"Y coordinate must be finite, got {self.y}")
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        """좌표 튜플 (x, y)"""
        return (self.x, self.y)
    
    @property
    def int_coordinates(self) -> Tuple[int, int]:
        """정수 좌표 튜플 (int(x), int(y))"""
        return (int(self.x), int(self.y))
    
    def distance_to(self, other: 'Point') -> float:
        """
        다른 점까지의 거리 계산
        
        Args:
            other: 대상 점
            
        Returns:
            float: 유클리드 거리
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def distance_to_origin(self) -> float:
        """원점(0, 0)까지의 거리"""
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def manhattan_distance_to(self, other: 'Point') -> float:
        """
        다른 점까지의 맨하탄 거리 계산
        
        Args:
            other: 대상 점
            
        Returns:
            float: 맨하탄 거리
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def angle_to(self, other: 'Point') -> float:
        """
        다른 점까지의 각도 계산 (라디안)
        
        Args:
            other: 대상 점
            
        Returns:
            float: 각도 (라디안, -π ~ π)
        """
        dx = other.x - self.x
        dy = other.y - self.y
        return math.atan2(dy, dx)
    
    def angle_to_degrees(self, other: 'Point') -> float:
        """
        다른 점까지의 각도 계산 (도)
        
        Args:
            other: 대상 점
            
        Returns:
            float: 각도 (도, -180 ~ 180)
        """
        return math.degrees(self.angle_to(other))
    
    def midpoint_with(self, other: 'Point') -> 'Point':
        """
        다른 점과의 중점 계산
        
        Args:
            other: 대상 점
            
        Returns:
            Point: 중점
        """
        mid_x = (self.x + other.x) / 2
        mid_y = (self.y + other.y) / 2
        return Point(x=mid_x, y=mid_y)
    
    def translate(self, dx: float, dy: float) -> 'Point':
        """
        점을 평행이동
        
        Args:
            dx: x축 이동량
            dy: y축 이동량
            
        Returns:
            Point: 이동된 점
        """
        return Point(x=self.x + dx, y=self.y + dy)
    
    def scale(self, factor: float, origin: 'Point' = None) -> 'Point':
        """
        점을 스케일링
        
        Args:
            factor: 스케일 팩터
            origin: 스케일링 기준점 (None이면 원점)
            
        Returns:
            Point: 스케일링된 점
        """
        if origin is None:
            origin = Point(0, 0)
        
        # 기준점을 원점으로 이동 -> 스케일링 -> 기준점으로 복원
        translated_x = self.x - origin.x
        translated_y = self.y - origin.y
        
        scaled_x = translated_x * factor + origin.x
        scaled_y = translated_y * factor + origin.y
        
        return Point(x=scaled_x, y=scaled_y)
    
    def rotate(self, angle_radians: float, origin: 'Point' = None) -> 'Point':
        """
        점을 회전
        
        Args:
            angle_radians: 회전 각도 (라디안)
            origin: 회전 중심점 (None이면 원점)
            
        Returns:
            Point: 회전된 점
        """
        if origin is None:
            origin = Point(0, 0)
        
        # 회전 중심을 원점으로 이동
        translated_x = self.x - origin.x
        translated_y = self.y - origin.y
        
        # 회전 변환
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        # 회전 중심으로 복원
        final_x = rotated_x + origin.x
        final_y = rotated_y + origin.y
        
        return Point(x=final_x, y=final_y)
    
    def is_inside_rectangle(self, top_left: 'Point', bottom_right: 'Point') -> bool:
        """
        사각형 내부에 있는지 확인
        
        Args:
            top_left: 사각형 왼쪽 위 점
            bottom_right: 사각형 오른쪽 아래 점
            
        Returns:
            bool: 내부에 있으면 True
        """
        return (top_left.x <= self.x <= bottom_right.x and 
                top_left.y <= self.y <= bottom_right.y)
    
    def is_inside_circle(self, center: 'Point', radius: float) -> bool:
        """
        원 내부에 있는지 확인
        
        Args:
            center: 원의 중심점
            radius: 원의 반지름
            
        Returns:
            bool: 내부에 있으면 True
        """
        distance = self.distance_to(center)
        return distance <= radius
    
    def round(self, decimals: int = 0) -> 'Point':
        """
        좌표를 반올림
        
        Args:
            decimals: 소수점 자릿수
            
        Returns:
            Point: 반올림된 점
        """
        rounded_x = round(self.x, decimals)
        rounded_y = round(self.y, decimals)
        return Point(x=rounded_x, y=rounded_y)
    
    def floor(self) -> 'Point':
        """
        좌표를 내림
        
        Returns:
            Point: 내림된 점
        """
        return Point(x=math.floor(self.x), y=math.floor(self.y))
    
    def ceil(self) -> 'Point':
        """
        좌표를 올림
        
        Returns:
            Point: 올림된 점
        """
        return Point(x=math.ceil(self.x), y=math.ceil(self.y))
    
    def to_tuple(self) -> Tuple[float, float]:
        """
        튜플로 변환
        
        Returns:
            Tuple[float, float]: (x, y) 튜플
        """
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        """
        정수 튜플로 변환
        
        Returns:
            Tuple[int, int]: (int(x), int(y)) 튜플
        """
        return (int(self.x), int(self.y))
    
    def to_list(self) -> List[float]:
        """
        리스트로 변환
        
        Returns:
            List[float]: [x, y] 리스트
        """
        return [self.x, self.y]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환
        
        Returns:
            Dict[str, Any]: 점 정보
        """
        return {
            'x': self.x,
            'y': self.y,
            'coordinates': self.coordinates,
            'distance_to_origin': self.distance_to_origin()
        }
    
    @classmethod
    def origin(cls) -> 'Point':
        """원점 (0, 0) 생성"""
        return cls(x=0.0, y=0.0)
    
    @classmethod
    def from_tuple(cls, coordinates: Tuple[float, float]) -> 'Point':
        """
        튜플에서 점 생성
        
        Args:
            coordinates: (x, y) 튜플
            
        Returns:
            Point: 생성된 점
        """
        x, y = coordinates
        return cls(x=x, y=y)
    
    @classmethod
    def from_list(cls, coordinates: List[float]) -> 'Point':
        """
        리스트에서 점 생성
        
        Args:
            coordinates: [x, y] 리스트
            
        Returns:
            Point: 생성된 점
        """
        if len(coordinates) != 2:
            raise ValueError(f"List must have exactly 2 elements, got {len(coordinates)}")
        
        return cls(x=coordinates[0], y=coordinates[1])
    
    def __add__(self, other: 'Point') -> 'Point':
        """점 덧셈"""
        return Point(x=self.x + other.x, y=self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        """점 뺄셈"""
        return Point(x=self.x - other.x, y=self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        """스칼라 곱셈"""
        return Point(x=self.x * scalar, y=self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Point':
        """스칼라 나눗셈"""
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Point(x=self.x / scalar, y=self.y / scalar)
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"({self.x:.1f}, {self.y:.1f})"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현"""
        return f"Point(x={self.x}, y={self.y})" 