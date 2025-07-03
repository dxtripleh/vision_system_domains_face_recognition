#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Confidence Score Value Object.

신뢰도 점수를 나타내는 값 객체입니다.
"""

from typing import Dict, Any, Union
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ConfidenceScore:
    """
    신뢰도 점수 값 객체
    
    0.0 ~ 1.0 범위의 신뢰도 점수를 안전하게 관리합니다.
    """
    
    value: float
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Confidence value must be numeric, got {type(self.value)}")
        
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.value}")
        
        if math.isnan(self.value) or math.isinf(self.value):
            raise ValueError(f"Confidence must be a finite number, got {self.value}")
    
    @property
    def percentage(self) -> float:
        """신뢰도를 백분율로 반환"""
        return self.value * 100.0
    
    @property
    def is_high(self) -> bool:
        """높은 신뢰도인지 확인 (>= 0.8)"""
        return self.value >= 0.8
    
    @property
    def is_medium(self) -> bool:
        """중간 신뢰도인지 확인 (0.5 <= x < 0.8)"""
        return 0.5 <= self.value < 0.8
    
    @property
    def is_low(self) -> bool:
        """낮은 신뢰도인지 확인 (< 0.5)"""
        return self.value < 0.5
    
    @property
    def confidence_level(self) -> str:
        """신뢰도 수준을 문자열로 반환"""
        if self.is_high:
            return "high"
        elif self.is_medium:
            return "medium"
        else:
            return "low"
    
    def meets_threshold(self, threshold: float) -> bool:
        """
        주어진 임계값을 만족하는지 확인
        
        Args:
            threshold: 임계값 (0.0 ~ 1.0)
            
        Returns:
            bool: 임계값 이상이면 True
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        return self.value >= threshold
    
    def distance_from(self, other: 'ConfidenceScore') -> float:
        """
        다른 신뢰도 점수와의 거리 계산
        
        Args:
            other: 비교할 신뢰도 점수
            
        Returns:
            float: 절대 거리
        """
        return abs(self.value - other.value)
    
    def is_close_to(self, other: 'ConfidenceScore', tolerance: float = 0.01) -> bool:
        """
        다른 신뢰도 점수와 유사한지 확인
        
        Args:
            other: 비교할 신뢰도 점수
            tolerance: 허용 오차
            
        Returns:
            bool: 허용 오차 내에 있으면 True
        """
        return self.distance_from(other) <= tolerance
    
    def scale(self, factor: float) -> 'ConfidenceScore':
        """
        신뢰도를 스케일링 (결과는 0.0~1.0 범위로 클램핑)
        
        Args:
            factor: 스케일 팩터
            
        Returns:
            ConfidenceScore: 스케일링된 신뢰도
        """
        if factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {factor}")
        
        scaled_value = min(1.0, max(0.0, self.value * factor))
        return ConfidenceScore(scaled_value)
    
    def combine_with(self, other: 'ConfidenceScore', method: str = 'average') -> 'ConfidenceScore':
        """
        다른 신뢰도와 결합
        
        Args:
            other: 결합할 신뢰도
            method: 결합 방법 ('average', 'min', 'max', 'product')
            
        Returns:
            ConfidenceScore: 결합된 신뢰도
        """
        if method == 'average':
            combined_value = (self.value + other.value) / 2.0
        elif method == 'min':
            combined_value = min(self.value, other.value)
        elif method == 'max':
            combined_value = max(self.value, other.value)
        elif method == 'product':
            combined_value = self.value * other.value
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return ConfidenceScore(combined_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리 형식으로 변환
        
        Returns:
            Dict[str, Any]: 신뢰도 정보
        """
        return {
            'value': self.value,
            'percentage': self.percentage,
            'level': self.confidence_level,
            'is_high': self.is_high,
            'is_medium': self.is_medium,
            'is_low': self.is_low
        }
    
    @classmethod
    def from_percentage(cls, percentage: float) -> 'ConfidenceScore':
        """
        백분율에서 신뢰도 생성
        
        Args:
            percentage: 백분율 (0.0 ~ 100.0)
            
        Returns:
            ConfidenceScore: 생성된 신뢰도
        """
        if not (0.0 <= percentage <= 100.0):
            raise ValueError(f"Percentage must be between 0.0 and 100.0, got {percentage}")
        
        return cls(percentage / 100.0)
    
    @classmethod
    def from_logits(cls, logits: float) -> 'ConfidenceScore':
        """
        로짓 값에서 신뢰도 생성 (시그모이드 적용)
        
        Args:
            logits: 로짓 값
            
        Returns:
            ConfidenceScore: 생성된 신뢰도
        """
        # 시그모이드 함수 적용
        sigmoid_value = 1.0 / (1.0 + math.exp(-logits))
        return cls(sigmoid_value)
    
    @classmethod
    def zero(cls) -> 'ConfidenceScore':
        """신뢰도 0.0 생성"""
        return cls(0.0)
    
    @classmethod
    def one(cls) -> 'ConfidenceScore':
        """신뢰도 1.0 생성"""
        return cls(1.0)
    
    @classmethod
    def half(cls) -> 'ConfidenceScore':
        """신뢰도 0.5 생성"""
        return cls(0.5)
    
    def __lt__(self, other: Union['ConfidenceScore', float]) -> bool:
        """작음 비교"""
        if isinstance(other, ConfidenceScore):
            return self.value < other.value
        return self.value < other
    
    def __le__(self, other: Union['ConfidenceScore', float]) -> bool:
        """작거나 같음 비교"""
        if isinstance(other, ConfidenceScore):
            return self.value <= other.value
        return self.value <= other
    
    def __gt__(self, other: Union['ConfidenceScore', float]) -> bool:
        """큼 비교"""
        if isinstance(other, ConfidenceScore):
            return self.value > other.value
        return self.value > other
    
    def __ge__(self, other: Union['ConfidenceScore', float]) -> bool:
        """크거나 같음 비교"""
        if isinstance(other, ConfidenceScore):
            return self.value >= other.value
        return self.value >= other
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.percentage:.1f}%"
    
    def __repr__(self) -> str:
        """개발자용 문자열 표현"""
        return f"ConfidenceScore({self.value:.3f}, {self.confidence_level})"
    
    def __float__(self) -> float:
        """float로 변환"""
        return self.value
    
    def __int__(self) -> int:
        """int로 변환 (백분율로)"""
        return int(self.percentage) 