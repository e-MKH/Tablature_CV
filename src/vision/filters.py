"""
Hand Coordinate Filtering Module

손 좌표의 떨림을 제거하기 위한 One Euro Filter 구현
"""

import math
from ..config.vision_config import FILTER_CONFIG


class OneEuroFilter:
    """
    One Euro Filter

    저지연 노이즈 제거 필터. 속도에 따라 적응적으로 cutoff를 조정하여
    빠른 움직임은 정확하게, 느린 움직임은 부드럽게 필터링합니다.

    Reference:
        Casiez, G., Roussel, N., & Vogel, D. (2012).
        1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    """

    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        """
        Args:
            t0: 초기 시간
            x0: 초기 값
            dx0: 초기 속도 (기본값: 0.0)
            min_cutoff: 최소 cutoff 주파수 (기본값: 1.0)
            beta: 속도 민감도 (기본값: 0.05)
            d_cutoff: 속도 필터의 cutoff 주파수 (기본값: 1.0)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        """
        Smoothing factor 계산

        Args:
            t_e: 시간 간격
            cutoff: cutoff 주파수

        Returns:
            float: smoothing factor (0~1)
        """
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        """
        지수 평활법

        Args:
            a: smoothing factor
            x: 현재 값
            x_prev: 이전 값

        Returns:
            float: 평활된 값
        """
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        """
        필터링 수행

        Args:
            t: 현재 시간
            x: 현재 값

        Returns:
            float: 필터링된 값
        """
        # 시간 간격 계산
        t_e = t - self.t_prev

        # 시간이 역행하거나 동일하면 이전 값 반환
        if t_e <= 0:
            return self.x_prev

        # 속도 필터링
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 속도에 따른 cutoff 조정
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 값 필터링
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 상태 업데이트
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class PointFilter:
    """
    2D 포인트 필터

    x, y 좌표를 각각 OneEuroFilter로 필터링합니다.
    손가락 끝 좌표의 떨림 제거에 사용됩니다.
    """

    def __init__(self, min_cutoff=None, beta=None, d_cutoff=None):
        """
        Args:
            min_cutoff: OneEuroFilter의 min_cutoff (기본값: config에서 가져옴)
            beta: OneEuroFilter의 beta (기본값: config에서 가져옴)
            d_cutoff: OneEuroFilter의 d_cutoff (기본값: config에서 가져옴)
        """
        self.first_time = True
        self.x_filter = None
        self.y_filter = None

        # config에서 설정값 가져오기
        self.min_cutoff = min_cutoff if min_cutoff is not None else FILTER_CONFIG["min_cutoff"]
        self.beta = beta if beta is not None else FILTER_CONFIG["beta"]
        self.d_cutoff = d_cutoff if d_cutoff is not None else FILTER_CONFIG["d_cutoff"]

        self.config = (self.min_cutoff, self.beta, self.d_cutoff)

    def process(self, t, pos):
        """
        2D 포인트 필터링

        Args:
            t: 현재 시간
            pos: (x, y) 좌표 튜플

        Returns:
            tuple: 필터링된 (x, y) 좌표
        """
        x, y = pos

        # 첫 호출 시 필터 초기화
        if self.first_time:
            self.x_filter = OneEuroFilter(t, x, 0.0, *self.config)
            self.y_filter = OneEuroFilter(t, y, 0.0, *self.config)
            self.first_time = False
            return x, y

        # x, y 각각 필터링
        x_hat = self.x_filter.filter(t, x)
        y_hat = self.y_filter.filter(t, y)

        return x_hat, y_hat

    def reset(self):
        """필터 리셋"""
        self.first_time = True
        self.x_filter = None
        self.y_filter = None


class MultiPointFilter:
    """
    여러 포인트를 관리하는 필터 매니저

    손가락 4개 등 여러 포인트를 동시에 필터링할 때 사용합니다.
    """

    def __init__(self):
        """초기화"""
        self.filters = {}  # {point_id: PointFilter}

    def process(self, t, point_id, pos):
        """
        특정 포인트 필터링

        Args:
            t: 현재 시간
            point_id: 포인트 식별자 (예: "I", "M", "R", "P")
            pos: (x, y) 좌표

        Returns:
            tuple: 필터링된 (x, y) 좌표
        """
        # 해당 포인트의 필터가 없으면 생성
        if point_id not in self.filters:
            self.filters[point_id] = PointFilter()

        return self.filters[point_id].process(t, pos)

    def reset(self, point_id=None):
        """
        필터 리셋

        Args:
            point_id: 리셋할 포인트 ID (None이면 전체 리셋)
        """
        if point_id is None:
            # 전체 리셋
            self.filters.clear()
        elif point_id in self.filters:
            # 특정 포인트만 리셋
            del self.filters[point_id]

    def has_filter(self, point_id):
        """
        특정 포인트의 필터 존재 여부 확인

        Args:
            point_id: 포인트 식별자

        Returns:
            bool: 필터 존재 여부
        """
        return point_id in self.filters