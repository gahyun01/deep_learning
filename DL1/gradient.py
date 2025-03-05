# coding: utf-8
import numpy as np

# 편미분 ( 1차원 배열에 대한 수치적 경량을 계산하는 함수 )
def _numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성 ( 원소가 모두 0인 배열 )
    
    # x의 각 원소에 대해서 수치미분
    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x + h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h) # 중앙 차분을 이용하여 기울기 계산
        x[idx] = tmp_val # 원래 값으로 복원
    
    return grad

# 2차원 배열에 대한 수치적 경량을 계산하는 함수
def numerical_gradient_2d(f, X):
    if X.ndim == 1: # X가 1차원 배열일 경우
        return _numerical_gradient_1d(f, X) # 1차원 함수 호출
    else:
        grad = np.zeros_like(X) # X와 같은 크기의 0 배열 생성
        
        for idx, x in enumerate(X): # X의 각 요소에 대해 반복
            grad[idx] = _numerical_gradient_1d(f, x) # 각 요소에 대한 기울기 계산
        
        return grad # 계산된 기울기를 반환


# 일반 n차원 배열에 대한 수치적 경량을 계산하는 함수
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성 ( 원소가 모두 0인 배열 )
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # 다차원 배열을 반복하기 위한 niter 객체 생성
    while not it.finished: # 반복이 끝날 때까지
        idx = it.multi_index # 현재 인덱스 저장
        tmp_val = x[idx]

        # f(x + h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h) # 중앙 차분을 이용하여 기울기 계산
        x[idx] = tmp_val # 원래 값으로 복원
        
        it.iternext() # 다음 요소로 이동
        
    return grad # 계산된 기울기를 반환
