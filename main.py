import torch
import numpy as np

# Задание 1
# 1.1 Создание тензоров
def create_four_tensors():
    t1 = torch.rand(3, 4) # двумерный массив 3x4
    t2 = torch.zeros(2, 3, 4) # трехмерный массив 3x4, заполненный нулями
    t3 = torch.ones(5, 5) # двумерный массив 5x5, заполненный единицами
    t4 = torch.arange(16.).reshape((4, 4)) # массив 0-15 с reshape 4x4 и float32 типом
    return t1, t2, t3, t4
    
    
# 1.2 Операции с тензорами
def transpose_matrix(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("Invalid type")
    return A.T


def matrix_mul(A, B):
    if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        raise TypeError("Invalid type")
    return A @ B


def matrix_el_mul_with_transpose(A, B):
    if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        raise TypeError("Invalid type")
    return A * B.T

a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a)
print(b)
print("___")
print(matrix_el_mul_with_transpose(a, b))