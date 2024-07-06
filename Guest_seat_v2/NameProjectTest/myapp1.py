import torch

# Create tensors A and B
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nMatrix C (result of A @ B):")
print(C)
