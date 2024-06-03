###
 # File: /Hessian/main.py
 # Created Date: Monday, June 3rd 2024
 # Author: Zihan
 # -----
 # Last Modified: Monday, 3rd June 2024 10:48:14 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import torch
import torch.nn as nn
import torch.optim as optim

# Check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create a model instance and move it to the device
model = SimpleModel().to(device)

# Define a loss function
criterion = nn.MSELoss()

# Dummy dataset
data = torch.randn(100, 10).to(device)  # 100 data points, each of dimension 10
targets = torch.randn(100, 1).to(device)  # Corresponding targets

print(f"Input data shape: {data.shape}")
print(f"Target shape: {targets.shape}")

# Compute Hessian (second-order derivatives) for each data point
hessian_matrices = []

for i in range(len(data)):
    # Get the data point and target
    input_data = data[i].unsqueeze(0)  # Add batch dimension
    target = targets[i].unsqueeze(0)  # Add batch dimension
    
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target)
    
    # Compute gradients
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # Flatten the gradients into a single vector
    grad_vector = torch.cat([g.view(-1) for g in grad_params])
    
    # Compute the Hessian matrix
    hessian_matrix = []
    for grad in grad_vector:
        second_order_grads = torch.autograd.grad(grad, model.parameters(), retain_graph=True)
        second_order_grads_vector = torch.cat([g.view(-1) for g in second_order_grads])
        hessian_matrix.append(second_order_grads_vector.unsqueeze(0))
    
    hessian_matrix = torch.cat(hessian_matrix, 0)
    hessian_matrices.append(hessian_matrix)

hessian_matrices = torch.stack(hessian_matrices)

print(hessian_matrices.shape)  # Shape: (number of data points, number of parameters, number of parameters)

# assert if H is positive definite and symmetric
for i in range(len(data)):
    H = hessian_matrices[i]
    # assert torch.allclose(H, H.t(), atol=1e-6), "Hessian matrix is not symmetric"
    # assert torch.all(torch.eig(H, eigenvectors=False)[0] > 0), "Hessian matrix is not positive definite"
    # RuntimeError: This function was deprecated since version 1.9 and is now removed. `torch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors mimicking complex tensors.
    # L, _ = torch.eig(A) should be replaced with:
    # L_complex = torch.linalg.eigvals(A)
    assert torch.all(torch.linalg.eigvals(H).real > 0), "Hessian matrix is not positive definite"
    assert torch.allclose(H, H.T, atol=1e-6), "Hessian matrix is not symmetric"
