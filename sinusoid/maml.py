import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#---------------------------
# Utility Functions (unchanged)
#---------------------------
def generate_sinusoid_task(batch_size=10, x_range=(-5.0, 5.0)):
    A = np.random.uniform(0.1, 5.0)
    phi = np.random.uniform(0, np.pi)
    xs = np.random.uniform(x_range[0], x_range[1], size=batch_size)
    ys = A * np.sin(xs + phi)
    return xs, ys, A, phi

def generate_sinusoid_query(batch_size=10, x_range=(-5.0, 5.0), A=None, phi=None):
    xs = np.random.uniform(x_range[0], x_range[1], size=batch_size)
    ys = A * np.sin(xs + phi)
    return xs, ys, A, phi

#---------------------------
# Model Definition (unchanged)
#---------------------------
class MAMLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MAMLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#---------------------------
# MAML Training Procedure
#---------------------------
def functional_forward(x, params):
    # params: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
    # Make sure that the params order matches exactly the order of model's parameters.
    x = torch.relu(x @ params[0].T + params[1])   # fc1
    x = torch.relu(x @ params[2].T + params[3])   # fc2
    x = x @ params[4].T + params[5]               # fc3
    return x

def maml_train(
    model,
    loss_fn,
    meta_optimizer,
    meta_iterations=1000,
    inner_steps=3,
    inner_lr=0.01,
    task_batch_size=10,
    test_batch_size=10
):
    # Extract the parameters once and keep track of their order
    param_list = [p for p in model.parameters()]

    for iteration in range(meta_iterations):
        # Sample a new task
        xs, ys, A, phi = generate_sinusoid_task(batch_size=task_batch_size)
        xq, yq, _, _ = generate_sinusoid_query(batch_size=test_batch_size, A=A, phi=phi)
        
        xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
        ys_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        xq_t = torch.tensor(xq, dtype=torch.float32).unsqueeze(-1)
        yq_t = torch.tensor(yq, dtype=torch.float32).unsqueeze(-1)

        # Initialize fast weights as copies of current model parameters
        # Ensure requires_grad=True so that we can compute gradients
        fast_weights = [p.clone().detach().requires_grad_(True) for p in param_list]

        # Inner-loop adaptation using functional forward
        for _ in range(inner_steps):
            preds = functional_forward(xs_t, fast_weights)
            inner_loss = loss_fn(preds, ys_t)
            # Compute gradients wrt fast_weights
            grads = torch.autograd.grad(inner_loss, fast_weights, create_graph=True)
            # Update fast_weights
            fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]
        
        # Compute query loss with adapted parameters (fast_weights)
        query_preds = functional_forward(xq_t, fast_weights)
        query_loss = loss_fn(query_preds, yq_t)

        # Meta-update with query_loss
        meta_optimizer.zero_grad()
        query_loss.backward()
        
        # Update original model parameters
        meta_optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Query Loss: {query_loss.item():.4f}")

    print("MAML meta-training completed.")

#---------------------------
# MAML Evaluation
#---------------------------
def maml_evaluate(model, loss_fn, inner_steps=5, inner_lr=0.01, task_batch_size=10):
    # Freeze model parameters for meta-testing
    for param in model.parameters():
        param.requires_grad = True  # For adaptation steps

    # Sample a new unseen task
    xs, ys, A, phi = generate_sinusoid_task(batch_size=task_batch_size)
    xq = np.linspace(-5, 5, 100)
    yq = A * np.sin(xq + phi)

    xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    ys_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    xq_t = torch.tensor(xq, dtype=torch.float32).unsqueeze(-1)

    # Adaptation: create a separate optimizer for the model parameters
    adaptation_optimizer = optim.SGD(model.parameters(), lr=inner_lr)

    model.train()
    for _ in range(inner_steps):
        preds = model(xs_t)
        loss = loss_fn(preds, ys_t)
        adaptation_optimizer.zero_grad()
        loss.backward()
        adaptation_optimizer.step()

    # After adaptation, evaluate on query
    model.eval()
    with torch.no_grad():
        preds = model(xq_t).squeeze().numpy()

    # Plot results
    plt.figure(figsize=(10,6))
    plt.scatter(xs, ys, color='red', label='Adaptation Points')
    plt.plot(xq, yq, label='True Function', color='blue')
    plt.plot(xq, preds, label='MAML Prediction after Adaptation', color='green')
    plt.title(f"MAML Adaptation - A={A:.2f}, phi={phi:.2f}")
    plt.legend()
    plt.show()
