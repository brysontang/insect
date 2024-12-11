import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#---------------------------
# Utility Functions
#---------------------------
def generate_sinusoid_task(batch_size=10, x_range=(-5.0, 5.0)):
    # Random amplitude and phase
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
# Model Definition
#---------------------------
class INSECTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(INSECTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# During meta-training, for each iteration:
# 1. Sample a new task (A, phi).
# 2. Initialize a memory vector (requires_grad=True).
# 3. Perform adaptation steps on the support set by backprop into memory only.
# 4. Compute the loss on a query point after adaptation and backprop into model (meta-update).
def meta_train(meta_iterations=1000, inner_steps=3, adapt_lr=0.01, task_batch_size=10, test_batch_size=10, memory_dim=5, model=None, loss_fn=None, meta_optimizer=None):
    for iteration in range(meta_iterations):
        # Sample task
        xs, ys, A, phi = generate_sinusoid_task(batch_size=task_batch_size)
        xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
        ys_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        
        # Generate query set using the SAME A and phi
        xq, yq, A, phi = generate_sinusoid_query(batch_size=test_batch_size, A=A, phi=phi)
        xq_t = torch.tensor(xq, dtype=torch.float32).unsqueeze(-1)
        yq_t = torch.tensor(yq, dtype=torch.float32).unsqueeze(-1)
        
        # Initialize memory for this task
        memory = torch.zeros(memory_dim, requires_grad=True, dtype=torch.float32)
        
        # Inner-loop adaptation (memory update)
        # We'll do a few gradient steps to fit the support set by only updating memory
        for _ in range(inner_steps):
            # Concatenate input with memory for each support input
            memory_rep = memory.unsqueeze(0).expand(xs_t.size(0), -1)
            inputs = torch.cat([xs_t, memory_rep], dim=1)
            preds = model(inputs)
            loss = loss_fn(preds, ys_t)
            
            # Compute gradients w.r.t. memory only
            grads = torch.autograd.grad(loss, memory, retain_graph=False)[0]
            # Update memory with a simple gradient step
            memory = memory - adapt_lr * grads
        
        # After adaptation, evaluate on query
        memory_rep = memory.unsqueeze(0).expand(xq_t.size(0), -1)
        inputs = torch.cat([xq_t, memory_rep], dim=1)
        preds = model(inputs)
        query_loss = loss_fn(preds, yq_t)
        
        # Meta-update the model parameters
        meta_optimizer.zero_grad()
        query_loss.backward()
        meta_optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Query Loss: {query_loss.item():.4f}")

    print("Meta-training completed.")

def evaluate_model(model, task_batch_size=10, test_batch_size=10, adapt_lr=0.01, memory_dim=5, loss_fn=None):
    #---------------------------
    # Test-Time Adaptation
    #---------------------------
    # Now we freeze the model weights and adapt the memory for a new task.
    for param in model.parameters():
        param.requires_grad = False

    # Sample a new task unseen before
    xs, ys, A, phi = generate_sinusoid_task(batch_size=task_batch_size)
    xs_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    ys_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)

    # Query points to test generalization
    xq = np.linspace(-5, 5, 100)
    yq = A * np.sin(xq + phi)
    xq_t = torch.tensor(xq, dtype=torch.float32).unsqueeze(-1)

    memory = torch.zeros(memory_dim, requires_grad=True, dtype=torch.float32)
    memory_optimizer = optim.SGD([memory], lr=adapt_lr)

    # Adapt memory to the new task using the support set
    adapt_steps = 100
    for _ in range(adapt_steps):
        memory_optimizer.zero_grad()
        memory_rep = memory.unsqueeze(0).expand(xs_t.size(0), -1)
        inputs = torch.cat([xs_t, memory_rep], dim=1)
        preds = model(inputs)
        loss = loss_fn(preds, ys_t)
        loss.backward()
        memory_optimizer.step()

    # After adaptation, predict on query points
    with torch.no_grad():
        memory_rep = memory.unsqueeze(0).expand(xq_t.size(0), -1)
        inputs = torch.cat([xq_t, memory_rep], dim=1)
        preds = model(inputs).squeeze().numpy()

    # Plot the results
    plt.figure(figsize=(10,6))
    plt.scatter(xs, ys, label="Adaptation Points", color="red")
    plt.plot(xq, yq, label="True Function", color="blue")
    plt.plot(xq, preds, label="Model Prediction after Memory Adaptation", color="green")
    plt.title(f"Sinusoid: A={A:.2f}, phi={phi:.2f}")
    plt.legend()
    plt.show()