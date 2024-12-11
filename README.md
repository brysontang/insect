# INSECT: Incremental Note-taking System for Embedded Continual Transfer

INSECT is a few-shot meta-learning framework designed to efficiently adapt to new tasks through a dynamic memory mechanism. Instead of fine-tuning all model parameters like standard approaches (e.g., MAML), INSECT leverages a learnable memory component that allows for rapid, low-overhead adaptation. This design enables faster per-task updates and scales effectively to larger models and more complex task distributions.

## Key Features:

- **Memory-Based Inner Loop Updates:** INSECT focuses on updating only a small memory tensor during inner loop adaptation, reducing overhead and making the approach highly scalable.
- **Modular Architecture:** By separating task-specific adaptation (via the memory) from the feature extraction layers, the backbone remains stable while only the memory is tuned.
- **Scalability for Large Models:** Traditional meta-learning methods can become prohibitively expensive for large architectures. INSECT mitigates this by confining the adaptation to a small memory structure.
- **Easy Integration:** INSECT uses standard PyTorch operations and can be integrated with various feature extractors, classifiers, or loss functions.

## Advantages Over MAML:

- **Computational Efficiency:** MAML updates all model parameters for each task in the inner loop, which is costly. INSECT updates only a small memory representation, significantly reducing computational and memory requirements.
- **Faster Adaptation:** During test-time adaptation, only the memory is updated, allowing INSECT to quickly adapt to new tasks with minimal overhead.
- **Reduced Memory Footprint:** Since INSECT does not need to backprop through large sets of parameters for each inner step, it lowers the overall training complexity.

## Getting Started:

1. Since the project is provided as a single notebook file, simply open it in Jupyter or a compatible environment.
2. Ensure you have Python 3.7+, PyTorch installed, and access to a GPU for faster training if desired.
3. The notebook includes cells for data loading, model definition, training, and evaluation steps.
4. Adjust parameters (e.g., number of inner steps, memory dimension, learning rates) directly in the notebook cells as needed.
5. Run the notebook cells sequentially to train and evaluate INSECT on your chosen tasks.

## Configuration:

- Within the notebook, you can modify code cells to set parameters such as `inner_steps`, `memory_dim`, `inner_lr`, `outer_lr`, and the number of support and query samples.
- Experiment with different backbone models or task distributions by changing the corresponding code cells.

## Architecture Overview:

INSECT consists of:

- A feature extractor to encode inputs into embeddings.
- A learnable memory module that is updated with support data during the inner loop.
- A classifier that uses both the extracted features and the task-specific memory to make predictions.

The training loop is:

1. Sample a task with support and query sets.
2. Use the support set to update the memory, minimizing support loss.
3. Evaluate the adapted memory on the query set, compute the query loss, and update the overall system (e.g., the initialization of the memory) to improve future adaptation.

## Contact and Contributions:

- If you encounter issues or have suggestions, consider opening a discussion or contacting the maintainers through the appropriate channel.
- Contributions are welcome, including improvements to the notebook code, documentation, or overall methodology.

## License:

INSECT is released under the MIT License. Use it as you see fit, with attribution where appropriate.
