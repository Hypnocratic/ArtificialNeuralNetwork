import matplotlib.pyplot as plt
import re
import ast

# Read the file content
with open('log.txt', 'r') as file:
    content = file.read()

# Extract gradients using regular expressions
weight_grad_str = re.search(r'Pytorch Weight Gradients\n(\[.*?\])', content, re.DOTALL)
bias_grad_str = re.search(r'Pytorch Bias Gradients\n(\[.*?\])', content, re.DOTALL)

# Convert string representations of lists to actual lists using 'ast.literal_eval'
weight_gradients = ast.literal_eval(weight_grad_str.group(1)) if weight_grad_str else []
bias_gradients = ast.literal_eval(bias_grad_str.group(1)) if bias_grad_str else []

# Plot the weight and bias gradients
plt.figure(figsize=(14, 7))

# Weight gradients plot
plt.subplot(1, 2, 1)
plt.plot(weight_gradients, label='Weight Gradients')
plt.title('Weight Gradients')
plt.xlabel('Iteration')
plt.ylabel('Gradient Value')
plt.legend()

# Bias gradients plot
plt.subplot(1, 2, 2)
plt.plot(bias_gradients, label='Bias Gradients', color='orange')
plt.title('Bias Gradients')
plt.xlabel('Iteration')
plt.ylabel('Gradient Value')
plt.legend()

plt.show()