import numpy as np
import matplotlib.pyplot as plt

# Load loss data from the .npz file
loss_log_file = "./loss_log_20250217130252_.npy.npz"

try:
    data = np.load(loss_log_file, allow_pickle=True)
    loss_history = data["loss"]  # Extract stored loss values
except Exception as e:
    print(f"Error loading loss file: {e}")
    loss_history = []

# Convert to list of tuples (epoch, step, loss_value)
loss_history = list(loss_history)

# Extract separate lists for plotting
epochs = [entry[0] for entry in loss_history]  # Epoch numbers
steps = [entry[1] for entry in loss_history]  # Step numbers
loss_values = [entry[2] for entry in loss_history]  # Loss values

# 🔹 Create the loss plot
plt.figure(figsize=(12, 6))
plt.plot(steps, loss_values, label="Training Loss", color="blue", alpha=0.7)
plt.scatter(steps, loss_values, color="red", s=10, alpha=0.5)  # Scatter points for visibility

# 🔹 Labels and Titles
plt.xlabel("Training Steps")  # ✅ Use Steps on X-axis
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
