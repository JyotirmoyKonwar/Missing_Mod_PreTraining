import re
import matplotlib.pyplot as plt
import seaborn as sns

# Read the log file
with open("./run1.txt", "r") as f:
    lines = f.readlines()

# Regex to match lines like:
# [Stage 1] Epoch 1/50 | Train Loss: 1.7034 | Val Acc: 0.986
pattern = re.compile(r"\[Stage (\d+)\] Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Acc: ([\d.]+)")

# Data storage
stages, epochs, train_loss, val_acc = [], [], [], []

# Extract values
for line in lines:
    match = pattern.search(line)
    if match:
        stages.append(int(match.group(1)))
        epochs.append(int(match.group(2)))
        train_loss.append(float(match.group(3)))
        val_acc.append(float(match.group(4)))

# Plotting
sns.set(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot Train Loss
sns.lineplot(x=epochs, y=train_loss, ax=ax1, color='tab:red')
ax1.set_ylabel('Train Loss', color='tab:red')
ax1.set_title("Train Loss Over Epochs")
ax1.tick_params(axis='y', labelcolor='tab:red')
for stage_break in [10, 35]:
    ax1.axvline(x=stage_break + 0.5, color='gray', linestyle='--', linewidth=1)

# Plot Val Accuracy
sns.lineplot(x=epochs, y=val_acc, ax=ax2, color='tab:blue')
ax2.set_ylabel('Validation Accuracy', color='tab:blue')
ax2.set_title("Validation Accuracy Over Epochs")
ax2.set_xlabel('Epoch')
ax2.tick_params(axis='y', labelcolor='tab:blue')
for stage_break in [10, 35]:
    ax2.axvline(x=stage_break + 0.5, color='gray', linestyle='--', linewidth=1)

# Save the figure
plt.tight_layout()
fig.savefig("train_val_separate_plots.png")
print("Plot saved as train_val_separate_plots.png")
