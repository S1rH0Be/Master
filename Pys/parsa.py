import re
import matplotlib.pyplot as plt
import numpy as np
# Read the file
with open("/Users/fritz/Downloads/rainerschüttler.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Find all occurrences of the pattern "digit : digit"
scores = re.findall(r"\b(\d+):(\d+)\b", content)

# Convert and sum each pair
sums = [int(x) + int(y) for x, y in scores]

# Create the boxplot
plt.figure(figsize=(6, 4))
box = plt.boxplot(
    sums,
    vert=True,
    patch_artist=True,  # Fill the boxes with color
    boxprops=dict(facecolor="skyblue", color="black", linewidth=1),  # Box fill color
    medianprops=dict(color="black", linewidth=1),  # Median line
    whiskerprops=dict(color="black", linewidth=1),  # Whiskers
    capprops=dict(color="black", linewidth=1),  # End caps
    flierprops=dict(marker="o", color="black", markersize=6, alpha=1)  # Outliers
)
# Labeling
plt.title("Goals Scored in each match of the 2023/2024 Bundesliga Season")
plt.ylabel("Number of Goals Scored")
# plt.show()

print((sum(sums)-25)/len(sums))

def shifted_geometric_mean(values, shift):
    values = np.array(values)
    # Shift the values by the constant and check for any negative values after shifting
    shifted_values = values + shift
    if shifted_values.dtype == 'object':
        # Attempt to convert to float
        shifted_values = shifted_values.astype(float)

    shifted_values_log = np.log(shifted_values)  # Step 1: Log of each element in shifted_values
    log_mean = np.mean(shifted_values_log)  # Step 2: Compute the mean of the log values
    geo_mean = np.exp(log_mean) - shift
    geo_mean = np.round(geo_mean, 6)
    return geo_mean
