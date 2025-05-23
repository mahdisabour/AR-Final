import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file names
filenames = ["10cm.csv", "15cm.csv", "20cm.csv", "25cm.csv", "30cm.csv", "37cm.csv"]

# Create output directory
output_dir = "measurement_analysis"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store results
results = {}

# Process each file
for filename in filenames:
    # Load the CSV file
    df = pd.read_csv(f"{filename}")
    measurements = df["Measurement"]

    # Calculate statistics
    mean = measurements.mean()
    variance = measurements.var()
    results[filename] = {"mean": mean, "variance": variance}

    # Histogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(measurements, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of {filename}")
    plt.xlabel("Measurement")
    plt.ylabel("Frequency")

    # PDF
    plt.subplot(1, 2, 2)
    x = np.linspace(measurements.min(), measurements.max(), 300)
    pdf = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(- (x - mean)**2 / (2 * variance))
    plt.plot(x, pdf, color='red')
    plt.title(f"PDF of {filename}")
    plt.xlabel("Measurement")
    plt.ylabel("Probability Density")
    plt.show()

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename.replace('.csv', '_plot.png')))
    plt.close()

# Save results to a summary CSV
summary_df = pd.DataFrame.from_dict(results, orient='index')
summary_df.index.name = "File"
summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"))