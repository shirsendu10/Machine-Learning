import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import skew , kurtosis

# Generate or  load data (you can replace this with your data)
data = {
  "A": np.random.normal(50, 10, 5),
  "B": np.random.uniform(100, 80, 5),
  "C": np.random.poisson(30, 5)
}
print(data)
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)


# Compute statistics manually
statistics = []
for column in df.columns:
    col_data = df[column].values
    n = len(col_data)
    
    # Sort the data
    sorted_data = sorted(col_data)
    
    # Compute mean
    sum_data = sum(sorted_data)
    mean_value = sum_data / n
    
    # Compute median
    if n % 2 == 0:
        median_value = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median_value = sorted_data[n//2]
    
    # Compute mode
    freq = {}
    for value in sorted_data:
        freq[value] = freq.get(value, 0) + 1
    mode_value = max(freq, key=freq.get)
    
    # Compute min and max
    min_value = sorted_data[0]
    max_value = sorted_data[-1]
    
    # Compute range
    range_value = max_value - min_value
    
    # Store the computed statistics
    stats = {
        "Column": column,
        "Mean": mean_value,
        "Median": median_value,
        "Mode": mode_value,
        "Standard Deviation": df[column].std(),  # Using built-in for this example
        "Variance": df[column].var(),            # Using built-in for this example
        "Skewness": skew(col_data),
        "Kurtosis": kurtosis(col_data),
        "Min": min_value,
        "Max": max_value,
        "Range": range_value
    }
    statistics.append(stats)

 
 #save statistics to csv 
stats_df = pd.DataFrame(statistics)
 #print(statistics)
stats_df.to_csv("statistics.csv", index=False)
print("statistics saved to 'statistics.csv'")

#visualization '
for coloumn in df.columns:
  plt.figure(figsize=(15, 5))

  #Box plot 
  plt.subplot(1, 3, 1)
  sns.boxplot(df[coloumn])
  plt.title(f"{coloumn} Box plot")

  #create the line plot

  plt.subplot(1, 3, 1)
  plt.plot(df[coloumn], marker='o', linestyle='--', color='b', label='line 1')
  plt.title(f"{coloumn} Line plot")

  #histogram 
  plt.subplot(1, 3, 2)
  plt.hist(df[coloumn], bins=20, edgecolor='black', alpha=0.7)
  plt.title(f"{coloumn} Histogram")

  #bar plot(frequency counts)
  plt.subplot(1, 3, 3)
  value_counts = df[coloumn].value_counts().head(10)
  plt.bar(value_counts.index, value_counts.values)
  plt.title(f"{coloumn} Bar plot")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

#correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation matrix")
plt.show()
print("Done")