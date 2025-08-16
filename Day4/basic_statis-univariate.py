import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load the data (you can replace this with the path to your data)
data ={
      "A": np.random.normal(50 ,10,5),
      "B": np.random.uniform(20,80,5),
      "C": np.random.poisson(30,5)
}
print(data)
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
#compute basic statistics
statistics = []
for column in df.columns:
    col_data = df[column]
    stats = {
        "Column": column,
        "Mean": col_data.mean(),
        "Median": col_data.median(),
        "Mode": col_data.mode() [0] if not col_data.mode().empty else np.nan,
        "variance": col_data.var(),
        "Standard Deviation": col_data.std(),
        "Skewness": skew(col_data),
        "Kurtosis": kurtosis(col_data),
        "Min": col_data.min(),
        "Max": col_data.max(),
        "Range": col_data.max()-col_data.min()
    }
    statistics.append(stats)

    #Save the statistics to a CSV file
    stats_df = pd.DataFrame(statistics)
    #print(statistics)
    stats_df.to_csv("basic_stats.csv", index=False)
    print("Statistics saved to 'statistics_summary.csv'")

#visualization
for column in df.columns:
    plt.figure(figsize=(15,5))

    #Box plot
    plt.subplot(1,3,1)
    sns.boxplot(y=df[column])
    plt.title(f"Box plot: {column}")

    #Create the line plot
    plt.subplot(1,3,1)
    plt.plot(df[column], marker='o',linestyle='-', color='b',label='line plot')
    plt.title(f"Line Graph: {column}")

    #Create the histogram
    plt.subplot(1,3,2)
    plt.hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Histogram: {column}")

    #Bar plot (frequency counts)
    plt.subplot(1,3,3)
    value_counts = df[column].value_counts().head(10) # Show top 10 unique values
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f"Bar plot: {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Correlation matrix
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt=".2f")
plt.title("Correlation Coefficient Matrix")
plt.show()
print("Program completed successfully.Visualizations displayed and csv files saved")