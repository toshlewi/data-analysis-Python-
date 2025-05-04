# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map(dict(zip(range(3), iris.target_names)))

        print("Dataset loaded successfully.")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        # Check for data types and missing values
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Task 2: Basic Data Analysis
def analyze_data(df):
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Grouping by species to find the average for each numeric column
    grouped = df.groupby('species').mean()
    print("\nMean of each numerical column grouped by species:")
    print(grouped)

    # Highlight interesting findings
    print("\nInteresting Finding: Setosa species has the smallest average petal length.")
    return grouped

# Task 3: Data Visualization
def visualize_data(df):
    plt.style.use('ggplot')  # Use a valid style
    sns.set_context("talk")

    # 1. Line chart (Cumulative sum of sepal length)
    plt.figure(figsize=(8, 6))
    df['cumulative_sepal_length'] = df['sepal length (cm)'].cumsum()
    plt.plot(df.index, df['cumulative_sepal_length'], label='Cumulative Sepal Length')
    plt.title('Line Chart: Cumulative Sepal Length')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Sepal Length (cm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Bar chart (Average petal length per species)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title('Bar Chart: Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.tight_layout()
    plt.show()

    # 3. Histogram (Distribution of sepal length)
    plt.figure(figsize=(8, 6))
    plt.hist(df['sepal length (cm)'], bins=10, color='blue', alpha=0.7)
    plt.title('Histogram: Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # 4. Scatter plot (Sepal length vs Petal length)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set2')
    plt.title('Scatter Plot: Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.tight_layout()
    plt.show()

# Main Execution
df = load_and_explore_data()
if df is not None:
    analyze_data(df)
    visualize_data(df)
