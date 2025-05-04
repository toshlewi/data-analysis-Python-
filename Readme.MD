# Iris Dataset Analysis - README

## Overview
This project performs a comprehensive analysis of the famous Iris flower dataset using Python. The analysis includes data loading, exploration, statistical analysis, and visualization to understand the characteristics of different Iris species.

## Features

### 1. Data Loading and Exploration
- Loads the Iris dataset from scikit-learn
- Converts the data into a pandas DataFrame
- Adds species names for better readability
- Provides initial exploration including:
  - First 5 rows of data
  - Data types information
  - Missing values check

### 2. Data Analysis
- Calculates descriptive statistics (mean, std, min, max, etc.)
- Groups data by species and calculates mean values
- Highlights interesting findings about species characteristics

### 3. Data Visualization
- **Line Chart**: Shows cumulative sum of sepal length
- **Bar Chart**: Displays average petal length per species
- **Histogram**: Visualizes distribution of sepal length
- **Scatter Plot**: Illustrates relationship between sepal and petal length, colored by species

## Requirements
- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation
1. Clone this repository
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

## Usage
Simply run the `iris_analysis.py` script:
```bash
python iris_analysis.py
```

The script will:
1. Load and explore the dataset
2. Perform statistical analysis
3. Generate and display four different visualizations

## Output
The script outputs:
1. Dataset information to the console
2. Statistical summaries to the console
3. Four matplotlib visualizations in separate windows

## Key Findings
- Setosa species has the smallest average petal length
- Virginica has the longest average petal length
- Clear separation between species visible in the scatter plot
- Sepal length distribution shows distinct patterns

## Author
[Your Name]

## License
[MIT License] (or specify your preferred license)
