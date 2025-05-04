# Iris Dataset Analysis

This Python script (`iris_analysis.py`) performs a basic data analysis and visualization of the Iris dataset. The Iris dataset is a classic dataset in machine learning and statistics, containing measurements of sepal length, sepal width, petal length, and petal width for three different species of iris flowers: setosa, versicolor, and virginica.

## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

* **pandas:** For data manipulation and analysis.
* **matplotlib:** For creating basic plots.
* **seaborn:** For enhanced data visualization.
* **scikit-learn (sklearn):** For loading the Iris dataset.

You can install these libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
Script DescriptionThe iris_analysis.py script consists of the following functions:1.  load_and_explore_data()Loads the Iris dataset using the load_iris() function from scikit-learn.Creates a pandas DataFrame from the dataset.Adds a 'species' column by mapping the numeric target values to the corresponding species names.Prints the first 5 rows of the DataFrame using df.head().Displays the data types of each column using df.dtypes.Checks for missing values in the dataset using df.isnull().sum() and prints the result.Returns the pandas DataFrame.Handles potential errors during data loading and prints an informative error message.2.  analyze_data(df)Takes the pandas DataFrame as input.Calculates and prints descriptive statistics of the dataset using df.describe().Groups the data by the 'species' column and calculates the mean of each numerical column for each species.Prints the resulting grouped DataFrame.Highlights an interesting finding: that the Setosa species has the smallest average petal length.Returns the grouped DataFrame.3.  visualize_data(df)Takes the pandas DataFrame as input.Creates four visualizations using matplotlib and seaborn:Line Chart: Cumulative sum of sepal length.Bar Chart: Average petal length per species.Histogram: Distribution of sepal length.Scatter Plot: Sepal length vs. petal length, with points colored by species.Adds appropriate titles, labels, and legends to each plot.Uses plt.tight_layout() to prevent labels from overlapping.Displays each plot using plt.show().4.  Main ExecutionCalls the load_and_explore_data() function to load and explore the dataset.If the dataset is loaded successfully (i.e., the returned DataFrame is not None), it calls the analyze_data() and visualize_data() functions to analyze and visualize the data.How to Run the ScriptSave the code as iris_analysis.py.Ensure you have the required libraries installed (see "Prerequisites" above).Open a terminal or command prompt.Navigate to the directory where you saved the file.Run the script using the following command:python iris_analysis.py
The script will:Load the Iris dataset.Print the first few rows, data types, and missing values.Print descriptive statistics and the mean of each feature per species.Display four plots: a line chart, a bar chart, a histogram, and a scatter plot.OutputThe script will print descriptive statistics and analysis results to the console. It will also generate four plots, which will be displayed in separate windows.Additional NotesThe script uses plt.style.use('ggplot') for a visually appealing plot style.The seaborn library is used to create more informative and aesthetically pleasing plots, such as the bar chart and scatter plot.Error handling is implemented in the load_and_explore_data() function to catch potential exceptions during data loading.The script assumes that the Iris dataset is available through the scikit-learn library
