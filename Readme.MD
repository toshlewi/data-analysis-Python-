Data Analysis and Visualization Assignment
Objective
The purpose of this assignment is to:

Load and analyze a dataset using the pandas library in Python.
Create simple plots and charts with the matplotlib and seaborn libraries for visualizing the data.
Tasks Completed
Task 1: Load and Explore the Dataset
Dataset Used: The Iris dataset, a classic dataset for classification problems, was loaded using sklearn.datasets.load_iris().
Steps Performed:
Loaded the dataset into a Pandas DataFrame.
Displayed the first 5 rows of the dataset using .head() to inspect the data.
Checked the structure of the dataset by printing data types and identifying missing values.
Verified that there were no missing values in the dataset.
Task 2: Basic Data Analysis
Steps Performed:
Computed basic statistics of the numerical columns using .describe().
Grouped the data by the species column and calculated the mean of numerical columns for each group.
Highlighted an interesting finding: Setosa species has the smallest average petal length.
Task 3: Data Visualization
Four different types of visualizations were created to gain insights into the dataset:

Line Chart:
Displayed the cumulative sum of sepal length (cm) to observe trends.
Bar Chart:
Compared the average petal length across species.
Histogram:
Showed the distribution of sepal length (cm) to understand its frequency.
Scatter Plot:
Visualized the relationship between sepal length (cm) and petal length (cm) for each species.
All plots were customized with titles, axis labels, and legends for better readability.

How to Run the Code
Ensure you have Python installed (version 3.7 or higher is recommended).
Install the required libraries by running the following command in your terminal:
Save the script as dataanalysis.py and run it using:
Libraries Used
pandas: For data manipulation and analysis.
matplotlib: For creating visualizations.
seaborn: For enhanced plotting styles and additional visualization options.
scikit-learn: For loading the Iris dataset.
Observations and Findings
The dataset contains three species of flowers: Setosa, Versicolor, and Virginica.
The Setosa species has the smallest average petal length compared to the other species.
The scatter plot shows a clear distinction between species based on sepal and petal lengths.
File Structure
dataanalysis.py: The Python script containing all the code for loading, analyzing, and visualizing the dataset.
Example Outputs
Line Chart: Cumulative Sepal Length
Line Chart

Bar Chart: Average Petal Length per Species
Bar Chart

Histogram: Distribution of Sepal Length
!Histogram

Scatter Plot: Sepal Length vs Petal Length
Scatter Plot

Notes
The Iris dataset was used as it is readily available and does not contain missing values. If using a different dataset, ensure it is in CSV format and modify the loading step accordingly.
The code includes error handling to manage potential issues during dataset loading.
Feel free to reach out if you have any questions or need further assistance!