Aim
To study and implement Clustering techniques in Machine Learning using Python and understand how unlabeled data can be grouped into meaningful clusters.
The objective of this experiment is to perform clustering using Python libraries, visualize grouped data, and learn functions used for data preprocessing, model training, prediction, and evaluation.

Theory
Introduction to Clustering

Clustering is an unsupervised machine learning technique used to group similar data points into clusters without predefined labels.
Data points in the same cluster are more similar to each other than to those in other clusters.
Unlike classification, clustering works on unlabeled data.

Examples of Clustering:
Customer segmentation
Market basket analysis
Image segmentation
Document grouping
Social network analysis
Disease pattern detection
Libraries Used in Experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Theory of Libraries:
pandas
Used for loading and handling datasets.

numpy
Used for numerical calculations and arrays.

matplotlib.pyplot
Used for plotting graphs.

seaborn
Used for attractive statistical visualizations.

Functions Used in Detail
1. Reading Dataset
df = pd.read_csv("data.csv")
Function Used: pd.read_csv()
Loads CSV file into DataFrame format.

2. Display Dataset
df.head()
Function Used: head()
Displays first five rows of dataset.

3. Dataset Shape
df.shape
Function Used: shape

Returns:
(rows, columns)
Used to know size of dataset.

4. Dataset Information
df.info()
Function Used: info()

Displays:
Column names
Data types
Non-null values
Memory usage
5. Statistical Summary
df.describe()
Function Used: describe()

Shows:
Mean
Count
Std deviation
Minimum
Maximum
Quartiles
6. Selecting Features
X = df.iloc[:, :]
Functions Used:
iloc[]

Selects rows and columns by integer indexing.

:Means all rows / all columns.

Clustering Algorithm Used
K-Means Clustering
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
Theory of K-Means

K-Means divides dataset into K clusters.

Steps:
Choose number of clusters K
Select centroids
Assign nearest data points
Update centroid mean
Repeat until stable
Functions Used:
KMeans()

Creates K-Means clustering model.

Parameters:
n_clusters=3 → Number of clusters
random_state=42 → Reproducible result
fit(X)
Trains model using dataset.

Predict Cluster Labels
labels = model.fit_predict(X)
Function Used:
fit_predict()
Performs training and returns cluster labels.

Example Output:
[0,1,2,1,0,2...]
Cluster Centers
model.cluster_centers_
Function Used:
cluster_centers_
Returns centroid coordinates of clusters.

Inertia Value
model.inertia_
Function Used:
inertia_

Measures within-cluster sum of squares.
Lower value = Better compact clusters.

Elbow Method
wcss = []
for i in range(1,11):
    model = KMeans(n_clusters=i)
    model.fit(X)
    wcss.append(model.inertia_)
Theory:
Elbow method is used to find best value of K.

WCSS = Within Cluster Sum of Squares
When graph forms elbow shape, that K is selected.

Plotting Elbow Graph
plt.plot(range(1,11), wcss)
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

Functions Used:
plt.plot()
Draws line graph.

xlabel()
Sets X-axis label.

ylabel()
Sets Y-axis label.

title()
Sets graph title.

show()
Displays graph.

Scatter Plot for Clusters
plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()
Function Used:
scatter()

Displays clustered data points.
Other Clustering Algorithms (May Be Included)
Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
Forms clusters in tree structure.

DBSCAN
from sklearn.cluster import DBSCAN
Density-based clustering algorithm.

Applications of Clustering
Customer segmentation
Recommendation systems
Fraud detection
Social media grouping
Image compression
Pattern recognition

Conclusion
Thus, the experiment on Clustering using Python was successfully performed. The dataset was loaded and explored using Pandas functions.
K-Means clustering algorithm was implemented to group similar data points into clusters. Functions such as read_csv(), head(), info(), describe(), iloc[], KMeans(), fit(), fit_predict(), cluster_centers_(), inertia_(), plot(), scatter(), show() were studied in detail
. This experiment helped in understanding how unsupervised learning techniques organize unlabeled data efficiently.
