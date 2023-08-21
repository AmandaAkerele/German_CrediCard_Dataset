# Credit Card Analysis

---

# Data Analysis Project: Exploratory Data Analysis and Clustering

This repository contains code and analysis for performing Exploratory Data Analysis (EDA) and clustering on a German credit dataset. The project aims to understand the dataset's characteristics, identify patterns, and apply clustering techniques to group similar instances.

## Table of Contents

- [Project Aim](#project-aim)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Clustering](#clustering)
- [Conclusion](#conclusion)

## Project Aim

The main objectives of this project are as follows:

1. Perform EDA and necessary data cleaning.
2. Apply one-hot encoding to categorical variables.
3. Visualize histograms of numerical features to identify skewness. Apply log transformation if needed.
4. Apply feature scaling to prepare the data for clustering.
5. Utilize the elbow method to determine the optimal number of clusters.
6. Visualize the chosen number of clusters using PCA.
7. Implement K-Fold Cross Validation with a selected classifier and report evaluation metrics.
8. Draw conclusions based on the analysis.

## Libraries Used

The following libraries were used for this project:

- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn

## Dataset

The dataset used for this project is the "German Dataset.csv," which contains information about credit applicants. It includes attributes like age, sex, job, housing, credit amount, duration, purpose, and risk. The dataset has both numerical and categorical features.

## Data Preprocessing

The data preprocessing steps included handling missing values in columns like "Saving accounts" and "Checking account," converting categorical variables into numerical format using one-hot encoding, and performing feature scaling to ensure comparable scales for clustering.

## Exploratory Data Analysis

EDA involved visualizing the distributions of numerical features such as age, credit amount, and duration using histograms. The analysis aimed to identify any patterns or trends in the data and determine if transformations were needed.

## Clustering

The clustering process began with determining the optimal number of clusters using the elbow method. The chosen number of clusters was visualized using Principal Component Analysis (PCA) for dimensionality reduction. A selected clustering algorithm was then applied to group similar instances.

## Conclusion

The project successfully explored the German credit dataset through EDA, performed clustering analysis, and reported findings. By implementing K-Fold Cross Validation with a chosen classifier, the project also assessed the performance of the clustering approach. The README file provides an overview of the steps taken, the tools used, and the insights gained from the analysis.

---
