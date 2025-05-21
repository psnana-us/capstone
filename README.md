# capstone

# Credit Card Fraud EDA Summary Report
Overview
This notebook performs an extensive Exploratory Data Analysis (EDA) on the Kaggle Credit Card Fraud dataset. The goal is to understand the dataset characteristics, identify patterns, visualize feature distributions, and uncover insights related to fraudulent transactions, going beyond basic summary statistics to inform potential modeling efforts.

Jupyter Notebook location:  https://github.com/psnana-us/capstone/blob/main/CreditCard_Fraud_EDA_Detailed.ipynb

# Executive Summary: 
	1.	We conducted a credit card fraud detection analysis using real transaction data and applied a technique to balance the rare fraud cases.
	2.	Two predictive models were trained and tuned: a Random Forest and a Logistic Regression.
	3.	We optimized each model’s settings on balanced data to improve their ability to spot fraud.
	4.	In cross-validation tests, the Random Forest achieved an almost perfect F1 score, while Logistic Regression also performed admirably.
	5.	On the original imbalanced test data, Random Forest maintained high precision and recall, catching 85% of frauds with very few false alarms.
	6.	Logistic Regression matched the recall but generated many more false positives, leading to less reliable alerts.
	7.	The results demonstrate that ensemble tree methods are particularly effective for detecting fraud in uneven datasets.
	8.	By emphasizing precision and recall over overall accuracy, the analysis shows how each model behaves in real-world scenarios.
	9.	Organizations can leverage these insights to deploy fraud detectors that minimize financial losses and customer inconvenience.
	10.	Overall, this study provides a clear roadmap for improving fraud prevention, reducing costs, and enhancing customer trust.


Data Loading and Initial Processing
•	The dataset creditcard.csv was loaded into a pandas DataFrame.
•	A new feature LogAmount was created by applying a log transformation (np.log1p) to the 'Amount' feature to handle its skewed distribution.
Key Findings and EDA Details
1.	Missing Values & Summary Statistics:
o	No missing values were found in the dataset after initial checks and cleaning steps (specifically forward-filling, although initially none were present in the raw data based on the isnull().sum() output).
o	Summary statistics provided an overview of the central tendency, dispersion, and range of features. Features V1-V28 appear to be PCA-transformed, hence centered around zero with varying scales. 'Time' and 'Amount' (and 'LogAmount') have different scales.
2.	Class Distribution:
o	The dataset is highly imbalanced, with a significantly larger number of legitimate transactions (Class 0) compared to fraudulent transactions (Class 1). This imbalance is a critical factor to consider for modeling.
3.	Distribution of Log(Amount):
o	The distribution of LogAmount is approximately bimodal or contains multiple peaks, suggesting different typical transaction sizes after the log transformation.
4.	Distribution of Transaction Time:
o	The distribution of Time shows two distinct peaks, roughly around the start of the period and then another period of high activity later on, potentially representing daily cycles (transactions are recorded over 2 days).
5.	Boxplot of Log(Amount) by Class:
o	Fraudulent transactions tend to have lower LogAmount values on average compared to legitimate transactions, although there is significant overlap. This suggests that smaller transactions are relatively more susceptible to fraud in this dataset.
6.	Top 10 Features by Absolute Correlation with Class:
o	Several of the V features (V17, V14, V12, V10, V16, V3, V9, V11, V4) show the highest absolute correlation with the 'Class' variable. These features are likely important for distinguishing between legitimate and fraudulent transactions.
7.	Mean Feature Values by Class for Top Features:
o	Plotting the mean values of the top correlated features for both classes highlights the difference in their distributions. Fraudulent transactions show notably different mean values for these features compared to legitimate ones, reinforcing their importance.
8.	Transactions per Hour by Class:
o	The distribution of legitimate transactions over time shows clear peaks corresponding to high activity periods. Fraudulent transactions also follow a similar pattern but occur in much lower absolute numbers. Analyzing the rate of fraud per hour is more informative (see point 15).
9.	Correlation Heatmap of Top Features:
o	The heatmap reveals the inter-correlation among the top 10 features. Understanding these relationships is important for potential multicollinearity issues in linear models or for feature selection. Some pairs of V features show moderate to high correlations.
10.	Pairplot of Key Features:
o	A pairplot of the top correlated features allows for visualization of relationships between pairs of features, colored by class. This helps identify potential clusters or separation in feature space, especially how combinations of features might separate fraudulent cases.
11.	Scatter Plot: Log(Amount) vs V17 by Class:
o	This scatter plot specifically examines the relationship between LogAmount and V17. It shows that fraudulent transactions (highlighted by color) tend to occupy specific regions in this 2D space, suggesting a potential separation boundary.
12.	Cumulative Fraud Transactions Over Time:
o	This plot shows the total number of fraud cases accumulated over the recorded time period. It increases steadily, indicating that fraud occurs throughout the duration of the dataset.
13.	Feature Interaction: V14 vs V17 Colored by Class:
o	Similar to the Log(Amount) vs V17 plot, this scatter plot between V14 and V17, colored by class, further illustrates how these key features distinguish between legitimate and fraudulent transactions. Fraud cases often appear clustered in distinct areas.
14.	Fraud Rate by Transaction Amount Bins:
o	This analysis calculates the proportion of fraudulent transactions within different deciles of LogAmount. It shows how the likelihood of fraud varies with the transaction amount, confirming the earlier finding that lower amounts might have a higher fraud rate.
15.	Hour-of-Day Fraud Rate:
o	By converting transaction time to the hour of the day (0-23), this plot shows the fraud rate at different hours. Certain hours show higher fraud rates than others, potentially indicating periods when fraudulent activity is more prevalent.
16.	PCA Projection of Transactions:
o	A 2D PCA projection of a sample of the V features (V1-V28) illustrates how the data is distributed in a lower-dimensional space. While PCA often doesn't completely separate the highly imbalanced fraud class, it can show the overall structure and potential separation.
17.	t-SNE Projection of Transactions:
o	A 2D t-SNE projection of a sample of the V features (V1-V28) aims to preserve local neighborhood structures. t-SNE often does a better job than PCA at visually separating clusters, and in this case, it might show clearer distinctions or clusters related to the fraudulent transactions.
18.	Violin Plots of Key Features by Class:
o	Violin plots for features like V14 and V17 by class provide a detailed view of the distribution shape, median, and quartiles for each class. They visually confirm the significant differences in the distributions of these features between legitimate and fraudulent transactions.
19.	Anomaly Score Distribution (Isolation Forest):
o	An Isolation Forest model was trained to generate anomaly scores. The KDE plot of these scores, separated by class, shows that fraudulent transactions tend to have higher anomaly scores (more negative decision function values), indicating they are more likely to be outliers according to the model.
20.	ROC and Precision-Recall Curves for Isolation Forest Scores:
o	The ROC and Precision-Recall curves evaluate the performance of the Isolation Forest's anomaly score as a classifier. The AUC values (ROC-AUC and PR-AUC) quantify its ability to rank fraudulent transactions higher than legitimate ones. PR-AUC is particularly informative for imbalanced datasets.
Data Cleaning Details
•	Duplicate rows were identified and removed from the DataFrame.
•	Missing values were handled using a forward-fill (ffill) strategy.
Outlier Analysis Details
•	Outliers were identified using the Interquartile Range (IQR) method for numeric columns (excluding 'Class').
•	A summary showed the count of outliers per feature based on the 1.5*IQR rule, with several V features having a high number of identified outliers.
Feature Engineering Details
•	New time-based features (Hour_of_Day, Minute_of_Hour) were extracted from the 'Time' column.
•	An amount-based feature (Amount_Decile) was created by binning LogAmount into 10 deciles.
•	An interaction feature (V1_plus_V2) was created by summing V1 and V2.
•	These engineered features were added to the DataFrame for potential use in modeling.
Baseline Classification Model (Logistic Regression)
•	A Logistic Regression model was trained as a baseline classifier.
•	Data Preparation: Features AmountBin, Hour, Hour24, Hour_of_Day, Minute_of_Hour, Amount_Decile, and AnomalyScore were excluded from the feature set X used for the model, as they were either categorical interval types or derived analysis results not suitable for direct inclusion without further encoding/consideration.
•	Splitting: Data was split into training (80%) and testing (20%) sets using stratification to maintain the class distribution in both sets.
•	Scaling: Features (X_train, X_test) were scaled using StandardScaler to improve the convergence of the Logistic Regression optimizer.
•	Model: A LogisticRegression with max_iter=5000 and class_weight='balanced' was used to handle the class imbalance and allow sufficient iterations for convergence.
•	Evaluation: The model's performance was evaluated using:
o	Classification Report: Providing precision, recall, F1-score, and support for each class (Legit and Fraud). Given the imbalance, the metrics for the 'Fraud' class are particularly important.
o	ROC-AUC Score: A metric less sensitive to imbalance, assessing the model's ability to rank positive (fraud) instances correctly.
•	Evaluation Metric Rationale: ROC-AUC was chosen for its robustness to imbalance, while F1-score was highlighted for balancing precision and recall, both crucial in fraud detection.
Conclusion
The EDA revealed the highly imbalanced nature of the dataset and highlighted key features (several V features, Time, and Amount) that are indicative of fraudulent transactions. Visualizations provided insights into the distributions and relationships between features and their connection to the target variable. Feature engineering created potentially useful new features. A baseline Logistic Regression model was successfully trained and evaluated, providing initial performance metrics while addressing data issues like scaling and class imbalance. These findings and the baseline model serve as a strong foundation for further model development and comparison.

![image](https://github.com/user-attachments/assets/af8ac7d7-124d-4669-aabc-de5d7bb2c1b6)
