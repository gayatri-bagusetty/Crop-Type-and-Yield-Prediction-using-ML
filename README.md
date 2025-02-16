**DATASET:**

Dataset is downloaded form the below website.

Link - 

**CODE DETAILS:**

_Import Libraries:_ Libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn are imported.

_Load Dataset:_ The dataset is loaded from Google Drive using pandas.read_csv().

_EDA and Preprocessing:_
Displayed shape, column info, and statistics.
Checked for missing and duplicate values.
Encoded categorical features using LabelEncoder.
Scaled numerical features using StandardScaler.

_Data Visualization:_
Created a heatmap to show feature correlations.
Detected outliers using histograms and boxplots.

_Data Splitting:_
Split dataset into training and testing sets with train_test_split().
Model Training and Evaluation:

_Classification Models (Crop Type):_ SVC, Random Forest, Decision Tree, Logistic Regression, XGBoost.

_Regression Models (Yield Prediction):_ SVR, Random Forest Regressor, Gradient Boosting, Linear Regression, XGBoost.

Printed accuracy (classification) and MSE (regression).

Visualized results using bar plots.


**_Libraries Used:_** pandas, numpy, matplotlib, seaborn, sklearn, xgboost

**_Models Used:_**
_Classification:_ SVC, Random Forest, Decision Tree, Logistic Regression, XGBoost
_Regression:_ SVR, Random Forest Regressor, Gradient Boosting, Linear Regression, XGBoost

_**Outputs:**_
Model accuracies and errors
Visualizations of performance

_Instructions to Run:_
Upload the dataset to your Google Drive.
Change the path variable to your dataset location.
Run the code in Google Colab step by step.

_Dataset contains:_

Features: Crop, Season, State, Area, Fertilizer, Pesticide, Yield

Preprocessing: Label encoding, scaling, outlier detection.

_**Conclusion:**_
SVC and Random Forest performed well in classification.
Gradient Boosting and XGBoost yielded the best results for yield prediction
