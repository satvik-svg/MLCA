# Smart Study Recommendation and Student Performance Prediction

## 1. Project Overview

This project presents a machine learning-based student analytics system that predicts academic performance and supports targeted improvement planning. The implementation includes both:

- A trained classification model for student performance prediction.
- A Streamlit web interface for live input, prediction, and visual explanation.

The deployed model artifact used by the web app is:

- tuned_random_forest_model.pkl

Dataset reference:

- https://www.kaggle.com/datasets/dhrumilgohel/student-performace-dataset

## 2. Problem Statement

The system predicts a student performance category from internal academic indicators and attendance behavior. Predicted categories include:

- Fail
- Pass
- Good
- Excellent

## 3. Input Features Used by the Deployed Model

The model expects the following features in this exact order:

1. sem_eval_lec_test_1_mark
2. sem_eval_lab_test_1_mark
3. semester_evaluation_mid_mark
4. sem_eval_lec_test_2_mark
5. sem_eval_lab_test_2_mark
6. semester_evaluation_pre_gtu_mark
7. semester_evaluation_internal_mark
8. attendance_percentage

Derived feature formula:

attendance_percentage = (sem_present_count / (sem_present_count + sem_absent_count)) * 100

The web app collects sem_present_count and sem_absent_count and computes attendance_percentage internally before prediction.

## 4. Mandatory ML Pipeline Steps

This project implementation should clearly demonstrate and document the following stages.

### I. Data Pre-processing and EDA

1. Handling Missing Values
- Missing values should be handled using imputation or row/column dropping based on data loss impact.
- Recommended documentation: percentage of missing values per feature and rationale for selected strategy.

2. Exploratory Data Analysis (EDA)
- Use visualizations to explain data quality and behavior, such as:
	- Distribution plots (histogram/KDE)
	- Correlation heatmap
	- Outlier checks (boxplots)
	- Relationship trends (scatter/line plots)

3. Feature Engineering
- Document all transformations and new features (for example, attendance_percentage).
- If scaling or encoding is used, explain where and why it is required.

### II. Model Selection

1. Algorithm Choice
- Compare at least two algorithms and justify the final selection.
- Example comparison: Random Forest vs Logistic Regression or SVM.

2. Hyperparameter Tuning
- Demonstrate tuning using GridSearchCV or RandomizedSearchCV.
- Record best parameters and validation performance.

### III. Evaluation Metrics

1. Metrics Selection
- Choose metrics appropriate to the problem type (Accuracy, F1-score, RMSE, MAE, Silhouette Score, etc.).

2. Metric Justification
- Explain why the chosen metric is the right success indicator for this use case.
- For class imbalance, include F1-score in addition to Accuracy.

## 5. Implementation Summary

The current web application includes:

- Structured input form for all required model fields.
- Automatic computation of attendance_percentage.
- Prediction output with confidence visualization.
- Additional explanation charts for interpretability.

## 6. Submission and Implementation Requirements

1. Working Project
- The implemented project must run on the laptop during viva/presentation.
- All required files (model, scripts, dependencies) must be available locally.

2. Code Quality
- Implementation should be maintained in Jupyter Notebooks or Python scripts with clear structure.

3. Documentation
- A brief report or README that summarizes the full ML pipeline is required.

4. Frontend (Optional)
- A frontend demonstration is optional but recommended to present the workflow clearly.

## 7. Project Structure

- app.py
- tuned_random_forest_model.pkl
- requirements.txt
- README.md
- implememntaion.md

## 8. Setup and Execution

1. Install dependencies:

pip install -r requirements.txt

2. Run the web application:

streamlit run app.py

3. Open the local URL shown in terminal and test with sample student inputs.

## 9. Viva Presentation Notes

During viva, be ready to explain:

- Data cleaning and preprocessing decisions.
- EDA findings and how they informed modeling.
- Why the final algorithm was selected over alternatives.
- Hyperparameter tuning method and best configuration.
- Why selected evaluation metrics reflect real project success.
- Live end-to-end execution in the frontend.
