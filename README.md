# Drug Consumption Risk Classification Using Machine Learning

## 📌 Project Overview

This project explores the use of Machine Learning to predict drug consumption patterns based on an individual's demographic information and personality traits.

The primary objective is not just to build an accurate classification model, but also to understand **why a model makes particular predictions** using Explainable AI (XAI) techniques.

Two explainability methods, **SHAP and LIME**, are implemented and compared to investigate whether they agree on the features influencing model predictions.

The project focuses on two classification tasks:
- **Cannabis Consumption:** Predicting whether an individual has used Cannabis.
- **Depressant Consumption:** Predicting whether an individual has used at least one depressant substance.

## 🎯 Objectives

- Analyze the relationship between personality traits, demographics, and drug consumption.
- Develop machine learning models for binary classification.
- Address class imbalance using SMOTE.
- Perform feature selection using RFECV.
- Compare multiple machine learning algorithms.
- Interpret model predictions using SHAP and LIME.
- Evaluate the agreement between different explainability techniques.

## 📂 Dataset

The project uses the **UCI Drug Consumption Dataset**, containing information about 1,884 participants.

The dataset includes:

| Feature Category | Description |
|---|---|
| Demographics | Age, Gender, Education, Country, Ethnicity |
| Personality Traits | Nscore, Escore, Oscore, AScore, Cscore, Impulsive, SS |
| Drug Consumption | Usage history of 19 drug-related categories |

The drug consumption columns use a categorical scale from `CL0` to `CL6`:

- `CL0` – Never used
- `CL1` – Used over a decade ago
- `CL2` to `CL6` – Used within the last decade, with different recency/frequency levels

**Binary Classification Rule:**

- User: `CL2` or higher
- Non-user: `CL0` or `CL1`

## 🛠️ Technologies & Libraries

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- SHAP
- LIME
- SciPy

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Removed the participant ID column.
- Converted Age and Education into ordinal numerical values.
- Encoded Gender into binary values.
- Applied One-Hot Encoding to Country and Ethnicity.
- Retained existing numerical personality scores.

### 2. Exploratory Data Analysis (EDA)
- Examined personality trait distributions.
- Compared Cannabis users and non-users.
- Visualized differences across personality dimensions.

### 3. Train-Test Split
- 80% training data
- 20% testing data
- Stratified sampling to preserve class proportions.

### 4. Handling Class Imbalance
Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training dataset.

The test dataset was kept unchanged to ensure realistic evaluation.

### 5. Feature Selection
Used **Recursive Feature Elimination with Cross-Validation (RFECV)** with a Random Forest estimator to identify relevant features.

### 6. Model Training & Comparison

Six machine learning algorithms were trained and optimized using GridSearchCV:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Gaussian Naive Bayes
4. K-Nearest Neighbors (KNN)
5. Random Forest
6. XGBoost

A Soft Voting Ensemble was also implemented to combine predictions from all six models.

### 7. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

**Random Forest** achieved the best F1-score among the individual models and was selected for further explainability analysis.

### 8. Explainable AI (XAI)

Two explainability techniques were implemented:

**SHAP (SHapley Additive exPlanations)**
- Identifies globally important features.
- Shows how features influence predictions.
- Provides both positive and negative feature contributions.

**LIME (Local Interpretable Model-agnostic Explanations)**
- Explains individual predictions.
- Builds local interpretable approximations around individual observations.
- Identifies features influencing a particular prediction.

### 9. SHAP vs LIME Comparison

The two explainability methods were compared using:

- Global feature importance ranking correlation.
- Top feature overlap.
- Feature contribution direction agreement.
- Individual prediction explanation correlation.

The comparison helps determine whether different explainability techniques provide consistent interpretations.

## 📊 Key Findings

### Cannabis Classification

- Random Forest performed best among the individual models.
- SHAP and LIME shared 8 of their top 10 important features.
- Personality traits such as Sensation-Seeking, Openness, and Conscientiousness showed strong agreement between the explanation methods.
- Country-related features showed greater disagreement in contribution direction.

### Depressant Classification

Depressants were defined as the combined category of:
- Benzodiazepines
- Heroin
- Methadone

A Random Forest classifier was trained using the same preprocessing, train-test splitting, and SMOTE approach.

SHAP and LIME demonstrated stronger overall agreement for Depressant classification compared with Cannabis classification.

## 📁 Project Structure

```text
Drug-Consumption-Risk-Classification/
│
├── Drug_Consumption_Risk_Classification.ipynb
├── Drug_Consumption.csv
└── README.md
```

## 🚀 Installation & Execution

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Drug-Consumption-Risk-Classification
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost shap lime scipy jupyter
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `Drug_Consumption_Risk_Classification.ipynb` and execute the cells sequentially.

**Note:** Ensure that `Drug_Consumption.csv` is present in the same directory as the notebook.

## ⚠️ Limitations & Ethical Considerations

- The dataset is based on self-reported survey responses.
- Predictions represent statistical associations, not causal relationships.
- Demographic features may introduce bias into model predictions.
- SHAP and LIME provide explanations of model behaviour, not proof of real-world causal factors.
- The model is intended for academic and research purposes, not clinical diagnosis or individual-level risk assessment.

## 🔮 Future Improvements

- Experiment with additional classification algorithms.
- Explore alternative feature selection techniques.
- Perform hyperparameter optimization with broader search spaces.
- Investigate fairness and demographic bias.
- Extend the explainability comparison to additional drug categories.
- Evaluate model robustness using additional datasets.

## 👨‍💻 Author

**Aarush**

Machine Learning | Explainable AI | Data Science

---

*This project demonstrates how machine learning and Explainable AI can work together to build interpretable classification systems while highlighting the importance of responsible model evaluation.*
