# Bank Marketing Classification - Machine Learning Assignment 2

## Name: Pranav Shivam
## UID/Roll/Addmission Number: 2025AA05638
**Program**: M.Tech (S1-25_AIMLCZG565)

## Problem Statement

This project implements multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution. The goal is to compare the performance of six different classification algorithms and deploy an interactive web application for model evaluation.

The prediction task is a **binary classification problem** where:
- **Target Variable**: `deposit` (yes/no - will the client subscribe to a term deposit?)
- **Objective**: Build and compare multiple ML models to achieve the highest prediction accuracy and F1 score

---

## Dataset Description

**Dataset Source**: Bank Marketing Dataset (UCI Machine Learning Repository / Kaggle)

**Dataset Overview**:
- **Data Source**: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
- **Total Instances**: 11,162 records
- **Total Features**: 17 (16 features + 1 target variable)
- **Feature Types**: Mixed (Categorical and Numerical)
- **Target Distribution**: Imbalanced classification problem
- **Missing Values**: None

### Features Description:

#### Categorical Features (9):
1. **job**: Type of job (admin, technician, services, management, retired, blue-collar, unemployed, entrepreneur, housemaid, unknown, self-employed, student)
2. **marital**: Marital status (married, single, divorced)
3. **education**: Education level (primary, secondary, tertiary, unknown)
4. **default**: Has credit in default? (yes, no)
5. **housing**: Has housing loan? (yes, no)
6. **loan**: Has personal loan? (yes, no)
7. **contact**: Contact communication type (cellular, telephone, unknown)
8. **month**: Last contact month of year (jan, feb, mar, ..., dec)
9. **poutcome**: Outcome of previous marketing campaign (success, failure, other, unknown)

#### Numerical Features (8):
1. **age**: Age of the client (in years)
2. **balance**: Average yearly balance (in euros)
3. **day**: Last contact day of the month
4. **duration**: Last contact duration (in seconds)
5. **campaign**: Number of contacts performed during this campaign
6. **pdays**: Number of days since client was last contacted (-1 means not previously contacted)
7. **previous**: Number of contacts performed before this campaign
8. **previous_contact**: Engineered feature (0 if pdays = -1, else 1)

#### Target Variable:
- **deposit**: Has the client subscribed to a term deposit? (yes=1, no=0)

### Data Preprocessing:
- Binary encoding of target variable: `yes → 1`, `no → 0`
- Feature engineering: Created `previous_contact` from `pdays`
- Replaced `-1` values in `pdays` with `0`
- Standard scaling applied to numerical features
- One-hot encoding applied to categorical features

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.8262 | 0.9071 | 0.8284 | 0.7987 | 0.8133 | 0.6513 |
| **Decision Tree** | 0.7922 | 0.7909 | 0.7895 | 0.7656 | 0.7774 | 0.5829 |
| **K-Nearest Neighbors** | 0.8164 | 0.8795 | 0.8189 | 0.7864 | 0.8023 | 0.6315 |
| **Naive Bayes** | 0.7179 | 0.8015 | 0.7737 | 0.5718 | 0.6576 | 0.4409 |
| **Random Forest (Ensemble)** | 0.8531 | 0.9208 | 0.8219 | 0.8809 | 0.8504 | 0.7081 |
| **XGBoost (Ensemble)** | 0.8625 | 0.9245 | 0.8392 | 0.8781 | 0.8582 | 0.7256 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression demonstrates strong overall performance with an accuracy of 82.62% and excellent AUC of 0.9071, indicating good discrimination ability between classes. The balanced precision (0.8284) and recall (0.7987) suggest the model handles both false positives and false negatives reasonably well. It serves as a solid baseline model with interpretable coefficients and fast training time. The MCC of 0.6513 confirms good correlation between predictions and actual values. |
| **Decision Tree** | Decision Tree shows moderate performance with 79.22% accuracy and the lowest AUC (0.7909) among all models, indicating weaker probability estimates. While precision (0.7895) and recall (0.7656) are balanced, the overall metrics suggest the model may be overfitting to training data despite having reasonable test performance. The MCC of 0.5829 is the second-lowest, indicating moderate predictive quality. Decision trees are prone to high variance and may not generalize as well as ensemble methods. |
| **K-Nearest Neighbors** | KNN achieves 81.64% accuracy with a strong AUC of 0.8795, showing good discriminative power. The model maintains balanced precision (0.8189) and recall (0.7864), performing competitively with logistic regression. However, KNN is computationally expensive during prediction time as it requires distance calculations with all training samples. The MCC of 0.6315 indicates reliable predictions. Performance could vary significantly with different values of k and distance metrics. |
| **Naive Bayes** | Naive Bayes shows the weakest performance overall with only 71.79% accuracy, despite maintaining a reasonable AUC of 0.8015. The model suffers from notably low recall (0.5718), meaning it misses many positive cases (false negatives). While precision is acceptable (0.7737), the low F1 score (0.6576) and MCC (0.4409) indicate poor overall predictive quality. The strong independence assumption of Naive Bayes likely doesn't hold well for this dataset, as banking features are often correlated. |
| **Random Forest (Ensemble)** | Random Forest delivers excellent performance with 85.31% accuracy and strong AUC of 0.9208, ranking as the second-best model. It achieves outstanding recall (0.8809), meaning it successfully identifies most positive cases, though with slightly lower precision (0.8219). The F1 score of 0.8504 and MCC of 0.7081 demonstrate robust and balanced predictions. As an ensemble method, it reduces overfitting through bagging and provides good feature importance insights. The model is more interpretable than XGBoost while maintaining competitive performance. |
| **XGBoost (Ensemble)** | XGBoost achieves the best overall performance with 86.25% accuracy and the highest AUC (0.9245), demonstrating superior discriminative ability. It balances high precision (0.8392) and excellent recall (0.8781), resulting in the best F1 score (0.8582) and MCC (0.7256) among all models. The gradient boosting approach with regularization helps prevent overfitting while capturing complex non-linear relationships. XGBoost's iterative learning from previous errors makes it the most powerful model for this classification task, though at the cost of longer training time and reduced interpretability. |

---

## Key Findings and Insights

### Overall Model Ranking (by F1 Score):
1. **XGBoost** (F1: 0.8582) - Best overall performer
2. **Random Forest** (F1: 0.8504) - Strong second place
3. **Logistic Regression** (F1: 0.8133) - Good baseline
4. **K-Nearest Neighbors** (F1: 0.8023) - Competitive performance
5. **Decision Tree** (F1: 0.7774) - Moderate performance
6. **Naive Bayes** (F1: 0.6576) - Weakest performer

### Key Insights:
- **Ensemble methods (Random Forest and XGBoost) significantly outperform individual models**, demonstrating the power of combining multiple weak learners
- **XGBoost shows the best balance** between precision and recall, making it ideal for this banking application where both false positives and false negatives have costs
- **Logistic Regression performs surprisingly well** as a simple linear model, providing a good interpretable alternative
- **Naive Bayes struggles significantly**, likely due to violated independence assumptions in banking features
- **All models achieve AUC > 0.79**, indicating reasonable discriminative ability across the board
- **The dataset benefits from ensemble approaches**, suggesting complex non-linear relationships between features and target

---

## Project Structure

```
ml-bank-marketing-app-assignment-2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── model/
    ├── models.py                   # Model training and evaluation pipeline
    ├── helper.py                   # Data loading and preprocessing utilities
    └── utils/
    │       └── evaluation_metrics.py   # Metrics calculation functions
    └── datas/
         ├── load_data.py               # Data loader class
         └── bank.csv                   # Bank marketing dataset
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/Pranav-Shivam/ml-bank-marketing-app-assignment-2/
cd ml-bank-marketing-app-assignment-2
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Web Application

1. **Model Selection**: Choose from 6 classification models using the dropdown
2. **Upload Test Data**: Upload a CSV file with the same features as the training data (optional)
   - If no file is uploaded, the app uses the internal test split
3. **View Results**: 
   - Predictions table
   - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
   - Confusion matrix visualization
   - Detailed classification report

### Training Models from Command Line

```bash
python models/models.py
```

This will train all 6 models and display their evaluation metrics.

---

## Streamlit App Features

✅ **Dataset Upload**: Upload custom CSV test data for evaluation  
✅ **Model Selection**: Dropdown to choose from 6 trained models  
✅ **Evaluation Metrics**: Display all 6 required metrics in an organized layout  
✅ **Confusion Matrix**: Visual representation of model predictions  
✅ **Classification Report**: Detailed per-class performance metrics  
✅ **Data Preview**: Shows uploaded dataset structure and sample rows  
✅ **Error Handling**: Validates uploaded data for required features  

---

## Model Implementation Details

### 1. Logistic Regression
- **Library**: scikit-learn
- **Parameters**: `max_iter=1000`, `random_state=42`
- **Use Case**: Linear baseline model with interpretable coefficients

### 2. Decision Tree Classifier
- **Library**: scikit-learn
- **Parameters**: `random_state=42`
- **Use Case**: Non-linear model with visual interpretability

### 3. K-Nearest Neighbors
- **Library**: scikit-learn
- **Parameters**: Default (5 neighbors)
- **Use Case**: Instance-based learning approach

### 4. Naive Bayes (Gaussian)
- **Library**: scikit-learn
- **Type**: GaussianNB for continuous features
- **Use Case**: Probabilistic classifier with independence assumption

### 5. Random Forest (Ensemble)
- **Library**: scikit-learn
- **Parameters**: `n_estimators=200`, `random_state=42`, `n_jobs=-1`
- **Use Case**: Bagging ensemble with reduced overfitting

### 6. XGBoost (Ensemble)
- **Library**: xgboost
- **Parameters**: `n_estimators=200`, `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`
- **Use Case**: Gradient boosting with best overall performance

---

## Evaluation Metrics

All models are evaluated using the following metrics:

1. **Accuracy**: Overall correctness of predictions
2. **AUC Score**: Area Under the ROC Curve - discrimination ability
3. **Precision**: Ratio of true positives to predicted positives
4. **Recall**: Ratio of true positives to actual positives
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC Score**: Matthews Correlation Coefficient - quality of binary classification

---

## Deployment

### Streamlit Community Cloud

The application is deployed on Streamlit Community Cloud (FREE tier).

**Live App URL**: [[Streamlit App Link](https://ml-bank-marketing-app-assignment-2-qxdaiehpynz2axjavecryo.streamlit.app/)]

**Deployment Steps**:
1. Push code to GitHub repository
2. Visit https://streamlit.io/cloud
3. Sign in with GitHub account
4. Click "New App"
5. Select repository, branch (main), and app.py
6. Click "Deploy"

The app typically deploys within 2-3 minutes.

---

## Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Gradient boosting implementation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Visualization library
- **seaborn**: Statistical data visualization

---

## Performance Considerations

- **Training Time**: Models are trained once and cached using `@st.cache_resource`
- **Prediction Speed**: All models provide fast inference on test data
- **Memory Usage**: Efficient sparse matrix handling for one-hot encoded features
- **Scalability**: Can handle datasets with up to ~50,000 rows in free Streamlit tier

---


## Assignment Compliance

This project fulfills all requirements for **Machine Learning Assignment 2**:

✅ **Dataset Requirements**: 
- 11,162 instances (> 500 required)
- 17 features (> 12 required)

✅ **Model Implementation**: All 6 required models implemented
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

✅ **Evaluation Metrics**: All 6 metrics calculated
- Accuracy, AUC, Precision, Recall, F1, MCC

✅ **Streamlit App**: All required features
- Dataset upload option
- Model selection dropdown
- Display evaluation metrics
- Confusion matrix and classification report

✅ **Documentation**: Complete README with
- Problem statement
- Dataset description
- Model comparison table
- Performance observations

✅ **Deployment**: Live on Streamlit Community Cloud

---

## Author

**Name**: Pranav Shivam  
**UID/Roll/Addmission Number**:  2025AA05638   
**Program**: M.Tech (S1-25_AIMLCZG565)
**Institution**: BITS Pilani - Work Integrated Learning Programmes Division  
**Course**: Machine Learning  
**Assignment**: Assignment 2  

---

## License

This project is created for academic purposes as part of the Machine Learning course at BITS Pilani.

---

## Acknowledgments

- BITS Pilani for providing the assignment structure
- UCI Machine Learning Repository for the Bank Marketing dataset
- Streamlit team for the excellent deployment platform
- scikit-learn and XGBoost communities for robust ML libraries

---

## Contact

For questions or issues, please contact: 2025aa05638@wilp.bits-pilani.ac.in / pranav.shivam@cognida.ai 

**GitHub Repository**: [\[GitHub Repo Link\]](https://github.com/Pranav-Shivam/ml-bank-marketing-app-assignment-2/)  
**Live Application**: [\[Streamlit App Link\]](https://ml-bank-marketing-app-assignment-2-qxdaiehpynz2axjavecryo.streamlit.app/)  

---

*Last Updated: 15, February 2026*