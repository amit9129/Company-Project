import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = {
    'Employee_ID': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010',
                    'E011', 'E012', 'E013', 'E014', 'E015', 'E016', 'E017', 'E018', 'E019', 'E020',
                    'E021', 'E022', 'E023', 'E024', 'E025', 'E026', 'E027', 'E028', 'E029', 'E030'],
    'Name': ['John Doe', 'Jane Smith', 'Tom Brown', 'Emily Davis', 'Michael Lee', 'Sarah Khan', 
             'David Wong', 'Lily Zhang', 'Chris Hall', 'Rachel Kim', 'Jake Turner', 'Olivia Lee', 
             'Daniel Roy', 'Mia Clark', 'Ethan Scott', 'Zoe Moore', 'Aaron Fox', 'Bella Ross', 
             'Kevin Diaz', 'Fiona Gray', 'Noah Cole', 'Sofia King', 'Alex Reed', 'Lauren Hill', 
             'Ryan Baker', 'Ivy Evans', 'Lucas Reed', 'Nina Scott', 'Owen Perry', 'Grace Cox'],
    'Age': [34, 29, 45, 31, 27, 40, 52, 26, 38, 33, 44, 30, 37, 41, 28, 35, 48, 29, 36, 39, 31, 27, 42, 34, 46, 28, 30, 43, 32, 26],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 
               'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Department': ['Sales', 'Marketing', 'HR', 'IT', 'IT', 'Finance', 'Operations', 'Marketing', 'Sales', 'Customer Service',
                   'Finance', 'IT', 'Marketing', 'HR', 'Operations', 'Finance', 'Sales', 'Marketing', 'IT', 'Customer Service',
                   'Operations', 'HR', 'Finance', 'IT', 'Sales', 'Marketing', 'IT', 'Customer Service', 'Finance', 'HR'],
    'Job_Role': ['Sales Manager', 'Marketing Exec', 'HR Manager', 'Software Eng.', 'Data Analyst', 'Accountant', 'Ops Manager', 
                 'Content Creator', 'Sales Associate', 'CS Rep', 'Financial Analyst', 'DevOps Engineer', 'SEO Specialist', 
                 'Recruiter', 'Ops Coordinator', 'Accountant', 'Sales Director', 'Digital Marketer', 'Network Admin', 'CS Manager', 
                 'Ops Specialist', 'HR Coordinator', 'Auditor', 'QA Engineer', 'Sales Manager', 'Marketing Assistant', 
                 'System Admin', 'CS Rep', 'Investment Analyst', 'HR Specialist'],
    'Tenure_Years': [5, 2, 7, 3, 1, 8, 10, 1.5, 6, 4, 9, 3, 5, 7, 2, 6, 11, 3.5, 5, 8, 4, 2, 10, 5, 9, 2, 3, 9, 4, 1],
    'Reason_for_Leaving': ['Better Opportunity', 'Personal Reasons', 'Retirement', 'Relocation', 'Dissatisfaction', 'Career Change', 
                           'Health Issues', 'Better Opportunity', 'Better Opportunity', 'Relocation', 'Dissatisfaction', 'Career Change', 
                           'Personal Reasons', 'Relocation', 'Dissatisfaction', 'Better Opportunity', 'Retirement', 'Career Change', 
                           'Better Opportunity', 'Health Issues', 'Relocation', 'Personal Reasons', 'Career Change', 
                           'Better Opportunity', 'Dissatisfaction', 'Personal Reasons', 'Relocation', 'Retirement', 
                           'Career Change', 'Dissatisfaction'],
    'Turnover_Date': ['1/15/2024', '11-02-23', '2/20/2024', '03-12-24', '12/29/2023', '8/16/2023', '10-05-23', '9/21/2023', 
                      '01-08-24', '4/15/2024', '5/25/2024', '12-01-23', '06-10-24', '07-01-24', '11-11-23', '2/17/2024', 
                      '8/20/2024', '7/22/2023', '9/15/2023', '1/19/2024', '11-04-23', '6/18/2023', '3/22/2024', '02-11-24', 
                      '04-04-24', '10-09-23', '1/31/2024', '6/18/2024', '12-03-23', '8/24/2023'],
    'Performance_Rating': [4, 3, 5, 2, 3, 4, 5, 4, 3, 3, 2, 4, 3, 4, 2, 4, 5, 3, 3, 4, 3, 2, 4, 3, 3, 3, 3, 4, 3, 2],
    'Salary': ['65,000', '52,000', '80,000', '72,000', '58,000', '77,000', '88,000', '50,000', '55,000', '48,000',
               '82,000', '68,000', '60,000', '70,000', '50,000', '76,000', '95,000', '63,000', '65,000', '78,000',
               '62,000', '54,000', '85,000', '70,000', '78,000', '55,000', '68,000', '64,000', '75,000', '50,000']
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
df['Salary'] = df['Salary'].str.replace(',', '').astype(float)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Department'] = label_encoder.fit_transform(df['Department'])
df['Job_Role'] = label_encoder.fit_transform(df['Job_Role'])
df['Reason_for_Leaving'] = label_encoder.fit_transform(df['Reason_for_Leaving'])

scaler = StandardScaler()
df[['Age', 'Tenure_Years', 'Salary']] = scaler.fit_transform(df[['Age', 'Tenure_Years', 'Salary']])

# Step 3: Define Features and Target
X = df.drop(['Employee_ID', 'Name', 'Turnover_Date'], axis=1)
y = df['Reason_for_Leaving']  # Assuming we're predicting the reason for leaving as a proxy for turnover

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Step 7: Feature Importance
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()

# Step 8: Model Tuning (Optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Step 9: Prediction on New Data (Example)
new_employee = pd.DataFrame({
    'Age': [35],
    'Gender': label_encoder.transform(['Female']),
    'Department': label_encoder.transform(['IT']),
    'Job_Role': label_encoder.transform(['Software Eng.']),
    'Tenure_Years': scaler.transform([[5]]),
    'Performance_Rating': [4],
    'Salary': scaler.transform([[70000]])
})

new_prediction = grid_search.best_estimator_.predict(new_employee)
print(f"Predicted Reason for Leaving: {label_encoder.inverse_transform(new_prediction)[0]}")
