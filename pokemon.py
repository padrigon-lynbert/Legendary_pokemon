import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# collection
dataset_file = "D:\manualCDmanagement\codes\Projects\VMs\skl algorithms\Logistic Regression/00_datasets/Pokemon"
file_name = 'pokemon.csv'
dataset = pd.read_csv(os.path.join(dataset_file, file_name))
df = pd.DataFrame(dataset)

# cleaning
separate = ['Name', 'Type 1', 'Type 2', 'Legendary']

for column in df.columns:
    if column in separate:
        df[column], _ = pd.factorize(df[column])

# splitting
X = df.drop('Legendary', axis=1)
y = df['Legendary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fitting
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_pos = model.predict_proba(X_test_scaled)[:, 1]

import seaborn as sns
import matplotlib.pyplot as plt

manual_threshold = 0.7
y_pred = (y_pred_pos >= manual_threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)