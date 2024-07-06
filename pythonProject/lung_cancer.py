import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = 'lung_cancer_dataset.csv'
df = pd.read_csv(data)

# Check for missing values
print(df.isnull().sum())


# Encode categorical variables
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])

# Split the data into features and target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Neural Network': MLPClassifier(max_iter=500)
}

# Dictionary to store metrics
metrics = {}


# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # tp / (tp + fn)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # tn / (tn + fp)

    # Store metrics
    metrics[name] = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

    print(f'Model: {name}')
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Sensitivity: {sensitivity:.2f}')
    print(f'Specificity: {specificity:.2f}')

      # Plot ROC curve if available
    if y_proba is not None:
        y_test_binary = (y_test == 1).astype(int)  # Convert y_test to binary
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test_binary, y_proba):.2f})')

# Plot ROC curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()


# Plot performance metrics
metrics_df = pd.DataFrame(metrics).T
metrics_df[['accuracy', 'sensitivity', 'specificity']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

# Save the best model and scaler (assuming Random Forest here)
best_model = models['Random Forest']
joblib.dump(best_model, 'best_model_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')


metrics = {"accuracy": 0.95, "precision": 0.93}  # Example metrics
joblib.dump(metrics, 'model_metrics.pkl')

