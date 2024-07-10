import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Generating sample data
np.random.seed(42)
n_samples = 1000
data = {
    'feature1': np.random.rand(n_samples) * 100,
    'feature2': np.random.rand(n_samples) * 50,
    'feature3': np.random.rand(n_samples) * 200,
    'feature4': np.random.rand(n_samples) * 30,
    'feature5': np.random.rand(n_samples) * 10,
    'target': np.random.choice([0, 1], size=n_samples)  # Binary target variable
}
df = pd.DataFrame(data)

# Feature Engineering
def feature_engineering(df):
    df['new_feature1'] = df['feature1'] * df['feature2']
    df['new_feature2'] = df['feature3'] / (df['feature4'] + 1)
    df['new_feature3'] = np.log(df['feature5'] + 1)
    return df

df = feature_engineering(df)

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection using PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance by PCA components:", pca.explained_variance_ratio_)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy with PCA features:", accuracy_score(y_test, y_pred))

# Feature Importance
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
model_orig = RandomForestClassifier(n_estimators=100, random_state=42)
model_orig.fit(X_train_orig, y_train_orig)

# Select important features
selector = SelectFromModel(model_orig, prefit=True)
X_important_train = selector.transform(X_train_orig)
X_important_test = selector.transform(X_test_orig)

# Retrain model with selected features
model_important = RandomForestClassifier(n_estimators=100, random_state=42)
model_important.fit(X_important_train, y_train_orig)

# Evaluate model with selected features
y_pred_important = model_important.predict(X_important_test)
print("Accuracy with important features:", accuracy_score(y_test_orig, y_pred_important))

# Get feature importances
importances = model_orig.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Select top n important features (e.g., top 5)
top_features = feature_importance_df['Feature'][:5].values
X_top_features = X[top_features]

# Split the data with top features
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top_features, y, test_size=0.2, random_state=42)

# Train model on top features
model_top = RandomForestClassifier(n_estimators=100, random_state=42)
model_top.fit(X_train_top, y_train_top)

# Evaluate model on top features
y_pred_top = model_top.predict(X_test_top)
print("Accuracy with top features:", accuracy_score(y_test_top, y_pred_top))
