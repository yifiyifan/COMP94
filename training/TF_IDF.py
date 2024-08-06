import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load data
train_path = 'final_gpt_match_output_clean_res_20240727213717.csv'
test_path = 'holdout_gpt_match_output_checkpoint_40_20240727235744.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Label mapping
label_mapping = {
    'poor fit': 0,
    'good fit': 1,
    'Good Fit': 1,
    'No Fit': 0,
    'Potential Fit': 0
}
df_train['label'] = df_train['label'].map(label_mapping)
df_test['label'] = df_test['label'].map(label_mapping)

# Balance the classes in training data
min_class_size = df_train['label'].value_counts().min()
df_train = df_train.groupby('label').apply(lambda x: x.sample(min_class_size, random_state=42)).reset_index(drop=True)

# Combine job descriptions and resumes for vectorizer fitting
combined_text = df_train['job_desc'].tolist() + df_train['formatted_resume'].tolist()
vectorizer = TfidfVectorizer()
vectorizer.fit(combined_text)

# Transform job descriptions and resumes
job_desc_tfidf = vectorizer.transform(df_train['job_desc'].tolist())
resume_tfidf = vectorizer.transform(df_train['resume_skills'].tolist())

# Calculate cosine similarities
similarities = []
for i in range(len(df_train)):
    similarity = cosine_similarity(job_desc_tfidf[i], resume_tfidf[i])
    similarities.append(similarity[0][0])
df_train['similarity'] = similarities

# Features and labels
X = df_train[['similarity']]
y = df_train['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Prepare test data
job_desc_test = df_test['job_desc'].tolist()
resume_test = df_test['resume_skills'].tolist()

# Transform job descriptions and resumes for test data
job_desc_test_tfidf = vectorizer.transform(job_desc_test)
resume_test_tfidf = vectorizer.transform(resume_test)

# Calculate cosine similarities for test data
similarities_test = []
for i in range(len(df_test)):
    similarity = cosine_similarity(job_desc_test_tfidf[i], resume_test_tfidf[i])
    similarities_test.append(similarity[0][0])
df_test['similarity'] = similarities_test

X_test = df_test[['similarity']]
y_test = df_test['label']
y_test_encoded = label_encoder.transform(y_test)

# Make predictions
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_labels)
report = classification_report(y_test, y_pred_labels)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)