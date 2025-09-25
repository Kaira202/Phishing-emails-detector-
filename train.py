import sys
import pandas as pd
import numpy as np
import re
import string
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

import joblib 


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("SpamAssasin.csv")

# fill missing
df['subject'] = df['subject'].fillna("").astype(str)
df['receiver'] = df.get('receiver', "").fillna("unknown").astype(str)
df['body'] = df['body'].fillna("").astype(str)

# Keep raw copies
df['raw_subject'] = df['subject'].copy()
df['raw_body'] = df['body'].copy()
df['raw_text'] = df['raw_subject'] + " " + df['raw_body']

# ------------------ CLEANING ------------------
def clean_text_for_tfidf(text):
    if pd.isnull(text) or text == "":
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_subject'] = df['raw_subject'].apply(clean_text_for_tfidf)
df['clean_body'] = df['raw_body'].apply(clean_text_for_tfidf)
df['final_text'] = (df['clean_subject'] + " " + df['clean_body']).str.strip()

# ------------------ FEATURE EXTRACTION ------------------
SHORTENERS = ['bit.ly', 't.co', 'tinyurl', 'goo.gl', 'ow.ly', 'buff.ly', 'is.gd']
URL_REGEX = r'(https?://[^\s]+|www\.[^\s]+)'
EMAIL_REGEX = r'[\w\.-]+@[\w\.-]+\.\w+'

def extract_url_info(text):
    urls = re.findall(URL_REGEX, text, flags=re.IGNORECASE)
    urls = [u.rstrip('.,;:!?)"\'') for u in urls]
    num_urls = len(urls)
    domains = []
    has_ip = 0
    is_short = 0
    for u in urls:
        u_parsable = u if u.startswith('http') else 'http://' + u
        try:
            parsed = urlparse(u_parsable)
            domain = parsed.netloc.lower()
        except:
            domain = ''
        if re.match(r'^\d+\.\d+\.\d+\.\d+(:\d+)?$', domain.split(':')[0]):
            has_ip = 1
        if any(sh in domain for sh in SHORTENERS):
            is_short = 1
        if domain:
            domains.append(domain)
    num_unique_domains = len(set(domains))
    avg_url_len = np.mean([len(u) for u in urls]) if urls else 0
    return num_urls, num_unique_domains, has_ip, is_short, avg_url_len

def extract_basic_features(text):
    char_count = len(text)
    words = re.findall(r'\w+', text)
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    num_digits = sum(ch.isdigit() for ch in text)
    num_at = text.count('@')
    num_exclaim = text.count('!')
    num_question = text.count('?')
    num_special = sum(1 for ch in text if ch in string.punctuation)
    num_uppercase_words = sum(1 for w in re.findall(r'\b\w+\b', text) if w.isupper())
    uppercase_ratio = (num_uppercase_words / word_count) if word_count else 0
    num_emails = len(re.findall(EMAIL_REGEX, text))
    return {
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_len': avg_word_len,
        'num_digits': num_digits,
        'num_at': num_at,
        'num_exclaim': num_exclaim,
        'num_question': num_question,
        'num_special': num_special,
        'uppercase_ratio': uppercase_ratio,
        'num_emails': num_emails
    }

SUSPICIOUS_WORDS = [
    'verify', 'account', 'password', 'login', 'bank', 'secure',
    'click', 'update', 'urgent', 'suspend', 'confirm', 'payment',
    'ssn', 'social', 'immediately', 'risk', 'limited time', 'winner'
]

def suspicious_counts(text):
    low = text.lower()
    counts = {f'susp_{w.replace(" ", "_")}': len(re.findall(r'\b' + re.escape(w) + r'\b', low)) for w in SUSPICIOUS_WORDS}
    counts['suspicious_total'] = sum(counts.values())
    return counts

# build features dataframe
feat_list = []
for t in df['raw_text'].fillna("").astype(str):
    u_info = extract_url_info(t)
    basic = extract_basic_features(t)
    susp = suspicious_counts(t)
    row = {
        'num_urls': u_info[0],
        'num_unique_domains': u_info[1],
        'has_ip_in_url': u_info[2],
        'is_short_url': u_info[3],
        'avg_url_len': u_info[4],
    }
    row.update(basic)
    row.update(susp)
    feat_list.append(row)

X_feats = pd.DataFrame(feat_list).fillna(0)

# ------------------ TRAIN TEST SPLIT ------------------
y = df['label']
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
train_df = df.loc[train_idx].reset_index(drop=True)
test_df = df.loc[test_idx].reset_index(drop=True)

y_train = train_df['label'].values
y_test = test_df['label'].values

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_tfidf_train = tfidf.fit_transform(train_df['final_text'])
X_tfidf_test = tfidf.transform(test_df['final_text'])

# numeric features
X_feats_train = X_feats.loc[train_idx].reset_index(drop=True)
X_feats_test = X_feats.loc[test_idx].reset_index(drop=True)

scaler = StandardScaler()
X_feats_train_scaled = scaler.fit_transform(X_feats_train)
X_feats_test_scaled = scaler.transform(X_feats_test)

X_num_train_sparse = csr_matrix(X_feats_train_scaled)
X_num_test_sparse = csr_matrix(X_feats_test_scaled)

X_train_final = hstack([X_tfidf_train, X_num_train_sparse])
X_test_final = hstack([X_tfidf_test, X_num_test_sparse])

print("Final shapes -> X_train:", X_train_final.shape, "X_test:", X_test_final.shape)

# ------------------ MODEL ------------------
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_final, y_train)
y_pred = log_reg.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------ SAVE MODEL & VECTORIZER ------------------
joblib.dump(log_reg, "phishing_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")  # also save the scaler for numeric features
print("✅ Model, vectorizer, scaler, and feature columns saved!")





# ------------------ STATISTICAL SUMMARY & OUTLIERS ------------------
print("\n--- Statistical Summary of Features ---\n")
print(X_feats.describe().T)

plt.figure(figsize=(15, 8))
sns.boxplot(data=X_feats)
plt.xticks(rotation=90)
plt.title("Boxplots of Numeric Features (Outlier Detection)")
plt.tight_layout()
plt.show()

# ------------------ CLI INTERFACE ------------------

# Stub functions to avoid NameError
def get_domain_age_days(domain): return 0
def check_blacklist(domain): return False
def fuzzy_domain_similarity(domain): return False

# =============== FEATURE EXTRACTION FUNCTIONS ===============
def extract_url_features(text):
    urls = re.findall(r'https?://\S+', text)
    domains = [urlparse(u).netloc for u in urls]
    num_urls = len(urls)
    unique_domains = len(set(domains))
    has_ip_in_url = any(re.search(r'\d+\.\d+\.\d+\.\d+', d) for d in domains)
    has_at_or_double_slash = any(re.search(r'@|//', u) for u in urls)
    avg_url_len = sum(len(u) for u in urls) / num_urls if num_urls else 0
    num_subdomains = sum(d.count('.') - 1 for d in domains)
    num_https = sum(u.startswith('https') for u in urls)
    num_http = sum(u.startswith('http:') for u in urls)
    # stub features
    domain_age_days = np.mean([get_domain_age_days(d) for d in domains]) if domains else 0
    blacklist_flag = int(any(check_blacklist(d) for d in domains))
    fuzzy_flag = int(any(fuzzy_domain_similarity(d) for d in domains))
    return {
        'num_urls': num_urls,
        'num_unique_domains': unique_domains,
        'has_ip_in_url': int(has_ip_in_url),
        'has_at_or_double_slash': int(has_at_or_double_slash),
        'avg_url_len': avg_url_len,
        'num_subdomains': num_subdomains,
        'num_https_urls': num_https,
        'num_http_urls': num_http,
        'domain_age_days': domain_age_days,
        'blacklist_flag': blacklist_flag,
        'fuzzy_domain_flag': fuzzy_flag
    }

def extract_sender_features(headers):
    from_addr = headers.get('From', '') if headers else ''
    reply_to = headers.get('Reply-To', '') if headers else ''
    from_domain = from_addr.split('@')[-1].lower() if '@' in from_addr else ''
    reply_domain = reply_to.split('@')[-1].lower() if '@' in reply_to else from_domain
    domain_mismatch = int(from_domain != reply_domain)
    display_name_mismatch = int('<' in from_addr and from_domain not in from_addr)
    return {
        'domain_mismatch': domain_mismatch,
        'display_name_mismatch': display_name_mismatch
    }

def extract_structural_features(html_body, attachments=[]):
    soup = BeautifulSoup(html_body, 'html.parser')
    num_images = len(soup.find_all('img'))
    num_scripts = len(soup.find_all('script'))
    num_forms = len(soup.find_all('form'))
    image_to_text_ratio = num_images / (len(soup.get_text().split()) + 1)
    num_attachments = len(attachments)
    attachment_types = [att.split('.')[-1].lower() for att in attachments]
    dangerous_attachment = int(any(t in attachment_types for t in [
        'exe', 'js', 'vbs', 'scr', 'bat', 'zip', 'rar', 'docm', 'xlsm'
    ]))
    return {
        'num_images': num_images,
        'num_scripts': num_scripts,
        'num_forms': num_forms,
        'image_to_text_ratio': image_to_text_ratio,
        'num_attachments': num_attachments,
        'dangerous_attachment': dangerous_attachment
    }

def extract_statistical_features(subject, body):
    text = (subject or '') + " " + (body or '')
    words = text.split()
    return {
        'email_len_chars': len(text),
        'email_len_words': len(words),
        'subject_len': len(subject or ''),
        'num_uppercase_words': sum(1 for w in words if w.isupper()),
        'num_exclamations': text.count('!'),
        'num_question_marks': text.count('?'),
        'link_to_text_ratio': (text.count('http') / (len(words) + 1))
    }

# =============== MAIN PREDICT FUNCTION ===============
def predict_mail(subject, body, model, tfidf, scaler=None, feature_columns=None):
    text = subject + " " + body
    clean_txt = clean_text_for_tfidf(text)
    tfidf_vec = tfidf.transform([clean_txt])

    u_info = extract_url_info(text)
    basic = extract_basic_features(text)
    susp = suspicious_counts(text)

    row = {
        'num_urls': u_info[0],
        'num_unique_domains': u_info[1],
        'has_ip_in_url': u_info[2],
        'is_short_url': u_info[3],
        'avg_url_len': u_info[4],
    }
    row.update(basic)
    row.update(susp)

    row_ordered = [row.get(col, 0) for col in feature_columns]
    extra_features_scaled = scaler.transform([row_ordered])
    final_features = hstack([tfidf_vec, csr_matrix(extra_features_scaled)])
    prediction = int(model.predict(final_features)[0])
    return prediction, row


feature_columns = X_feats_train.columns.tolist()  # from training

joblib.dump(feature_columns, "feature_columns.pkl") 

# ------------------ ADDITIONAL VISUALIZATIONS ------------------

# 1. Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution of Labels (Ham vs. Phishing)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham","Phishing"], yticklabels=["Ham","Phishing"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# 3. Correlation Heatmap of numeric features
plt.figure(figsize=(12, 8))
sns.heatmap(X_feats.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Engineered Features")
plt.show()

# 4. Feature Importance (Numeric Features only)
coefs = log_reg.coef_[0][-len(feature_columns):]  # last part = numeric feats
feat_imp = pd.Series(coefs, index=feature_columns).sort_values()

plt.figure(figsize=(10,6))
feat_imp.plot(kind="barh", color="purple")
plt.title("Feature Importance (Numeric Features)")
plt.show()

# 5. Top TF-IDF Words (Spam vs. Ham indicative)
def plot_top_words(vectorizer, model, n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0][:len(feature_names)]

    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]

    plt.figure(figsize=(12,6))
    plt.barh(feature_names[top_pos], coefs[top_pos], color="red", label="Spam-indicative")
    plt.barh(feature_names[top_neg], coefs[top_neg], color="green", label="Ham-indicative")
    plt.legend()
    plt.title("Top TF-IDF Features")
    plt.show()

plot_top_words(tfidf, log_reg, n=15)

# 6. Distribution plots of selected features
key_features = ["num_urls", "num_exclaim", "uppercase_ratio"]
for feat in key_features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(x=X_feats[feat], hue=y, fill=True, common_norm=False, alpha=0.4)
    plt.title(f"Distribution of {feat} by Label")
    plt.show()

# 7. PCA Visualization (sampling for performance)
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_train_final[:1000].toarray())  # limit for speed
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train[:1000], alpha=0.6, palette="Set1")
plt.title("PCA Projection of Emails")
plt.show()

# ------------------ CLI INTERFACE ------------------

if __name__ == "__main__":
    print("Paste the email subject and body (type 'exit' to quit):")
    while True:
        subject = input("\nSubject: ")
        if subject.lower() == 'exit':
            break
        body = input("Body: ")
        pred, feats = predict_mail(subject, body, log_reg, tfidf,
                                   scaler=scaler,
                                   feature_columns=feature_columns)
        if pred == 1:
            print("\n⚠️ This looks like a PHISHING mail!\n")
        else:
            print("\n✅ This looks SAFE.\n")





