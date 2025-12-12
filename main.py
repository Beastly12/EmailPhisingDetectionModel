import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle
import os
import re
import hashlib
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                             SpectralClustering, Birch, OPTICS)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except:
    HAS_XGB = False

SEED = 42
np.random.seed(SEED)


class FeatureExtractor:
    """Extract features from email data"""

    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        text = str(text).lower()
        text = re.sub(r"http\S+", " link ", text)
        text = re.sub(r"\S+@\S+", " email ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_from_dataframe(df):
        """Extract features from entire dataframe"""
        df = df.copy()

        # Handle column names
        if 'Email_Subject' in df.columns:
            df['subject'] = df['Email_Subject']
        if 'Email_Content' in df.columns:
            df['body'] = df['Email_Content']

        df['subject'] = df.get('subject', '').fillna('').astype(str)
        df['body'] = df.get('body', '').fillna('').astype(str)
        df['clean_content'] = df['body'].apply(FeatureExtractor.clean_text)

        # Extract features
        df['subject_length'] = df['subject'].apply(len)
        df['subject_num_special'] = df['subject'].str.count(r'[^a-zA-Z0-9 ]')
        df['subject_upper_words'] = df['subject'].apply(
            lambda x: sum(1 for w in x.split() if w.isupper())
        )

        df['body_length'] = df['body'].apply(len)
        df['body_word_count'] = df['clean_content'].apply(lambda x: len(x.split()))
        df['body_num_digits'] = df['body'].str.count(r'\d')
        df['body_num_exclaim'] = df['body'].str.count('!')
        df['body_num_question'] = df['body'].str.count(r'\?')

        df['num_urls'] = df['body'].apply(lambda x: len(re.findall(r'http\S+', str(x))))
        df['has_url'] = (df['num_urls'] > 0).astype(int)

        urgency_keywords = ['urgent', 'immediate', 'action required', 'verify',
                            'suspended', 'expired', 'confirm', 'alert']
        df['urgency_score'] = df['subject'].str.lower().apply(
            lambda x: sum(kw in x for kw in urgency_keywords)
        )

        financial_keywords = ['bank', 'account', 'payment', 'credit', 'paypal',
                              'transaction', 'invoice', 'refund']
        df['financial_score'] = df['body'].str.lower().apply(
            lambda x: sum(kw in x for kw in financial_keywords)
        )

        return df

    @staticmethod
    def extract_from_single_email(email_data):
        """Extract features from single email"""
        features = {}

        subject = str(email_data.get('subject', ''))
        body = str(email_data.get('body', ''))
        timestamp = email_data.get('timestamp', datetime.now())

        # Basic features
        features['subject_length'] = len(subject)
        features['subject_word_count'] = len(subject.split())
        features['subject_num_special'] = len(re.findall(r'[^a-zA-Z0-9\s]', subject))
        features['subject_upper_words'] = sum(1 for w in subject.split() if w.isupper())

        features['body_length'] = len(body)
        features['body_word_count'] = len(body.split())
        features['body_num_digits'] = len(re.findall(r'\d', body))
        features['body_num_exclaim'] = body.count('!')
        features['body_num_question'] = body.count('?')
        features['body_num_caps'] = sum(1 for c in body if c.isupper())

        # URL features
        urls = re.findall(r'http[s]?://\S+', body)
        features['num_urls'] = len(urls)
        features['has_url'] = 1 if urls else 0
        features['has_ip_url'] = 1 if any(re.search(r'\d+\.\d+\.\d+\.\d+', url)
                                          for url in urls) else 0
        features['has_url_shortener'] = 1 if any(short in url.lower() for url in urls
                                                 for short in ['bit.ly', 'tinyurl', 'goo.gl']) else 0

        # Keyword features
        urgency_keywords = ['urgent', 'immediate', 'action required', 'verify',
                            'suspended', 'expired', 'confirm', 'alert', 'attention',
                            'security', 'warning']
        features['urgency_score'] = sum(kw in subject.lower() + ' ' + body.lower()
                                        for kw in urgency_keywords)

        financial_keywords = ['bank', 'account', 'payment', 'credit', 'paypal',
                              'transaction', 'invoice', 'refund', 'wire', 'transfer']
        features['financial_score'] = sum(kw in subject.lower() + ' ' + body.lower()
                                          for kw in financial_keywords)

        personal_keywords = ['password', 'social security', 'ssn', 'pin', 'cvv']
        features['personal_info_score'] = sum(kw in subject.lower() + ' ' + body.lower()
                                              for kw in personal_keywords)

        # Time features
        if isinstance(timestamp, datetime):
            features['hour_of_day'] = timestamp.hour
            features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            features['is_night'] = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
        else:
            features['hour_of_day'] = 12
            features['is_weekend'] = 0
            features['is_night'] = 0

        features['has_attachment'] = email_data.get('has_attachment', 0)
        features['num_attachments'] = email_data.get('num_attachments', 0)

        return features


class ThreatAnalyzer:
    """Analyze and classify threats"""

    @staticmethod
    def get_threat_level(probability):
        """Determine threat level from probability"""
        if probability >= 0.9:
            return "CRITICAL", "threat-critical"
        elif probability >= 0.75:
            return "HIGH", "threat-high"
        elif probability >= 0.5:
            return "MEDIUM", "threat-medium"
        elif probability >= 0.25:
            return "LOW", "threat-low"
        else:
            return "SAFE", "threat-safe"

    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = None
        return metrics


class PrivacyProtection:
    """Handle privacy-preserving techniques"""

    @staticmethod
    def add_laplace_noise(df, cols, epsilon=1.0):
        """Add Laplace noise for differential privacy"""
        df = df.copy()
        scale = 1.0 / max(epsilon, 0.01)
        for c in cols:
            if c in df.columns:
                noise = np.random.laplace(0, scale, len(df))
                df[c] = df[c] + noise
        return df

    @staticmethod
    def hash_email(email):
        """Hash email address"""
        if pd.isna(email) or email == '':
            return 'unknown'
        return hashlib.sha256(str(email).encode()).hexdigest()[:16]


class ModelManager:
    """Manage model loading and training"""

    @staticmethod
    @st.cache_resource
    def load_pretrained_models():
        """Load pre-trained models from disk"""
        models = {}
        model_dir = 'outputs/ml_models'

        if not os.path.exists(model_dir):
            return {}, None, None

        model_patterns = {
            'XGBoost': ['best_model_xgboost.pkl', 'xgboost_model.pkl'],
            'Random Forest': ['best_model_random_forest.pkl', 'rf_model.pkl'],
            'Logistic Regression': ['best_model_logistic_regression.pkl', 'lr_model.pkl'],
            'SVM': ['best_model_svm.pkl', 'best_model_svm_rbf.pkl', 'svm_model.pkl']
        }

        for model_name, patterns in model_patterns.items():
            for pattern in patterns:
                path = os.path.join(model_dir, pattern)
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                        break
                    except:
                        continue

        scaler = None
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except:
                scaler = StandardScaler()

        metrics_df = None
        metrics_path = os.path.join(model_dir, 'model_comparison.csv')
        if os.path.exists(metrics_path):
            try:
                metrics_df = pd.read_csv(metrics_path)
            except:
                pass

        return models, scaler, metrics_df

    @staticmethod
    def train_models(X_train, y_train, X_test, y_test, model_names):
        """Train selected models"""
        results = {}
        trained_models = {}

        for model_name in model_names:
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=SEED)
            elif model_name == "SVM":
                model = SVC(kernel='rbf', probability=True, random_state=SEED)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            elif model_name == "XGBoost" and HAS_XGB:
                model = XGBClassifier(random_state=SEED, eval_metric='logloss')
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            results[model_name] = ThreatAnalyzer.compute_metrics(y_test, y_pred, y_prob)
            trained_models[model_name] = model

        return trained_models, results


class UIComponents:
    """UI rendering components"""

    @staticmethod
    def render_css():
        """Render custom CSS"""
        st.markdown("""
        <style>
            .main {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .custom-card {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .metric-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin: 10px;
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 0.9em;
                opacity: 0.9;
            }
            .threat-critical {
                background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #cc0000;
                margin: 20px 0;
                box-shadow: 0 6px 12px rgba(238,9,121,0.4);
                animation: pulse 2s infinite;
            }
            .threat-high {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #ff4444;
                margin: 20px 0;
                box-shadow: 0 6px 12px rgba(245,87,108,0.4);
            }
            .threat-medium {
                background: linear-gradient(135deg, #ffa751 0%, #ffe259 100%);
                color: #333;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #ff8800;
                margin: 20px 0;
                box-shadow: 0 6px 12px rgba(255,167,81,0.4);
            }
            .threat-low {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #00aa88;
                margin: 20px 0;
            }
            .threat-safe {
                background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #228B22;
                margin: 20px 0;
                box-shadow: 0 6px 12px rgba(86,171,47,0.4);
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: rgba(255,255,255,0.1);
                border-radius: 10px 10px 0 0;
                padding: 10px 20px;
            }
            .stTabs [aria-selected="true"] {
                background-color: white;
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_metric_box(label, value):
        """Render a metric box"""
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_threat_alert(threat_level, threat_class, probability):
        """Render threat alert box"""
        alerts = {
            "CRITICAL": ("⛔ CRITICAL THREAT", "Extremely high risk. Delete immediately."),
            "HIGH": ("⚠️ HIGH RISK", "Strong phishing indicators detected."),
            "MEDIUM": ("⚠ SUSPICIOUS", "Multiple suspicious indicators."),
            "LOW": ("ℹ️ LOW RISK", "Minor concerns detected."),
            "SAFE": ("✅ SAFE", "No significant threats.")
        }

        title, message = alerts.get(threat_level, ("", ""))

        st.markdown(f"""
        <div class="{threat_class}">
            <h2>{title}</h2>
            <h3>Probability: {probability * 100:.1f}%</h3>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)


class PhishingDetectionApp:
    """Main application class"""

    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.load_models()
        UIComponents.render_css()

    def setup_page(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="Phishing Shield Pro",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'scan_history' not in st.session_state:
            st.session_state.scan_history = []
        if 'threat_log' not in st.session_state:
            st.session_state.threat_log = []
        if 'total_scans' not in st.session_state:
            st.session_state.total_scans = 0
        if 'threats_detected' not in st.session_state:
            st.session_state.threats_detected = 0
        if 'trained_models_session' not in st.session_state:
            st.session_state.trained_models_session = {}

    def load_models(self):
        """Load pre-trained models"""
        self.pretrained_models, self.pretrained_scaler, self.metrics_df = \
            ModelManager.load_pretrained_models()

        # Try to load feature names from pretrained models
        self.pretrained_feature_names = None
        model_dir = 'outputs/ml_models'
        feature_path = os.path.join(model_dir, 'feature_names.pkl')
        if os.path.exists(feature_path):
            try:
                with open(feature_path, 'rb') as f:
                    self.pretrained_feature_names = pickle.load(f)
            except:
                pass

    def get_available_models(self):
        """Get all available models"""
        models = {}
        if self.pretrained_models:
            models.update(self.pretrained_models)
        if st.session_state.trained_models_session:
            models.update(st.session_state.trained_models_session)
        return models

    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/security-shield-green.png",
                     width=80)
            st.title("Phishing Shield Pro")
            st.markdown("---")

            st.subheader("🔧 System Status")

            if self.pretrained_models:
                st.success(f"✓ {len(self.pretrained_models)} pre-trained models")
                for model_name in self.pretrained_models.keys():
                    st.markdown(f"  - {model_name}")
            else:
                st.warning("⚠ No pre-trained models")

            if st.session_state.trained_models_session:
                st.info(f"✓ {len(st.session_state.trained_models_session)} session models")

            st.markdown("---")

            st.subheader("📊 Statistics")
            st.metric("Total Scans", st.session_state.total_scans)
            st.metric("Threats Blocked", st.session_state.threats_detected)

            if st.session_state.total_scans > 0:
                threat_rate = (st.session_state.threats_detected /
                               st.session_state.total_scans) * 100
                st.metric("Threat Rate", f"{threat_rate:.1f}%")

            st.markdown("---")

            st.subheader("⚙️ Settings")
            self.detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
            self.show_feature_analysis = st.checkbox("Show Feature Analysis", value=True)

            if st.button("🔄 Reload Models"):
                st.cache_resource.clear()
                st.rerun()

            if st.button("🗑️ Clear History"):
                st.session_state.scan_history = []
                st.session_state.threat_log = []
                st.rerun()

    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>🛡️ Advanced Phishing Detection System</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>
                Complete Pipeline: Data Upload → Feature Engineering → Clustering → ML Training → Detection
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_tab_data_upload(self):
        """Render data upload tab"""
        st.markdown("### 📁 Data Upload and Feature Engineering")
        st.markdown("Upload your email dataset to extract features and prepare for training")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
            use_sample = st.checkbox("Use sample dataset (outputs/Engineered_Features.csv)")

        with col2:
            st.markdown("**Required Columns:**")
            st.markdown("- `subject` or `Email_Subject`")
            st.markdown("- `body` or `Email_Content`")
            st.markdown("- `label` or `Label`")

        if uploaded_file or use_sample:
            # Load data
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Uploaded file loaded: {len(df)} rows")
            elif use_sample and os.path.exists('outputs/Engineered_Features.csv'):
                df = pd.read_csv('outputs/Engineered_Features.csv')
                st.success(f"✓ Sample data loaded: {len(df)} rows")
            else:
                st.error("Sample file not found")
                return

            # Preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10))

            # Check label
            if 'label' not in df.columns and 'Label' in df.columns:
                df['label'] = df['Label']

            if 'label' not in df.columns:
                st.error("Dataset must have 'label' or 'Label' column")
                return

            # Feature engineering
            st.markdown("---")
            st.markdown("### 🔧 Feature Engineering")

            if st.button("🚀 Extract Features", type="primary"):
                with st.spinner("Extracting features..."):
                    df = FeatureExtractor.extract_from_dataframe(df)
                    st.session_state['uploaded_data'] = df
                    st.success("✓ Features extracted successfully!")

            if 'uploaded_data' in st.session_state:
                df = st.session_state['uploaded_data']

                feature_cols = [
                    'subject_length', 'subject_num_special', 'subject_upper_words',
                    'body_length', 'body_word_count', 'body_num_digits',
                    'body_num_exclaim', 'body_num_question', 'num_urls', 'has_url',
                    'urgency_score', 'financial_score'
                ]

                st.success(f"✓ Extracted {len(feature_cols)} features")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Emails", len(df))

                label_counts = df['label'].value_counts()
                col2.metric("Phishing", label_counts.iloc[0] if len(label_counts) > 0 else 0)
                col3.metric("Legitimate", label_counts.iloc[1] if len(label_counts) > 1 else 0)

                # ==========================================
                # NEW EDA SECTION
                # ==========================================
                st.markdown("---")
                st.markdown("### 📊 Exploratory Data Analysis (EDA)")

                eda_tabs = st.tabs(["Correlation Heatmap", "Feature Distribution", "Scatter Plots", "Class Balance"])

                with eda_tabs[0]:
                    st.subheader("Feature Correlations")
                    # Calculate correlation
                    corr_cols = feature_cols + ['label']
                    # Ensure label is numeric
                    temp_df = df.copy()
                    if temp_df['label'].dtype == 'object':
                        le = LabelEncoder()
                        temp_df['label'] = le.fit_transform(temp_df['label'])

                    corr_matrix = temp_df[corr_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix of Features"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                with eda_tabs[1]:
                    st.subheader("Feature Distributions by Label")
                    selected_dist_feature = st.selectbox("Select Feature to Visualize", feature_cols)

                    fig_hist = px.histogram(
                        df,
                        x=selected_dist_feature,
                        color='label',
                        barmode='overlay',
                        title=f"Distribution of {selected_dist_feature}",
                        opacity=0.7,
                        color_discrete_sequence=['#56ab2f', '#ee0979']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    col_box1, col_box2 = st.columns(2)
                    with col_box1:
                        fig_box = px.box(
                            df,
                            x='label',
                            y=selected_dist_feature,
                            color='label',
                            title=f"Box Plot of {selected_dist_feature}",
                            color_discrete_sequence=['#56ab2f', '#ee0979']
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

                with eda_tabs[2]:
                    st.subheader("Scatter Plot Analysis")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        x_axis = st.selectbox("X Axis", feature_cols, index=0)
                    with col_s2:
                        y_axis = st.selectbox("Y Axis", feature_cols, index=3)

                    fig_scatter = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        color='label',
                        title=f"{x_axis} vs {y_axis}",
                        opacity=0.6,
                        color_discrete_sequence=['#56ab2f', '#ee0979']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with eda_tabs[3]:
                    st.subheader("Class Distribution")
                    fig_pie = px.pie(
                        df,
                        names='label',
                        title='Phishing vs Legitimate Distribution',
                        color_discrete_sequence=['#ee0979', '#56ab2f']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ==========================================
                # END EDA SECTION
                # ==========================================

                # Privacy
                st.markdown("---")
                st.markdown("### 🔒 Privacy Protection")

                privacy_mode = st.selectbox("Apply privacy technique",
                                            ["None", "Differential Privacy (Laplace Noise)"])

                if privacy_mode == "Differential Privacy (Laplace Noise)":
                    epsilon = st.slider("Epsilon (ε) - Privacy Budget", 0.1, 5.0, 1.0)

                    if st.button("Apply Privacy"):
                        with st.spinner("Adding noise..."):
                            df[feature_cols] = PrivacyProtection.add_laplace_noise(
                                df[feature_cols], feature_cols, epsilon
                            )
                            st.session_state['uploaded_data'] = df
                            st.success(f"✓ Applied DP with ε={epsilon}")

                # Clustering
                st.markdown("---")
                st.markdown("### 🎨 Clustering Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    cluster_algo = st.selectbox("Algorithm",
                                                ["KMeans", "DBSCAN", "Agglomerative", "Spectral", "Birch", "OPTICS"])

                with col2:
                    if cluster_algo in ["KMeans", "Agglomerative", "Spectral", "Birch"]:
                        n_clusters = st.slider("Number of clusters", 2, 10, 5)

                if st.button("Run Clustering"):
                    with st.spinner(f"Running {cluster_algo}..."):
                        X = df[feature_cols].fillna(0).values
                        scaler_cluster = StandardScaler()
                        X_scaled = scaler_cluster.fit_transform(X)

                        # Select algorithm
                        if cluster_algo == "KMeans":
                            model = KMeans(n_clusters=n_clusters, random_state=SEED)
                        elif cluster_algo == "DBSCAN":
                            model = DBSCAN(eps=0.5, min_samples=5)
                        elif cluster_algo == "Agglomerative":
                            model = AgglomerativeClustering(n_clusters=n_clusters)
                        elif cluster_algo == "Spectral":
                            model = SpectralClustering(n_clusters=n_clusters, random_state=SEED)
                        elif cluster_algo == "Birch":
                            model = Birch(n_clusters=n_clusters)
                        else:
                            model = OPTICS(min_samples=5)

                        labels = model.fit_predict(X_scaled)

                        # Visualization
                        pca = PCA(n_components=2, random_state=SEED)
                        X_2d = pca.fit_transform(X_scaled)

                        plot_df = pd.DataFrame({
                            'PC1': X_2d[:, 0],
                            'PC2': X_2d[:, 1],
                            'Cluster': labels.astype(str),
                            'Label': df['label'].astype(str)
                        })

                        fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                         symbol='Label', title=f"{cluster_algo} Results")
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("Cluster sizes:")
                        st.write(pd.Series(labels).value_counts().sort_index())

    def render_tab_live_scanner(self):
        """Render live email scanner tab"""
        st.markdown("### 🔍 Live Email Scanner")
        st.markdown("Analyze individual emails in real-time")

        available_models = self.get_available_models()

        if not available_models:
            st.warning("⚠️ No models available. Train models in Tab 3 or ensure pre-trained models exist.")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)

            # WRAPPED IN FORM TO PREVENT DATE/TIME RESET
            with st.form(key='email_scanner_form'):
                email_subject = st.text_input("📨 Email Subject", placeholder="Subject line...")

                # REMOVED RECEIVER EMAIL, KEPT SENDER ONLY
                sender_email = st.text_input("👤 Sender", placeholder="sender@example.com")

                email_body = st.text_area("📝 Email Body", height=200, placeholder="Email content...")

                st.markdown("#### 📎 Additional Details")

                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    has_attachments = st.checkbox("Has Attachments")
                    # Conditional logic inside form is okay
                    num_attachments = st.number_input("Number", 0, 10, 0) if has_attachments else 0
                with col_y:
                    # REMOVED DYNAMIC DEFAULT VALUE to prevent resetting
                    email_time = st.time_input("Time Received", value=datetime.now().time())
                with col_z:
                    email_date = st.date_input("Date Received", value=datetime.now().date())

                st.markdown('</div>', unsafe_allow_html=True)

                # Scan button IS NOW A SUBMIT BUTTON
                scan_button = st.form_submit_button("🔍 Analyze Email", type="primary", use_container_width=True)

        with col2:
            st.markdown("### 🤖 Model Selection")

            model_choice = st.selectbox("Choose Model", list(available_models.keys()))

            # Show model info
            if model_choice in st.session_state.trained_models_session:
                st.info(f"✓ Session Model")
                if 'selected_features' in st.session_state:
                    with st.expander("Features used"):
                        st.write(st.session_state['selected_features'])
            else:
                st.info(f"✓ Pre-trained Model")
                if self.pretrained_feature_names:
                    with st.expander("Features used"):
                        st.write(self.pretrained_feature_names)

            st.markdown("### 📊 Recent Scans")
            if st.session_state.scan_history:
                for scan in st.session_state.scan_history[-3:]:
                    threat_emoji = "🔴" if scan['is_phishing'] else "🟢"
                    st.markdown(f"{threat_emoji} **{scan['probability']:.1%}** - {scan['subject'][:30]}...")
            else:
                st.info("No scans yet")

        # Process scan (Only triggers on Form Submit)
        if scan_button:
            if not email_subject and not email_body:
                st.error("❌ Please enter at least subject or body text")
                return

            with st.spinner("🔍 Analyzing email..."):
                # Create timestamp
                timestamp = datetime.combine(email_date, email_time)

                # Prepare email data
                email_data = {
                    'subject': email_subject,
                    'body': email_body,
                    'timestamp': timestamp,
                    'has_attachment': 1 if has_attachments else 0,
                    'num_attachments': num_attachments
                }

                # Extract features
                features = FeatureExtractor.extract_from_single_email(email_data)

                # Determine which features to use
                if model_choice in st.session_state.trained_models_session:
                    # Use session model - get features from training
                    if 'selected_features' in st.session_state:
                        feature_list = st.session_state['selected_features']
                    else:
                        # Default features for session models
                        feature_list = [
                            'subject_length', 'subject_num_special', 'subject_upper_words',
                            'body_length', 'body_word_count', 'body_num_digits',
                            'body_num_exclaim', 'body_num_question', 'num_urls', 'has_url',
                            'urgency_score', 'financial_score'
                        ]
                else:
                    # Use pretrained model - try to get feature names
                    if self.pretrained_feature_names:
                        feature_list = self.pretrained_feature_names
                    else:
                        # Default to common features (adjust based on your pretrained model)
                        feature_list = [
                            'subject_length', 'subject_num_special', 'subject_upper_words',
                            'body_length', 'body_word_count', 'body_num_digits',
                            'body_num_exclaim', 'body_num_question'
                        ]

                # Create feature vector with only the needed features
                X = np.array([[features.get(f, 0) for f in feature_list]])

                # Scale if scaler exists
                if model_choice in st.session_state.trained_models_session:
                    # Use session scaler
                    if 'trained_scaler' in st.session_state:
                        X = st.session_state.trained_scaler.transform(X)
                else:
                    # Use pretrained scaler
                    if self.pretrained_scaler is not None:
                        try:
                            X = self.pretrained_scaler.transform(X)
                        except Exception as e:
                            st.warning(f"Could not apply scaling: {e}")

                # Predict
                model = available_models[model_choice]
                prediction = model.predict(X)[0]

                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(X)[0][1]
                else:
                    probability = float(prediction)

                is_phishing = probability >= self.detection_threshold

                # Update stats
                st.session_state.total_scans += 1
                if is_phishing:
                    st.session_state.threats_detected += 1

                # Log scan
                scan_record = {
                    'timestamp': datetime.now(),
                    'subject': email_subject,
                    'sender': sender_email,
                    'probability': probability,
                    'is_phishing': is_phishing,
                    'model': model_choice
                }
                st.session_state.scan_history.append(scan_record)

                if is_phishing:
                    st.session_state.threat_log.append(scan_record)

            # STORE RESULTS IN SESSION STATE TO SURVIVE RERUNS
            st.session_state['last_scan_result'] = {
                'probability': probability,
                'is_phishing': is_phishing,
                'model_choice': model_choice,
                'features': features
            }

        # Check if we have a result to display (either from just now or previous run)
        if 'last_scan_result' in st.session_state:
            res = st.session_state['last_scan_result']
            probability = res['probability']
            is_phishing = res['is_phishing']
            model_choice = res['model_choice']
            features = res['features']

            # Display results
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            threat_level, threat_class = ThreatAnalyzer.get_threat_level(probability)
            UIComponents.render_threat_alert(threat_level, threat_class, probability)

            # Detailed metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                UIComponents.render_metric_box("Model", model_choice)
            with col2:
                UIComponents.render_metric_box("Threat Score", f"{probability * 100:.1f}%")
            with col3:
                UIComponents.render_metric_box("Classification", "PHISHING" if is_phishing else "LEGITIMATE")
            with col4:
                UIComponents.render_metric_box("Confidence", threat_level)

            # Feature analysis
            if self.show_feature_analysis:
                st.markdown("---")
                st.markdown("### 🔬 Feature Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📝 Content Features")
                    feature_df = pd.DataFrame({
                        'Feature': ['Subject Length', 'Body Length', 'Word Count', 'URLs Found'],
                        'Value': [
                            features['subject_length'],
                            features['body_length'],
                            features['body_word_count'],
                            features['num_urls']
                        ]
                    })
                    st.dataframe(feature_df, use_container_width=True)

                with col2:
                    st.markdown("#### 🚨 Risk Indicators")
                    risk_df = pd.DataFrame({
                        'Indicator': ['Urgency Score', 'Financial Keywords', 'Personal Info', 'Suspicious URLs'],
                        'Score': [
                            features['urgency_score'],
                            features['financial_score'],
                            features['personal_info_score'],
                            features['has_ip_url'] + features['has_url_shortener']
                        ]
                    })

                    fig = px.bar(risk_df, x='Indicator', y='Score',
                                 title='Risk Indicator Scores',
                                 color='Score',
                                 color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)

                # Top features
                st.markdown("#### 🎯 Key Detection Factors")

                important_features = {
                    'Urgency Keywords': features['urgency_score'],
                    'Financial Terms': features['financial_score'],
                    'URL Count': features['num_urls'],
                    'Special Characters': features['subject_num_special'],
                    'Exclamation Marks': features['body_num_exclaim']
                }

                sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)

                for feat_name, feat_val in sorted_features[:5]:
                    if feat_val > 0:
                        st.warning(f"🔸 **{feat_name}**: {feat_val}")

    def render_tab_model_training(self):
        """Render model training tab"""
        st.markdown("### 🤖 Machine Learning Model Training")
        st.markdown("Train new models on your uploaded dataset")

        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ Please upload and extract features from data in Tab 1 first!")
            return

        df = st.session_state['uploaded_data']

        st.success(f"✓ Dataset loaded: {len(df)} samples")

        # Feature selection
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection")

        all_features = [
            'subject_length', 'subject_num_special', 'subject_upper_words',
            'body_length', 'body_word_count', 'body_num_digits',
            'body_num_exclaim', 'body_num_question', 'num_urls', 'has_url',
            'urgency_score', 'financial_score'
        ]

        available_features = [f for f in all_features if f in df.columns]

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_features = st.multiselect(
                "Select features for training",
                available_features,
                default=available_features[:8]
            )

        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if len(selected_features) < 2:
            st.error("❌ Please select at least 2 features")
            return

        # Model selection
        st.markdown("---")
        st.markdown("### 🎓 Model Selection")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            train_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            train_rf = st.checkbox("Random Forest", value=True)
        with col3:
            train_svm = st.checkbox("SVM")
        with col4:
            train_xgb = st.checkbox("XGBoost", value=HAS_XGB, disabled=not HAS_XGB)

        models_to_train = []
        if train_lr:
            models_to_train.append("Logistic Regression")
        if train_rf:
            models_to_train.append("Random Forest")
        if train_svm:
            models_to_train.append("SVM")
        if train_xgb and HAS_XGB:
            models_to_train.append("XGBoost")

        if not models_to_train:
            st.warning("⚠️ Please select at least one model")
            return

        # Train button
        st.markdown("---")

        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                # Prepare data
                X = df[selected_features].fillna(0).values
                y = df['label'].values

                # Encode labels if needed
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=SEED, stratify=y
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train models
                trained_models, results = ModelManager.train_models(
                    X_train_scaled, y_train, X_test_scaled, y_test, models_to_train
                )

                # Store models and scaler
                st.session_state.trained_models_session.update(trained_models)
                st.session_state.trained_scaler = scaler
                st.session_state.training_results = results
                st.session_state.selected_features = selected_features  # Store for later use

                st.success("✓ Training complete!")

        # Display results
        if 'training_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Training Results")

            results = st.session_state.training_results

            # Metrics table
            metrics_df = pd.DataFrame(results).T
            metrics_df = metrics_df.round(4)

            st.dataframe(metrics_df, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Accuracy comparison
                fig = px.bar(
                    x=list(results.keys()),
                    y=[results[m]['accuracy'] for m in results.keys()],
                    title="Model Accuracy Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'},
                    color=[results[m]['accuracy'] for m in results.keys()],
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # F1 Score comparison
                fig = px.bar(
                    x=list(results.keys()),
                    y=[results[m]['f1'] for m in results.keys()],
                    title="Model F1 Score Comparison",
                    labels={'x': 'Model', 'y': 'F1 Score'},
                    color=[results[m]['f1'] for m in results.keys()],
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Best model
            best_model = max(results.items(), key=lambda x: x[1]['f1'])

            st.markdown(f"""
            <div class="custom-card">
                <h3>🏆 Best Model: {best_model[0]}</h3>
                <p><strong>F1 Score:</strong> {best_model[1]['f1']:.4f}</p>
                <p><strong>Accuracy:</strong> {best_model[1]['accuracy']:.4f}</p>
                <p><strong>Precision:</strong> {best_model[1]['precision']:.4f}</p>
                <p><strong>Recall:</strong> {best_model[1]['recall']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    def render_tab_history(self):
        """Render scan history tab"""
        st.markdown("### 📜 Scan History & Analytics")

        if not st.session_state.scan_history:
            st.info("No scans performed yet. Use the Live Scanner tab to analyze emails.")
            return

        # Overview metrics
        st.markdown("### 📊 Overview")

        col1, col2, col3, col4 = st.columns(4)

        total = st.session_state.total_scans
        threats = st.session_state.threats_detected
        safe = total - threats
        threat_rate = (threats / total * 100) if total > 0 else 0

        col1.metric("Total Scans", total)
        col2.metric("Threats Detected", threats, delta=f"{threat_rate:.1f}%")
        col3.metric("Safe Emails", safe)
        col4.metric("Detection Rate", f"{threat_rate:.1f}%")

        # History table
        st.markdown("---")
        st.markdown("### 📋 Scan Details")

        history_df = pd.DataFrame(st.session_state.scan_history)

        if len(history_df) > 0:
            display_df = history_df[['timestamp', 'subject', 'sender', 'probability', 'is_phishing', 'model']].copy()
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x * 100:.1f}%")
            display_df['is_phishing'] = display_df['is_phishing'].apply(lambda x: "🔴 Phishing" if x else "🟢 Safe")
            display_df.columns = ['Timestamp', 'Subject', 'Sender', 'Threat Score', 'Status', 'Model']

            st.dataframe(display_df, use_container_width=True)

            # Visualization
            st.markdown("---")
            st.markdown("### 📈 Threat Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Threat distribution
                threat_counts = history_df['is_phishing'].value_counts()
                fig = px.pie(
                    values=threat_counts.values,
                    names=['Safe', 'Phishing'] if False in threat_counts.index else ['Phishing'],
                    title='Email Classification Distribution',
                    color_discrete_sequence=['#56ab2f', '#ee0979']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Threat score distribution
                fig = px.histogram(
                    history_df,
                    x='probability',
                    nbins=20,
                    title='Threat Score Distribution',
                    labels={'probability': 'Threat Probability'},
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Timeline
            if len(history_df) > 1:
                st.markdown("---")
                st.markdown("### ⏱️ Scan Timeline")

                timeline_df = history_df.copy()
                timeline_df['hour'] = pd.to_datetime(timeline_df['timestamp']).dt.hour

                hourly_counts = timeline_df.groupby(['hour', 'is_phishing']).size().reset_index(name='count')
                hourly_counts['type'] = hourly_counts['is_phishing'].apply(lambda x: 'Phishing' if x else 'Safe')

                fig = px.bar(
                    hourly_counts,
                    x='hour',
                    y='count',
                    color='type',
                    title='Scans by Hour',
                    labels={'hour': 'Hour of Day', 'count': 'Number of Scans'},
                    color_discrete_map={'Safe': '#56ab2f', 'Phishing': '#ee0979'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Export option
            st.markdown("---")
            st.markdown("### 💾 Export Data")

            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"phishing_scan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    def run(self):
        """Main application loop"""
        self.render_sidebar()
        self.render_header()

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Data Upload & Features",
            "🔍 Live Email Scanner",
            "🤖 Model Training",
            "📜 History & Analytics"
        ])

        with tab1:
            self.render_tab_data_upload()

        with tab2:
            self.render_tab_live_scanner()

        with tab3:
            self.render_tab_model_training()

        with tab4:
            self.render_tab_history()


if __name__ == "__main__":
    app = PhishingDetectionApp()
    app.run()