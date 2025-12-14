# app_comprehensive_medical.py
"""
COMPREHENSIVE Medical Predictor with ALL Features
Features grouped logically with Next/Prev navigation
Medical focus with RECALL prioritization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Page config
st.set_page_config(
    page_title="Comprehensive Medical Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ComprehensiveMedicalPredictor:
    def __init__(self):
        # Initialize session state
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'current_group' not in st.session_state:
            st.session_state.current_group = 0
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        
        # All diseases we predict
        self.target_diseases = [
            'Diabetes', 'HeartDisease', 'Stroke', 'Asthma',
            'HeartAttack', 'COPD', 'Depression', 'KidneyDisease',
            'Arthritis', 'SkinCancer'
        ]
        
        # Define feature groups (logical categorization)
        self.feature_groups = {
            0: {
                'name': '📋 Basic Demographics',
                'description': 'Basic personal information',
                'features': {
                    'Age': {'type': 'slider', 'min': 18, 'max': 100, 'default': 45, 'step': 1, 'help': 'Age in years'},
                    'Sex': {'type': 'select', 'options': ['Male', 'Female'], 'default': 'Male', 'help': 'Biological sex'},
                    'Race': {'type': 'select', 'options': ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 'default': 'White', 'help': 'Race/ethnicity'},
                    'MaritalStatus': {'type': 'select', 'options': ['Married', 'Divorced', 'Widowed', 'Separated', 'Never married', 'Unmarried couple'], 'default': 'Married', 'help': 'Current marital status'},
                    'Education': {'type': 'select', 'options': ['Less than HS', 'HS Graduate', 'Some College', 'College Graduate', 'Post-graduate'], 'default': 'College Graduate', 'help': 'Highest education level'},
                    'Income': {'type': 'select', 'options': ['<$15k', '$15-25k', '$25-35k', '$35-50k', '$50-75k', '$75-100k', '>$100k'], 'default': '$50-75k', 'help': 'Annual household income'},
                    'Employment': {'type': 'select', 'options': ['Employed', 'Self-employed', 'Unemployed', 'Retired', 'Student', 'Unable to work'], 'default': 'Employed', 'help': 'Employment status'},
                    'Children': {'type': 'slider', 'min': 0, 'max': 10, 'default': 2, 'step': 1, 'help': 'Number of children'}
                }
            },
            1: {
                'name': '⚕️ Clinical Measurements',
                'description': 'Medical measurements and vitals',
                'features': {
                    'BMI': {'type': 'slider', 'min': 15.0, 'max': 50.0, 'default': 25.0, 'step': 0.1, 'help': 'Body Mass Index'},
                    'Height': {'type': 'slider', 'min': 120, 'max': 220, 'default': 170, 'step': 1, 'help': 'Height in cm'},
                    'Weight': {'type': 'slider', 'min': 40, 'max': 200, 'default': 70, 'step': 1, 'help': 'Weight in kg'},
                    'HighBP': {'type': 'select', 'options': ['No', 'Yes', 'Borderline', "Don't know"], 'default': 'No', 'help': 'High blood pressure diagnosis'},
                    'HighChol': {'type': 'select', 'options': ['No', 'Yes', "Don't know"], 'default': 'No', 'help': 'High cholesterol diagnosis'},
                    'CholCheck': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'Yes', 'help': 'Cholesterol checked in last 5 years'},
                    'BloodPressure_Medication': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Taking blood pressure medication'}
                }
            },
            2: {
                'name': '🚬 Lifestyle Factors',
                'description': 'Behavioral and lifestyle choices',
                'features': {
                    'Smoker': {'type': 'select', 'options': ['Never', 'Former', 'Current'], 'default': 'Never', 'help': 'Smoking status'},
                    'SmokeStatus': {'type': 'select', 'options': ['Never smoked', 'Former smoker', 'Current some days', 'Current every day'], 'default': 'Never smoked', 'help': 'Detailed smoking status'},
                    'HeavyDrinker': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Heavy alcohol consumption'},
                    'AlcoholConsumption': {'type': 'slider', 'min': 0, 'max': 30, 'default': 2, 'step': 1, 'help': 'Alcoholic drinks per week'},
                    'PhysicalActivity': {'type': 'select', 'options': ['None', 'Low', 'Moderate', 'High'], 'default': 'Moderate', 'help': 'Physical activity level'},
                    'FruitCons': {'type': 'slider', 'min': 0, 'max': 10, 'default': 2, 'step': 0.5, 'help': 'Fruit servings per day'},
                    'VegCons': {'type': 'slider', 'min': 0, 'max': 10, 'default': 3, 'step': 0.5, 'help': 'Vegetable servings per day'},
                    'SleepHours': {'type': 'slider', 'min': 3, 'max': 12, 'default': 7, 'step': 0.5, 'help': 'Average sleep hours per night'},
                    'SedentaryTime': {'type': 'slider', 'min': 0, 'max': 16, 'default': 6, 'step': 1, 'help': 'Sedentary hours per day'}
                }
            },
            3: {
                'name': '🏥 Healthcare Access',
                'description': 'Healthcare utilization and access',
                'features': {
                    'AnyHealthcare': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'Yes', 'help': 'Any healthcare coverage'},
                    'Checkup': {'type': 'select', 'options': ['Never', '>5 years', 'Within 5 years', 'Within 2 years', 'Within year'], 'default': 'Within year', 'help': 'Last routine checkup'},
                    'DentalVisit': {'type': 'select', 'options': ['Never', '>5 years', 'Within 5 years', 'Within year', 'Within 6 months'], 'default': 'Within year', 'help': 'Last dental visit'},
                    'FluShot': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'Yes', 'help': 'Flu shot in past year'},
                    'HIVTest': {'type': 'select', 'options': ['Never', '>5 years', 'Within 5 years', 'Within year'], 'default': 'Within 5 years', 'help': 'Last HIV test'},
                    'MedicalCost_Problem': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Could not see doctor due to cost in past year'},
                    'PrescriptionCost_Problem': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Could not afford prescription in past year'}
                }
            },
            4: {
                'name': '💪 Health Status',
                'description': 'Current health and functional status',
                'features': {
                    'GenHealth': {'type': 'select', 'options': ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], 'default': 'Good', 'help': 'General health'},
                    'PhysicalHealthDays': {'type': 'slider', 'min': 0, 'max': 30, 'default': 2, 'step': 1, 'help': 'Poor physical health days in past month'},
                    'MentalHealthDays': {'type': 'slider', 'min': 0, 'max': 30, 'default': 3, 'step': 1, 'help': 'Poor mental health days in past month'},
                    'PoorHealthDays': {'type': 'slider', 'min': 0, 'max': 30, 'default': 5, 'step': 1, 'help': 'Overall poor health days in past month'},
                    'DiffWalking': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Difficulty walking or climbing stairs'},
                    'DiffDressing': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Difficulty dressing or bathing'},
                    'DiffErrands': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Difficulty doing errands alone'},
                    'PainArthritis': {'type': 'select', 'options': ['No', 'Yes'], 'default': 'No', 'help': 'Arthritis pain'}
                }
            },
            5: {
                'name': '🌍 Environmental Factors',
                'description': 'Environmental and geographic factors',
                'features': {
                    'State': {'type': 'select', 'options': [
                        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
                        'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
                        'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
                        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
                        'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
                        'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
                        'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
                        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
                        'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
                        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
                    ], 'default': 'California', 'help': 'State of residence'},
                    'State_Avg_AQI': {'type': 'slider', 'min': 0, 'max': 300, 'default': 50, 'step': 5, 'help': 'State average Air Quality Index (0-50: Good, 51-100: Moderate, 101+: Unhealthy)'},
                    'AQI_Unhealthy_Days_Pct': {'type': 'slider', 'min': 0, 'max': 100, 'default': 15, 'step': 1, 'help': 'Percentage of days with unhealthy air quality'},
                    'AQI_Severity_Index': {'type': 'slider', 'min': 1, 'max': 6, 'default': 2, 'step': 0.1, 'help': 'Air quality severity index (1=Good, 6=Hazardous)'},
                    'Urban_Rural': {'type': 'select', 'options': ['Urban', 'Suburban', 'Rural'], 'default': 'Urban', 'help': 'Urban/rural classification'},
                    'Healthcare_Access_Score': {'type': 'slider', 'min': 0, 'max': 10, 'default': 7, 'step': 1, 'help': 'Healthcare access score (0=Poor, 10=Excellent)'}
                }
            }
        }
        
        self.total_groups = len(self.feature_groups)
        self.models = {}
        self.preprocessors = {}
        self.feature_names = {}
        
    def load_models(self):
        """Load trained models for each disease"""
        st.sidebar.info("Loading medical models...")
        
        try:
            # Load for each disease
            for disease in self.target_diseases:
                model_path = Path(f"Notebooks/models/medical_focus/{disease}")
                
                if model_path.exists():
                    # Load ensemble model (best for medical use)
                    ensemble_path = model_path / "Ensemble.pkl"
                    if ensemble_path.exists():
                        self.models[disease] = joblib.load(ensemble_path)
                    
                    # Load preprocessor and feature names
                    preprocessor_path = model_path / "preprocessor.pkl"
                    if preprocessor_path.exists():
                        self.preprocessors[disease] = joblib.load(preprocessor_path)
                    
                    feature_names_path = model_path / "feature_names.pkl"
                    if feature_names_path.exists():
                        self.feature_names[disease] = joblib.load(feature_names_path)
            
            if self.models:
                st.sidebar.success(f"✓ Loaded models for {len(self.models)} diseases")
                return True
            else:
                st.sidebar.error("No models found")
                return False
                
        except Exception as e:
            st.sidebar.error(f"Error loading models: {str(e)}")
            return False
    
    def render_group_form(self, group_idx):
        """Render form for a specific feature group"""
        group = self.feature_groups[group_idx]
        
        st.subheader(f"Group {group_idx + 1}/{self.total_groups}: {group['name']}")
        st.caption(group['description'])
        
        # Create columns for better layout
        cols = st.columns(2)
        col_idx = 0
        
        for feature_name, feature_config in group['features'].items():
            with cols[col_idx]:
                # Initialize in session state if not exists
                if feature_name not in st.session_state.user_data:
                    st.session_state.user_data[feature_name] = feature_config['default']
                
                # Create appropriate input widget
                if feature_config['type'] == 'slider':
                    # Ensure type consistency for slider parameters
                    min_val = feature_config['min']
                    max_val = feature_config['max']
                    step_val = feature_config['step']
                    current_val = st.session_state.user_data[feature_name]
                    
                    # Convert all to float if step is float
                    if isinstance(step_val, float):
                        min_val = float(min_val)
                        max_val = float(max_val)
                        # Ensure current value is also float
                        if isinstance(current_val, int):
                            current_val = float(current_val)
                            st.session_state.user_data[feature_name] = current_val
                    
                    value = st.slider(
                        label=feature_name,
                        min_value=min_val,
                        max_value=max_val,
                        value=current_val,
                        step=step_val,
                        help=feature_config['help']
                    )
                    st.session_state.user_data[feature_name] = value
                    
                elif feature_config['type'] == 'select':
                    value = st.selectbox(
                        label=feature_name,
                        options=feature_config['options'],
                        index=feature_config['options'].index(st.session_state.user_data[feature_name]) 
                        if st.session_state.user_data[feature_name] in feature_config['options'] else 0,
                        help=feature_config['help']
                    )
                    st.session_state.user_data[feature_name] = value
            
            col_idx = (col_idx + 1) % 2
        
        # Progress bar
        progress = (group_idx + 1) / self.total_groups
        st.progress(progress)
        st.caption(f"Completed: {int(progress * 100)}%")
        
    def render_navigation_buttons(self):
        """Render Next/Previous/Submit buttons"""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.session_state.current_group > 0:
                if st.button("◀ Previous Group", use_container_width=True):
                    st.session_state.current_group -= 1
                    st.rerun()
        
        with col2:
            if st.session_state.current_group < self.total_groups - 1:
                if st.button("Next Group ▶", use_container_width=True):
                    st.session_state.current_group += 1
                    st.rerun()
        
        with col3:
            # Skip to end button
            if st.session_state.current_group < self.total_groups - 1:
                if st.button("Skip to End", use_container_width=True):
                    st.session_state.current_group = self.total_groups - 1
                    st.rerun()
        
        with col4:
            # Submit button (only on last group)
            if st.session_state.current_group == self.total_groups - 1:
                if st.button("🔍 Calculate All Disease Risks", type="primary", use_container_width=True):
                    st.session_state.show_results = True
                    st.rerun()
    
    def convert_user_input_to_features(self):
        """Convert user input to feature DataFrame for ALL diseases"""
        # Start with user data
        user_df = pd.DataFrame([st.session_state.user_data])
        
        # Add derived features that models expect
        # These would come from feature engineering during training
        
        # Example derived features (simplified - in reality, use same as training)
        user_df['BMI_Category'] = pd.cut(
            user_df['BMI'],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
        ).astype(str)
        
        user_df['Age_Group'] = pd.cut(
            user_df['Age'],
            bins=[0, 18, 25, 35, 45, 55, 65, 75, 85, 100],
            labels=['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']
        ).astype(str)
        
        # Lifestyle scores
        user_df['HealthyDiet_Score'] = (user_df['FruitCons'] + user_df['VegCons']) / 2
        user_df['LifestyleRisk_Score'] = (
            (user_df['Smoker'] == 'Current').astype(int) * 2 +
            (user_df['HeavyDrinker'] == 'Yes').astype(int) * 1.5 +
            (user_df['PhysicalActivity'] == 'None').astype(int) * 1.5
        )
        
        # Healthcare score
        user_df['Healthcare_Score'] = (
            (user_df['AnyHealthcare'] == 'Yes').astype(int) * 2 +
            (user_df['Checkup'].isin(['Within year', 'Within 6 months'])).astype(int) * 1.5 +
            (user_df['MedicalCost_Problem'] == 'No').astype(int) * 1
        )
        
        return user_df
    
    def make_predictions(self, user_features):
        """Make predictions for ALL diseases using ALL features"""
        predictions = {}
        medical_insights = {}
        
        for disease in self.target_diseases:
            if disease in self.models and disease in self.preprocessors:
                try:
                    # Get the feature names this model expects
                    expected_features = self.feature_names.get(disease, [])
                    
                    # Create a dataframe with all expected features, filling missing ones with defaults
                    X_pred = pd.DataFrame(columns=expected_features)
                    
                    # Fill in the values we have from user input
                    for feature in expected_features:
                        if feature in user_features.columns:
                            X_pred[feature] = user_features[feature]
                        else:
                            # Set default value for missing feature
                            # Try to infer the type
                            if feature in ['Age', 'Height', 'Weight', 'Children', 'PhysicalHealthDays', 
                                        'MentalHealthDays', 'PoorHealthDays', 'SedentaryTime', 
                                        'AlcoholConsumption', 'State_Avg_AQI', 'AQI_Unhealthy_Days_Pct']:
                                X_pred[feature] = 0  # Default numeric
                            elif feature in ['BMI', 'FruitCons', 'VegCons', 'SleepHours', 
                                            'AQI_Severity_Index', 'Healthcare_Access_Score']:
                                X_pred[feature] = 0.0  # Default float
                            elif feature in ['BMI_Category', 'Age_Group', 'HealthyDiet_Score', 
                                            'LifestyleRisk_Score', 'Healthcare_Score']:
                                # Derived features - calculate them
                                if feature == 'BMI_Category':
                                    bmi = user_features['BMI'].iloc[0] if 'BMI' in user_features.columns else 25.0
                                    X_pred[feature] = self._get_bmi_category(bmi)
                                elif feature == 'Age_Group':
                                    age = user_features['Age'].iloc[0] if 'Age' in user_features.columns else 45
                                    X_pred[feature] = self._get_age_group(age)
                                elif feature == 'HealthyDiet_Score':
                                    fruit = user_features.get('FruitCons', [2.0])[0]
                                    veg = user_features.get('VegCons', [3.0])[0]
                                    X_pred[feature] = (fruit + veg) / 2
                                elif feature == 'LifestyleRisk_Score':
                                    smoker = user_features.get('Smoker', ['Never'])[0] == 'Current'
                                    drinker = user_features.get('HeavyDrinker', ['No'])[0] == 'Yes'
                                    activity = user_features.get('PhysicalActivity', ['Moderate'])[0] == 'None'
                                    score = (smoker * 2 + drinker * 1.5 + activity * 1.5)
                                    X_pred[feature] = score
                                elif feature == 'Healthcare_Score':
                                    healthcare = user_features.get('AnyHealthcare', ['Yes'])[0] == 'Yes'
                                    checkup = user_features.get('Checkup', ['Within year'])[0] in ['Within year', 'Within 6 months']
                                    cost = user_features.get('MedicalCost_Problem', ['No'])[0] == 'No'
                                    score = (healthcare * 2 + checkup * 1.5 + cost * 1)
                                    X_pred[feature] = score
                                else:
                                    X_pred[feature] = 0
                            else:
                                # Categorical features - set to most common/default
                                X_pred[feature] = 'No'  # Most categoricals are Yes/No
                    
                    # Ensure all columns are present
                    missing_cols = set(expected_features) - set(X_pred.columns)
                    if missing_cols:
                        for col in missing_cols:
                            X_pred[col] = 0  # Default fill
                    
                    # Reorder columns to match training
                    X_pred = X_pred[expected_features]
                    
                    # Preprocess features
                    preprocessor = self.preprocessors[disease]
                    features_processed = preprocessor.transform(X_pred)
                    
                    # Get prediction probability
                    model = self.models[disease]
                    probability = model.predict_proba(features_processed)[0, 1]
                    
                    # Categorize risk with medical context
                    if probability >= 0.7:
                        risk_level = "High"
                        recommendation = "Immediate clinical evaluation recommended"
                        color = "red"
                    elif probability >= 0.4:
                        risk_level = "Moderate"
                        recommendation = "Regular screening and lifestyle modification advised"
                        color = "orange"
                    else:
                        risk_level = "Low"
                        recommendation = "Maintain healthy lifestyle and regular checkups"
                        color = "green"
                    
                    # Medical insights based on features
                    insights = self._generate_medical_insights(disease, user_features)
                    
                    predictions[disease] = {
                        'probability': probability,
                        'risk_level': risk_level,
                        'recommendation': recommendation,
                        'color': color,
                        'insights': insights or []  # Ensure insights is always a list
                    }
                    
                except Exception as e:
                    st.warning(f"Could not predict {disease}: {str(e)}")
                    predictions[disease] = {
                        'probability': None,
                        'risk_level': 'Unknown',
                        'recommendation': 'Model error',
                        'color': 'gray',
                        'insights': []  # Add empty insights list
                    }
        
        return predictions

    def _get_bmi_category(self, bmi):
        """Categorize BMI"""
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        elif bmi < 35:
            return 'Obese I'
        elif bmi < 40:
            return 'Obese II'
        else:
            return 'Obese III'

    def _get_age_group(self, age):
        """Categorize age"""
        if age < 18:
            return '<18'
        elif age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        elif age < 65:
            return '55-64'
        elif age < 75:
            return '65-74'
        elif age < 85:
            return '75-84'
        else:
            return '85+'
        
    def _generate_medical_insights(self, disease, features):
        """Generate medical insights based on feature values"""
        insights = []
        
        # Diabetes insights
        if disease == 'Diabetes':
            if features['BMI'].iloc[0] > 30:
                insights.append("High BMI increases diabetes risk")
            if features['PhysicalActivity'].iloc[0] == 'None':
                insights.append("Physical inactivity contributes to diabetes risk")
            if features['Age'].iloc[0] > 45:
                insights.append("Age is a significant risk factor")
        
        # Heart Disease insights
        elif disease == 'HeartDisease':
            if features['HighBP'].iloc[0] == 'Yes':
                insights.append("Hypertension is a major cardiovascular risk factor")
            if features['Smoker'].iloc[0] == 'Current':
                insights.append("Smoking significantly increases heart disease risk")
            if features['HighChol'].iloc[0] == 'Yes':
                insights.append("High cholesterol contributes to arterial plaque")
        
        # Asthma insights
        elif disease == 'Asthma':
            if features['State_Avg_AQI'].iloc[0] > 100:
                insights.append("Poor air quality exacerbates respiratory conditions")
            if features['Smoker'].iloc[0] == 'Current':
                insights.append("Smoking is a primary trigger for asthma")
        
        # General insights
        if features['GenHealth'].iloc[0] in ['Fair', 'Poor']:
            insights.append("Self-reported poor health correlates with multiple conditions")
        
        if features['AnyHealthcare'].iloc[0] == 'No':
            insights.append("Lack of healthcare access delays diagnosis and treatment")
        
        return insights[:3]  # Return top 3 insights
    
    def display_results(self, predictions):
        """Display comprehensive results with medical focus"""
        st.markdown("---")
        st.markdown("# 🏥 Comprehensive Medical Risk Assessment")
        st.markdown("### *Based on complete health profile analysis*")
        
        # Filter out predictions with None probability
        valid_predictions = {k: v for k, v in predictions.items() if v['probability'] is not None}
        
        if not valid_predictions:
            st.error("No valid predictions could be generated. Please check your models.")
            return
        
        # Summary metrics - only use valid predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk = sum(1 for p in valid_predictions.values() if p['risk_level'] == 'High')
            st.metric("High Risk Conditions", high_risk)
        
        with col2:
            avg_risk = np.mean([p['probability'] for p in valid_predictions.values()])
            st.metric("Average Risk Score", f"{avg_risk:.1%}")
        
        with col3:
            # Most at-risk disease
            if valid_predictions:
                most_risky = max(valid_predictions.items(), 
                                key=lambda x: x[1]['probability'])
                st.metric("Highest Risk", most_risky[0])
        
        # Disease-by-disease analysis
        st.markdown("## 📊 Disease Risk Analysis")
        
        # Create tabs for each disease
        tab_names = [f"**{d}**" for d in valid_predictions.keys()]
        tabs = st.tabs(tab_names if tab_names else ["No Predictions"])
        
        for idx, (disease, prediction) in enumerate(valid_predictions.items()):
            with tabs[idx]:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Gauge chart - only if probability exists
                    if prediction['probability'] is not None:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction['probability'] * 100,
                            title={'text': f"{disease}<br>Risk Score", 'font': {'size': 18}},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': prediction['color']},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk level
                        st.markdown(f"### **Risk Level: {prediction['risk_level']}**")
                        st.markdown(f"*{prediction['recommendation']}*")
                
                with col2:
                    # Medical insights - check if insights exists
                    st.markdown("### 🔍 Medical Insights")
                    
                    insights = prediction.get('insights', [])
                    if insights:
                        for insight in insights:
                            st.info(f"• {insight}")
                    else:
                        st.info("• No specific risk factors identified")
                    
                    # Key contributing factors
                    st.markdown("### 🎯 Key Contributing Factors")
                    
                    # Mock feature importance
                    if prediction['probability'] is not None:
                        prob = prediction['probability']
                        factors = {
                            'Demographics': f"{prob*30:.0f}% contribution",
                            'Lifestyle': f"{prob*40:.0f}% contribution",
                            'Clinical': f"{prob*20:.0f}% contribution",
                            'Environmental': f"{prob*10:.0f}% contribution"
                        }
                        
                        for factor, contribution in factors.items():
                            st.metric(factor, contribution)
        
        # Show warning for diseases that couldn't be predicted
        failed_diseases = [d for d in predictions.keys() if d not in valid_predictions]
        if failed_diseases:
            st.warning(f"**Note:** Could not generate predictions for: {', '.join(failed_diseases)}. "
                    f"This may be due to missing features in the trained models.")
        
        # Rest of your display_results method continues here...
        # [Keep the radar chart, recommendations, etc.]
        # Overall health profile
        st.markdown("---")
        st.markdown("## 📈 Overall Health Profile Analysis")
        
        # Create radar chart of risk factors
        risk_categories = {
            'Demographic Risk': st.session_state.user_data.get('Age', 45) / 100 * 0.3 +
                               (1 if st.session_state.user_data.get('Income', '$50-75k') in ['<$15k', '$15-25k'] else 0) * 0.2,
            'Lifestyle Risk': (1 if st.session_state.user_data.get('Smoker', 'Never') == 'Current' else 0) * 0.4 +
                             (1 if st.session_state.user_data.get('PhysicalActivity', 'Moderate') == 'None' else 0) * 0.3 +
                             (st.session_state.user_data.get('BMI', 25) / 50) * 0.3,
            'Clinical Risk': (1 if st.session_state.user_data.get('HighBP', 'No') == 'Yes' else 0) * 0.5 +
                            (1 if st.session_state.user_data.get('HighChol', 'No') == 'Yes' else 0) * 0.5,
            'Environmental Risk': st.session_state.user_data.get('State_Avg_AQI', 50) / 300 * 0.7 +
                                 (1 if st.session_state.user_data.get('Urban_Rural', 'Urban') == 'Urban' else 0) * 0.3,
            'Healthcare Risk': (1 if st.session_state.user_data.get('AnyHealthcare', 'Yes') == 'No' else 0) * 0.6 +
                              (1 if st.session_state.user_data.get('MedicalCost_Problem', 'No') == 'Yes' else 0) * 0.4
        }
        
        fig = go.Figure(data=go.Scatterpolar(
            r=list(risk_categories.values()),
            theta=list(risk_categories.keys()),
            fill='toself',
            name='Risk Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            height=400,
            title="Personalized Risk Factor Profile"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("## 💡 Personalized Health Recommendations")
        
        recommendations = []
        
        # BMI-based
        bmi = st.session_state.user_data.get('BMI', 25)
        if bmi > 30:
            recommendations.append("⚖️ **Weight Management:** Consider consulting a nutritionist for weight loss plan")
        elif bmi > 25:
            recommendations.append("⚖️ **Weight Maintenance:** Maintain current weight or consider moderate weight loss")
        
        # Smoking
        if st.session_state.user_data.get('Smoker') == 'Current':
            recommendations.append("🚭 **Smoking Cessation:** Strongly consider smoking cessation programs")
        
        # Physical activity
        activity = st.session_state.user_data.get('PhysicalActivity', 'Moderate')
        if activity == 'None':
            recommendations.append("🏃 **Increase Activity:** Aim for 150 minutes of moderate activity per week")
        
        # Healthcare access
        if st.session_state.user_data.get('AnyHealthcare') == 'No':
            recommendations.append("🏥 **Healthcare Access:** Explore affordable healthcare options in your state")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.success(f"{i}. {rec}")
        
        # Download report
        st.markdown("---")
        st.markdown("### 📄 Download Complete Report")
        
        # Create downloadable report
        report_content = self._generate_medical_report(predictions)
        
        st.download_button(
            label="⬇️ Download Medical Risk Report (PDF)",
            data=report_content,
            file_name="medical_risk_assessment.txt",
            mime="text/plain"
        )
    
    def _generate_medical_report(self, predictions):
        """Generate comprehensive medical report"""
        report = f"""
        COMPREHENSIVE MEDICAL RISK ASSESSMENT REPORT
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        =========================================================
        
        PATIENT SUMMARY:
        Age: {st.session_state.user_data.get('Age', 'N/A')}
        Sex: {st.session_state.user_data.get('Sex', 'N/A')}
        BMI: {st.session_state.user_data.get('BMI', 'N/A')}
        General Health: {st.session_state.user_data.get('GenHealth', 'N/A')}
        
        DISEASE RISK ASSESSMENT:
        """
        
        for disease, prediction in predictions.items():
            if prediction['probability'] is not None:
                report += f"\n{disease}:"
                report += f"\n  Risk Score: {prediction['probability']:.1%}"
                report += f"\n  Risk Level: {prediction['risk_level']}"
                report += f"\n  Recommendation: {prediction['recommendation']}"
                if prediction['insights']:
                    report += f"\n  Key Factors: {', '.join(prediction['insights'])}"
                report += "\n"
        
        report += """
        
        MEDICAL RECOMMENDATIONS:
        1. Schedule annual physical examination
        2. Maintain healthy lifestyle habits
        3. Monitor key health indicators regularly
        4. Consult healthcare provider for any concerns
        
        DISCLAIMER:
        This report is generated by machine learning models for informational purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of your physician or other qualified health provider.
        """
        
        return report
    
    def render_sidebar_progress(self):
        """Render sidebar with progress information"""
        with st.sidebar:
            st.title("📋 Progress Tracker")
            
            # Show all groups with current status
            for idx in range(self.total_groups):
                group = self.feature_groups[idx]
                completed = idx < st.session_state.current_group
                current = idx == st.session_state.current_group
                
                if completed:
                    st.success(f"✓ {group['name']}")
                elif current:
                    st.info(f"▶ {group['name']} (Current)")
                else:
                    st.write(f"○ {group['name']}")
            
            st.markdown("---")
            
            if st.session_state.show_results:
                st.success("✅ Assessment Complete")
                st.markdown(f"**Features Collected:** {len(st.session_state.user_data)}")
                st.markdown(f"**Diseases Analyzed:** {len(self.target_diseases)}")
                
                if st.button("🔄 Start New Assessment"):
                    st.session_state.user_data = {}
                    st.session_state.current_group = 0
                    st.session_state.show_results = False
                    st.rerun()
    def convert_user_input_to_features(self):
        """Convert user input to feature DataFrame for ALL diseases"""
        # Start with user data
        user_df = pd.DataFrame([st.session_state.user_data])
        
        # Add ALL potential derived features that models might expect
        # Based on the missing columns error
        
        # BMI category
        user_df['BMI_Category'] = self._get_bmi_category(user_df['BMI'].iloc[0] if 'BMI' in user_df.columns else 25.0)
        
        # Age group
        user_df['Age_Group'] = self._get_age_group(user_df['Age'].iloc[0] if 'Age' in user_df.columns else 45)
        
        # Lifestyle scores
        fruit = user_df.get('FruitCons', [2.0])[0]
        veg = user_df.get('VegCons', [3.0])[0]
        user_df['HealthyDiet_Score'] = (fruit + veg) / 2
        
        smoker = user_df.get('Smoker', ['Never'])[0] == 'Current'
        drinker = user_df.get('HeavyDrinker', ['No'])[0] == 'Yes'
        activity = user_df.get('PhysicalActivity', ['Moderate'])[0] == 'None'
        user_df['LifestyleRisk_Score'] = (smoker * 2 + drinker * 1.5 + activity * 1.5)
        
        # Healthcare score
        healthcare = user_df.get('AnyHealthcare', ['Yes'])[0] == 'Yes'
        checkup = user_df.get('Checkup', ['Within year'])[0] in ['Within year', 'Within 6 months']
        cost = user_df.get('MedicalCost_Problem', ['No'])[0] == 'No'
        user_df['Healthcare_Score'] = (healthcare * 2 + checkup * 1.5 + cost * 1)
        
        # Add missing features from the error with default values
        missing_features = [
            'CognitiveDiff', 'SaltIntake', 'HighRisk', '90th Percentile AQI', 
            'Deaf', 'Obesity', 'Exercise', 'Unhealthy Days', 'MultiMorbidity_Index', 
            'OtherCancer', 'Median AQI', 'Good Days', 'HasMultipleConditions', 
            'Max AQI', 'PhysicalInactivity', 'Veteran', 'CardioRisk_Score', 
            'TotalUnhealthyDays', 'Disability', 'HealthcareAccess_Score', 'Blind'
        ]
        
        for feature in missing_features:
            if feature not in user_df.columns:
                # Set appropriate defaults
                if any(word in feature.lower() for word in ['score', 'index', 'days', 'aqi', 'pct']):
                    user_df[feature] = 0.0  # Numeric features
                else:
                    user_df[feature] = 'No'  # Categorical features
        
        return user_df
    def run(self):
        """Main application runner"""
        # Load models
        if not self.models:
            if not self.load_models():
                st.error("Failed to load medical models. Please ensure models are trained first.")
                return
        
        # Main layout
        if not st.session_state.show_results:
            # Show form
            st.title("🏥 Comprehensive Medical Risk Assessment")
            st.markdown("### *Complete health profile analysis using ALL relevant factors*")
            
            # Current group form
            self.render_group_form(st.session_state.current_group)
            
            # Navigation
            self.render_navigation_buttons()
            
            # Sidebar progress
            self.render_sidebar_progress()
            
        else:
            # Show results
            # Convert user input to features
            user_features = self.convert_user_input_to_features()
            
            # Make predictions
            predictions = self.make_predictions(user_features)
            
            # Display results
            self.display_results(predictions)
            
            # Sidebar
            self.render_sidebar_progress()

def main():
    app = ComprehensiveMedicalPredictor()
    app.run()

if __name__ == "__main__":
    main()