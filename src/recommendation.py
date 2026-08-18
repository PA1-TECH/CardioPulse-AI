class LifestyleRecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(self, patient_data, risk_probability):
        """
        patient_data: dict containing feature key-value pairs:
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        risk_probability: float between 0.0 and 1.0 (e.g. 0.72)
        """
        age = float(patient_data.get('age', 50))
        sex = int(patient_data.get('sex', 1))
        cp = int(patient_data.get('cp', 0))
        trestbps = float(patient_data.get('trestbps', 120))
        chol = float(patient_data.get('chol', 200))
        fbs = int(patient_data.get('fbs', 0))
        thalach = float(patient_data.get('thalach', 150))
        exang = int(patient_data.get('exang', 0))
        oldpeak = float(patient_data.get('oldpeak', 0.0))
        ca = int(patient_data.get('ca', 0))
        
        # Risk Stratification
        if risk_probability >= 0.75:
            risk_category = "Very High Risk"
            risk_color = "#DC2626" # Red
            urgency = "Immediate Medical Consultation Required"
        elif risk_probability >= 0.50:
            risk_category = "High Risk"
            risk_color = "#EA580C" # Orange
            urgency = "Medical Evaluation Recommended within 1-2 weeks"
        elif risk_category_val := (risk_probability >= 0.25):
            risk_category = "Moderate Risk"
            risk_color = "#D97706" # Amber
            urgency = "Routine Preventive Care & Lifestyle Optimization"
        else:
            risk_category = "Low Risk"
            risk_color = "#059669" # Green
            urgency = "Maintain Healthy Lifestyle & Annual Checkup"
            
        dietary_advice = []
        bp_advice = []
        exercise_advice = []
        medical_advice = []
        monitoring_advice = []
        
        # 1. Blood Pressure Management (trestbps)
        if trestbps >= 140:
            bp_advice.append("Stage 2 Hypertension detected (>=140 mmHg). Adopt DASH (Dietary Approaches to Stop Hypertension) diet strictly.")
            bp_advice.append("Restrict daily sodium intake to < 1,500 mg/day.")
            bp_advice.append("Consult a physician regarding antihypertensive pharmacological therapy.")
        elif trestbps >= 130:
            bp_advice.append("Stage 1 Hypertension detected (130-139 mmHg). Limit daily sodium intake to < 2,300 mg/day.")
            bp_advice.append("Increase dietary potassium through bananas, spinach, and sweet potatoes unless contraindicated.")
        elif trestbps >= 120:
            bp_advice.append("Elevated Resting Blood Pressure (120-129 mmHg). Focus on weight management and stress reduction.")
        else:
            bp_advice.append("Resting blood pressure is within normal healthy range (<120 mmHg). Keep maintaining low sodium habits.")
            
        # 2. Lipid & Cholesterol Guidance (chol)
        if chol >= 240:
            dietary_advice.append("High Serum Cholesterol (>=240 mg/dL). Eliminate trans fats and restrict saturated fats to <5-6% of daily calories.")
            dietary_advice.append("Incorporate 10-25g of soluble fiber daily (oat bran, legumes, psyllium husk) to reduce LDL-C absorption.")
            dietary_advice.append("Add plant stanols/sterols (2g/day) and omega-3 fatty acids (salmon, walnuts, flaxseed).")
        elif chol >= 200:
            dietary_advice.append("Borderline High Serum Cholesterol (200-239 mg/dL). Replace butter/lard with extra virgin olive oil or avocado oil.")
            dietary_advice.append("Increase lean protein sources (poultry, fish, tofu) and reduce red/processed meats.")
        else:
            dietary_advice.append("Cholesterol levels are within normal range (<200 mg/dL). Continue a heart-healthy Mediterranean diet.")
            
        # 3. Glycemic & Sugar Control (fbs)
        if fbs == 1:
            dietary_advice.append("Elevated Fasting Blood Sugar (>120 mg/dL). Follow a low-glycemic index (GI) diet to reduce glucose spikes.")
            dietary_advice.append("Avoid sugar-sweetened beverages, refined carbohydrates, and pastry products.")
            dietary_advice.append("Schedule HbA1c screening to evaluate long-term glycemic control.")
        else:
            dietary_advice.append("Fasting blood sugar is normal (<=120 mg/dL). Maintain balanced complex carbohydrate intake.")
            
        # 4. Exercise & Physical Activity (thalach, exang, age)
        target_max_hr = 220 - age
        hr_percentage = (thalach / target_max_hr) * 100 if target_max_hr > 0 else 100
        
        if exang == 1 or oldpeak >= 2.0:
            exercise_advice.append("Exercise-Induced Angina or Significant ST Depression detected. DO NOT start vigorous unmonitored exercise.")
            exercise_advice.append("Engage only in medically supervised cardiac rehabilitation exercise programs (light walking 15-20 mins).")
        elif risk_probability >= 0.5:
            exercise_advice.append("Begin low-to-moderate intensity aerobic exercises (brisk walking, stationary cycling) 30 mins/day, 5 days/week.")
            exercise_advice.append("Monitor heart rate during exercise, keeping HR below 60-70% of maximum capacity.")
        else:
            exercise_advice.append("Perform moderate-to-vigorous aerobic exercise 150-300 minutes per week (cycling, swimming, running).")
            exercise_advice.append("Include muscle-strengthening activities at least 2 days a week.")
            
        # 5. Clinical & Diagnostic Follow-ups (cp, oldpeak, ca)
        if cp == 0 and risk_probability >= 0.5:
            medical_advice.append("Asymptomatic presentation despite elevated risk indicators. Recommended: Non-invasive stress imaging / Coronary Calcium score.")
        elif cp > 0:
            cp_types = {1: "Atypical Angina", 2: "Non-anginal pain", 3: "Typical Angina"}
            medical_advice.append(f"Reported chest pain status: {cp_types.get(cp, 'Chest discomfort')}. Clinical cardiology consultation strongly recommended.")
            
        if oldpeak >= 1.0:
            medical_advice.append(f"ST depression during exercise ({oldpeak} mm) suggests myocardial ischemia. Further cardiac evaluation is essential.")
            
        if ca > 0:
            medical_advice.append(f"Fluoroscopy shows {ca} major vessels involved. Close follow-up with a cardiologist for coronary artery disease management.")
            
        # 6. Monitoring & Lifestyle Habits
        monitoring_advice.append("Monitor blood pressure twice daily (morning & evening) and log readings.")
        monitoring_advice.append("Obtain a comprehensive fasting lipid panel and CRP test every 3-6 months.")
        monitoring_advice.append("Practice daily stress reduction (mindfulness, deep breathing, adequate sleep 7-8 hrs/night).")
        if sex == 1:
            monitoring_advice.append("Ensure tobacco products are strictly avoided (smoking increases CVD risk by >200%).")
            
        return {
            'risk_category': risk_category,
            'risk_percentage': round(risk_probability * 100, 1),
            'risk_color': risk_color,
            'urgency': urgency,
            'dietary_advice': dietary_advice,
            'bp_advice': bp_advice,
            'exercise_advice': exercise_advice,
            'medical_advice': medical_advice,
            'monitoring_advice': monitoring_advice
        }

if __name__ == "__main__":
    engine = LifestyleRecommendationEngine()
    test_patient = {
        'age': 60, 'sex': 1, 'cp': 0, 'trestbps': 145, 'chol': 260,
        'fbs': 1, 'restecg': 1, 'thalach': 120, 'exang': 1, 'oldpeak': 2.5,
        'slope': 1, 'ca': 2, 'thal': 3
    }
    rec = engine.generate_recommendations(test_patient, risk_probability=0.82)
    print("Risk Category:", rec['risk_category'])
    print("Dietary Advice:", rec['dietary_advice'])
