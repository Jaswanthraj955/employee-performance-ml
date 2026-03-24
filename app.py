import streamlit as st
import pickle
import pandas as pd

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))

st.title("Employee Performance Prediction System")

st.write("Provide employee details to predict performance level")

# ===== USER INPUTS =====
age = st.slider("Age",18,60,30)
distance = st.slider("Distance From Home",0,30,5)
salary_hike = st.slider("Last Salary Hike %",0,30,10)
experience = st.slider("Total Work Experience (Years)",0,40,5)
training = st.slider("Training Times Last Year",0,10,2)
overtime = st.selectbox("OverTime",["No","Yes"])

# ===== FULL TRAINING COLUMN LIST =====
cols = ['Age',
 'DistanceFromHome',
 'EmpEducationLevel',
 'EmpEnvironmentSatisfaction',
 'EmpHourlyRate',
 'EmpJobInvolvement',
 'EmpJobLevel',
 'EmpJobSatisfaction',
 'NumCompaniesWorked',
 'EmpLastSalaryHikePercent',
 'EmpRelationshipSatisfaction',
 'TotalWorkExperienceInYears',
 'TrainingTimesLastYear',
 'EmpWorkLifeBalance',
 'ExperienceYearsAtThisCompany',
 'ExperienceYearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager',
 'Gender_Male',
 'EducationBackground_Life Sciences',
 'EducationBackground_Marketing',
 'EducationBackground_Medical',
 'EducationBackground_Other',
 'EducationBackground_Technical Degree',
 'MaritalStatus_Married',
 'MaritalStatus_Single',
 'EmpDepartment_Development',
 'EmpDepartment_Finance',
 'EmpDepartment_Human Resources',
 'EmpDepartment_Research & Development',
 'EmpDepartment_Sales',
 'EmpJobRole_Data Scientist',
 'EmpJobRole_Delivery Manager',
 'EmpJobRole_Developer',
 'EmpJobRole_Finance Manager',
 'EmpJobRole_Healthcare Representative',
 'EmpJobRole_Human Resources',
 'EmpJobRole_Laboratory Technician',
 'EmpJobRole_Manager',
 'EmpJobRole_Manager R&D',
 'EmpJobRole_Manufacturing Director',
 'EmpJobRole_Research Director',
 'EmpJobRole_Research Scientist',
 'EmpJobRole_Sales Executive',
 'EmpJobRole_Sales Representative',
 'EmpJobRole_Senior Developer',
 'EmpJobRole_Senior Manager R&D',
 'EmpJobRole_Technical Architect',
 'EmpJobRole_Technical Lead',
 'BusinessTravelFrequency_Travel_Frequently',
 'BusinessTravelFrequency_Travel_Rarely',
 'OverTime_Yes',
 'Attrition_Yes',
 'exp_bucket_Mid',
 'exp_bucket_Senior']

# ===== CREATE DEFAULT INPUT VECTOR =====
input_dict = dict.fromkeys(cols,0)

# ===== FILL IMPORTANT USER INPUTS =====
input_dict['Age'] = age
input_dict['DistanceFromHome'] = distance
input_dict['EmpLastSalaryHikePercent'] = salary_hike
input_dict['TotalWorkExperienceInYears'] = experience
input_dict['TrainingTimesLastYear'] = training

if overtime == "Yes":
    input_dict['OverTime_Yes'] = 1

input_df = pd.DataFrame([input_dict])

# ===== PREDICTION BLOCK =====
if st.button("Predict Performance"):
    
    pred = model.predict(input_df)[0]

    performance_map = {
        0: "Low Performer",
        1: "Average Performer",
        2: "Good Performer",
        3: "Excellent Performer"
    }

    st.success(f"Predicted Performance Level : {performance_map[pred]}")

    prob = model.predict_proba(input_df).max()

    st.info(f"Model Confidence Score : {round(prob*100,2)} %")