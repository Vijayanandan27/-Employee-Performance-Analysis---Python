# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:54:30 2025

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.io as pio

# Step 1: Upload the data

emp = pd.read_excel(r"D:\01. Portfolio projects\Python Practice\Project 5 (Employee Data Analysis)\Employee Performance Dataset_C30_English.xlsx",sheet_name="employees")

stores = pd.read_excel(r"D:\01. Portfolio projects\Python Practice\Project 5 (Employee Data Analysis)\Employee Performance Dataset_C30_English.xlsx",sheet_name="stores")

perform = pd.read_excel(r"D:\01. Portfolio projects\Python Practice\Project 5 (Employee Data Analysis)\Employee Performance Dataset_C30_English.xlsx",sheet_name="monthly_performance")

kpi = pd.read_excel(r"D:\01. Portfolio projects\Python Practice\Project 5 (Employee Data Analysis)\Employee Performance Dataset_C30_English.xlsx",sheet_name="role_kpis")

Outcome = pd.read_excel(r"D:\01. Portfolio projects\Python Practice\Project 5 (Employee Data Analysis)\Employee Performance Dataset_C30_English.xlsx",sheet_name="business_outcomes")

# Step 2 : Data cleaning and quality

# Employee table cleanup
emp.info()
emp['Hire_Date'] = pd.to_datetime(emp['Hire_Date'], format='mixed')
emp['Exit_Date'] = pd.to_datetime(emp['Exit_Date'], format='mixed')
emp.isnull().sum()

emp[emp.isnull().any(axis=1)]

emp.duplicated().sum()
print(f"No of duplicated rows{emp.duplicated().sum()}")

emp['Exit_Date']= emp['Exit_Date'].fillna('Active')

#stores table cleanup
stores.info()

stores['Opening_Date']=pd.to_datetime(stores['Opening_Date'], format='mixed')

stores.isnull().sum()

stores[stores.isnull().any(axis=1)]

stores.duplicated().sum()

if stores['Store_Id'].is_unique:
    print("Unique")

# employee performance cleanup
perform.info()
perform['Year_Month'] = pd.to_datetime(perform['Year_Month'], format='%Y-%m')

perform.isnull().sum()

perform[perform.isnull().any(axis=1)]

perform.duplicated().sum()

# role Kpi table cleanup

kpi.info()
kpi['Year_Month']=pd.to_datetime(kpi['Year_Month'], format='%Y-%m')

kpi.isnull().sum()

kpi[kpi.isnull().any(axis=1)]

kpi.duplicated().sum()

# business outcome table cleanup

Outcome.info()

Outcome['Year_Month']=pd.to_datetime(Outcome['Year_Month'], format='%Y-%m')
Outcome.isnull().sum()

Outcome[Outcome.isnull().any(axis=1)]

Outcome.duplicated().sum()

dup_perf = perform[perform.duplicated(subset = ["Employee_Id", "Year_Month"])]

# Step 3: Workforce Demographics

# Age distribution by department & job level.

bins =[0,24,35,45,100]
labels = ['<25', '25-35', '36-45', '46+']

emp['Age Group'] =pd.cut(emp['Age'], bins=bins, labels=labels, right=True)

emp[['Employee_Id', 'Full_Name', 'Age', 'Age Group']].head()

Age_Dist = emp.groupby(['Department', 'Job_Level','Age Group'])['Age'].count().reset_index(name='Count')

Age_dist_pivot = Age_Dist.pivot(index='Department', columns = ['Job_Level', 'Age Group'], values='Count')

Age_dist_plot=sns.barplot(Age_dist_pivot, )

#Visual - Barchart

Age_Dist['Job_Level'] = Age_Dist['Job_Level'].astype(str)
Age_Dist['Age Group'] = Age_Dist['Age Group'].astype(str)

Age_Dist['Job_Age']= Age_Dist['Job_Level']+'-'+ Age_Dist['Age Group']

plt.figure(figsize=(20,16))

sns.barplot(Age_Dist, x= 'Department', y='Count', hue='Job_Age', ci=None)

plt.title('Age Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Total Count')
plt.show()


#visual - Heat map

plt.figure(figsize=(20,16))

sns.heatmap(Age_dist_pivot, cmap='YlGnBu', linewidths=0.5, annot=True, fmt='d')

plt.title('Age Distribution by Department and joblevel')
plt.ylabel('Department')
plt.xlabel('Job Level - Age Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Education level vs. job roles.

Edul=emp.groupby(['Education_Level','Job_Role'])['Job_Role'].count().reset_index(name='Total Count')

# Visual bar chart

plt.figure(figsize=(20,10))
sns.barplot(Edul, x='Education_Level', y='Total Count', hue='Job_Role')

plt.title('Education level vs. job roles')
plt.xlabel('Education Level')
plt.ylabel('Job_Role') 
plt.legend(title="Job Role", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#visual - Heat map

Edul_pivot=Edul.pivot(index = 'Education_Level', columns='Job_Role', values='Total Count')
plt.figure(figsize=(20,10))

sns.heatmap(Edul_pivot, cmap='viridis', fmt='.0f', annot=True, linewidths=1.0)
plt.title('Education level vs. job roles')
plt.xlabel('Education Level')
plt.ylabel('Job Roles')
plt.tight_layout()
plt.show()

# Department-wise headcount

HC_department=emp.groupby('Department')['Employee_Id'].count().reset_index(name='Total Count')

plt.figure(figsize=(20,15))
plt.bar(HC_department['Department'], HC_department['Total Count'], color='skyblue', edgecolor='black')

plt.title('Department wise Headcount', fontsize = 16)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Total Employee', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# Step 4 – Performance Analysis

# Avg performance rating by department, role, job level

cmb_emp=perform.merge(emp, on='Employee_Id', how='inner')
cmb_emp.info()

cmb_emp.duplicated().sum()
cmb_emp['Employee_Id'].duplicated().sum()

Avg_Perf_rating = cmb_emp.groupby(['Department','Job_Role',"Job_Level"])['Performance_Rating'].mean().reset_index(name='Avg Performance rating')

Avg_Perf_rating.head()

cmb_emp.to_csv('cmb_emp.csv', index=False)

print(os.getcwd())

Avg_Perf_rating_pivot = Avg_Perf_rating.pivot(index=['Department','Job_Role'], columns='Job_Level', values='Avg Performance rating')
Avg_Perf_rating_pivot.index=Avg_Perf_rating_pivot.index.map(lambda x: f"{x[0]}-{x[1]}")
plt.figure(figsize=(10,8))
sns.heatmap(Avg_Perf_rating_pivot, cmap='viridis', annot=True, fmt='.0f', linewidths=0.75)
plt.title('Avg performance rating by department, role, job level')
plt.tight_layout()
plt.show()

# Effect of training hours, overtime, absenteeism on performance

Corr_val=['Performance_Rating','Training_Hours', 'Overtime_Hours', 'Absenteeism_Days']

Corr_op= cmb_emp[Corr_val].corr()

plt.figure(figsize=(18,18))
sns.heatmap(Corr_op, cmap='coolwarm', annot=True, fmt='.0f', linewidths=0.75, annot_kws={"size":25})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# Relationship between manager evaluations and employee performance.

plt.figure(figsize=(20,18))
sns.scatterplot(data=cmb_emp, x='Manager_Evaluation', y='Performance_Rating')
plt.xlabel('Manager_Evaluation')
plt.ylabel('Performance_Rating')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,18))
sns.regplot(data=cmb_emp, x='Manager_Evaluation', y='Performance_Rating', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.xlabel('Manager_Evaluation')
plt.ylabel('Performance_Rating')
plt.tight_layout()
plt.show()

# Identify top-performing employees vs. low performers

def classify_performance(rating):
    if rating >= 4:
        return "Top_performer"
    elif rating <= 2:
        return "Low_performer"
    else:
        return "Average_performer"
    
cmb_emp['Performace_category']=cmb_emp['Performance_Rating'].apply(classify_performance)

cmb_emp['Performace_category'].value_counts()

# Step 5 - Employee Tenure & Retention

# Average tenure by department and store

emp['Hire_Date'] = pd.to_datetime(emp['Hire_Date'])
emp['Exit_Date'] = pd.to_datetime(emp['Exit_Date'], errors ='coerce')


def tenure_years(row):
    if pd.isna(row['Exit_Date']):
        return (pd.Timestamp.today() - row['Hire_Date']).days / 365
    else:
        return (row['Exit_Date'] - row['Hire_Date']).days / 365
    
emp['Tenure'] = emp.apply(tenure_years, axis = 1)

Avg_Depart_tenure = emp.groupby('Department')['Tenure'].mean().reset_index(name='Avg_Tenure')
Avg_store_tenure =  emp.groupby('Store_Id')['Tenure'].mean().reset_index(name='Avg_Tenure')

plt.figure(figsize=(12,18))

# Attrition (Exit_Date not null) analysis:

emp['Attrition'] = emp['Exit_Date'].notna().astype(int)

# Which departments/stores have the highest turnover?

Attri_Depart = emp.groupby('Department')['Attrition'].mean().reset_index(name="Attrition %").sort_values(ascending=False, by='Attrition %')

Attri_Depart['Attrition %'] = Attri_Depart['Attrition %'] * 100

plt.figure(figsize=(14,12))
sns.lineplot(data = Attri_Depart, x = 'Department', y='Attrition %')

for x, y in zip(Attri_Depart['Department'], Attri_Depart['Attrition %']):
    plt.text(x, y+0.2, f"{y:.1f}%", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

Attri_store = emp.groupby('Store_Id')['Attrition'].mean().reset_index(name='Attrition %').sort_values(by='Attrition %', ascending=False)

plt.figure(figsize=(16,14))
sns.barplot(data=Attri_store, x='Store_Id', y='Attrition %')
for x, y in zip(Attri_store['Store_Id'], Attri_store['Attrition %']):
    plt.text(x, y+0.1, f"{y:.1f}%",  fontsize=10)
plt.tight_layout()
plt.xlabel('Store')
plt.ylabel('Attrition Rate')
plt.show()

# Does tenure length impact performance or attrition?

cmb_emp1=perform.merge(emp, on="Employee_Id", how="inner")
print(cmb_emp1[['Tenure', 'Performance_Rating','Attrition']].corr())

sns.scatterplot(cmb_emp1, x='Tenure', y='Performance_Rating', hue='Attrition')
plt.title("Tenure vs performance")
plt.tight_layout()
plt.show()

Tenure_Attri = cmb_emp1.groupby('Attrition')['Tenure'].mean().reset_index(name='Avg_tenure')

plt.figure(figsize=(14,12))
sns.boxplot(data=cmb_emp1, x='Attrition', y='Tenure')
plt.title('Tenure Distribution by Attrition Status')
plt.show()

# Step 6: Compensation & Rewards

# Correlation between base salary, bonuses, benefits and performance.

Corr_Benefit=cmb_emp1[['Base_Salary_Annual', 'Monthly_Bonus', 'Benefits_Cost', 'Performance_Rating']].corr()

plt.figure(figsize=(16,16))
sns.heatmap(Corr_Benefit, cmap='coolwarm', annot=True, fmt='.1f', linewidths=1, annot_kws={"size":25}, cbar_kws={'label': 'Correlation Strength'})
plt.title('Correlation between base salary, bonuses, benefits and performance')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# Are bonuses equally distributed across stores or roles?

Bonus_Distribution_Stores = cmb_emp1.groupby ('Store_Id')['Monthly_Bonus'].mean().reset_index(name='Avg_bonus').sort_values(by='Avg_bonus', ascending=False)

Bonus_Distribution_Job_Role = cmb_emp1.groupby('Job_Role')['Monthly_Bonus'].mean().reset_index(name='Avg_Bonus').sort_values(by='Avg_Bonus', ascending=False)

plt.figure(figsize=(20,20))
sns.barplot(data=Bonus_Distribution_Stores, x='Store_Id', y='Avg_bonus',palette='viridis')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,20))
sns.barplot(data=Bonus_Distribution_Job_Role, x='Job_Role', y='Avg_Bonus',palette='viridis')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Step 7: Employee Satisfaction & Engagement

# Compare satisfaction scores vs. performance ratings.

plt.figure(figsize=(16,16))
#sns.scatterplot(data=cmb_emp1, x='Employee_Satisfaction', y='Performance_Rating')
sns.lmplot(data=cmb_emp1, x='Employee_Satisfaction', y='Performance_Rating', aspect=1.5, palette='viridis')
plt.title('satisfaction scores vs. performance ratings')
plt.xlabel('Employee_Satisfaction')
plt.ylabel('Performance_Rating')
plt.tight_layout()
plt.show()

# Engagement vs. absenteeism: Do disengaged employees take more leave?

plt.figure(figsize=(16,16))
sns.regplot(data=cmb_emp1, x='Engagement_Index', y='Absenteeism_Days', scatter_kws={'alpha':0.4}, line_kws={'line':'red'})
plt.title('Engagement vs. absenteeism')
plt.xlabel('Engagement_Index')
plt.ylabel('Absenteeism_Days')
plt.tight_layout()
plt.show()  

corr1=cmb_emp1[['Engagement_Index','Absenteeism_Days']].corr()

Corr2=cmb_emp1['Engagement_Index'].corr(cmb_emp1['Absenteeism_Days'])


# Step 8 :Store & Regional Analysis

# Store-wise performance comparison (avg. rating, engagement, turnover).

Store_Avg_rating = Outcome.groupby('Store_Id')['Customer_Satisfaction'].mean().reset_index(name='Avg_rating')

Store_engagement = cmb_emp1.groupby('Store_Id')['Engagement_Index'].mean().reset_index(name='Avg_Engagement')

Store_turnover = Outcome.groupby('Store_Id')['Sales_Actual'].sum().reset_index(name='Total Turnover')

# Do Superstores vs. Regular stores show differences in employee performance?

cmb_emp2 = cmb_emp1.merge(stores, on='Store_Id', how='inner')

store_data = cmb_emp2.groupby('Store_Type')['Performance_Rating'].mean()
Store_pref = cmb_emp2.groupby('Store_Type').agg({
    'Performance_Rating':'mean',
    'Engagement_Index' : 'mean',
    'Employee_Satisfaction': 'mean',
    'Absenteeism_Days':'mean'}).reset_index()

# Geographic trends (map-based insights).

#cmb_emp3 = cmb_emp2.merge(Outcome, on='Store_Id', how='inner')

cmb_emp4 = Outcome.merge(stores, on='Store_Id', how='inner')

Geo_Data = cmb_emp4.groupby('City').agg({
    'Sales_Actual':'mean',
    'Customer_Satisfaction':'mean',
    'On_Time_Delivery':'mean'}).reset_index()

# plotly map

fig = px.scatter_mapbox(cmb_emp4,lat='City_Latitude', lon='City_Longitude', size='Sales_Actual', color='Store_Type',
                        hover_name='Store_Id', hover_data=['Sales_Actual', 'Sales_Target'], zoom=4,  mapbox_style="carto-positron")

pio.renderers.default='browser'

fig.show()

# Step 9 :Business Outcomes Alignment

# Link employee KPIs to business outcomes (sales, customer satisfaction if present).

emp_kpi = ['Store_Id', 'Performance_Rating', 'Employee_Satisfaction', 'Engagement_Index', 'Department', 'Job_Role']

outcome_kpi =['Store_Id', 'Sales_Actual', 'Sales_Target', 'Customer_Satisfaction']

cmb_kpi = cmb_emp2[emp_kpi].merge(Outcome[outcome_kpi], on='Store_Id', how='inner')

emp_kpi_op = cmb_kpi.groupby('Store_Id').agg({
    'Performance_Rating':'mean',
    'Employee_Satisfaction':'mean',
    'Engagement_Index':'mean',
    'Sales_Actual':'mean',
    'Sales_Target':'mean',
    'Customer_Satisfaction':'mean'}).reset_index()

# Identify which roles/departments contribute most to business success.


depart_kpi = cmb_kpi.groupby('Department').agg({
    'Performance_Rating':'mean',
    'Employee_Satisfaction':'mean',
    'Engagement_Index':'mean',
    'Sales_Actual':'mean',
    'Sales_Target':'mean',
    'Customer_Satisfaction':'mean'}).reset_index()

jobrole_kpi = cmb_kpi.groupby('Job_Role).agg({
    'Performance_Rating':'mean',
    'Employee_Satisfaction':'mean',
    'Engagement_Index':'mean',
    'Sales_Actual':'mean',
    'Sales_Target':'mean',
    'Customer_Satisfaction':'mean'}).reset_index()


# Step 9: Advanced/Portfolio-Boost Ideas

# Segmentation: Cluster employees based on performance, satisfaction, and engagement.



















