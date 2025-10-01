# -Employee-Performance-Analysis---Python
This project analyzes workforce performance data to uncover insights into demographics, retention, satisfaction, compensation, and business outcomes. It was completed as part of my Python portfolio projects series.
📌 Project Overview

Employee performance is a key driver of organizational success. In this project, I analyzed a multi-table HR dataset to answer critical questions:

What factors drive employee performance?

How do demographics, tenure, and engagement impact retention?

How are compensation, satisfaction, and performance connected?

Which departments and roles contribute most to business outcomes?

🗂️ Dataset Description

The dataset contains multiple related tables:

employees – employee demographic & job details

stores – store information (location, type, opening date)

monthly_performance – monthly employee performance metrics

role_kpis – role-specific key performance indicators

business_outcomes – store-level business results (sales, satisfaction, delivery)

🔑 Key Analysis Steps
1. Data Cleaning & Preparation

Converted date fields (Hire, Exit, Opening)

Checked for duplicates & missing values

Ensured key integrity (Employee_Id, Store_Id)

Created new features: Tenure (years), Age Groups, Performance Categories

2. Workforce Demographics

Age distribution by department & job level

Education level vs. job roles

Department-wise headcount

3. Performance Analysis

Avg performance rating by department, role, job level

Effect of training hours, overtime, absenteeism

Manager evaluation vs. employee performance

Classified employees into Top, Average, Low performers

4. Tenure & Retention

Avg tenure by department & store

Attrition analysis by department & region

Relationship between tenure, attrition & performance

5. Compensation & Rewards

Correlation between salary, bonuses, benefits & performance

Bonus distribution across stores & roles

6. Satisfaction & Engagement

Employee satisfaction vs. performance

Engagement vs. absenteeism

7. Store & Regional Analysis

Performance comparison: Superstores vs. Regular stores

Regional/geographic trends (using Plotly Mapbox)

8. Business Outcomes Alignment

Linked employee KPIs to business outcomes (sales, satisfaction, delivery)

Identified high-impact departments & job roles

📊 Tools & Libraries

Python: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

EDA & Analysis: Correlation, Heatmaps, Regression plots, Geospatial visualization

📈 Key Insights

Engagement & Satisfaction strongly correlate with performance and lower absenteeism.

Tenure impacts attrition risk — employees with lower tenure are more likely to leave.

Compensation & Rewards (bonuses, benefits) vary significantly by role & store, influencing satisfaction.

Business success is closely linked to employee engagement and performance.
