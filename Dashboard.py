# Version 2 - Forcing a rebuildimport streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HR Attrition Dashboard",
    page_icon="🧑‍💼",
    layout="wide"
)

# --- FUNCTION TO LOAD CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {file_name}")

# --- APPLY CUSTOM STYLING ---
local_css("style.css")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df.drop(['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18'], axis=1, inplace=True)
    return df

df = load_data()

# --- MAIN TITLE ---
st.title("🧑‍💼 Executive HR Attrition Analysis")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")
department = st.sidebar.multiselect("Select Department:", options=df["Department"].unique(), default=df["Department"].unique())
job_role = st.sidebar.multiselect("Select Job Role:", options=df["JobRole"].unique(), default=df["JobRole"].unique())
gender = st.sidebar.multiselect("Select Gender:", options=df["Gender"].unique(), default=df["Gender"].unique())

df_selection = df.query("Department == @department & JobRole == @job_role & Gender == @gender")

if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop()

# --- KEY METRICS ---
st.header("Key Metrics Overview")
total_employees = df_selection.shape[0]
attrition_count = df_selection[df_selection["Attrition"] == "Yes"].shape[0]
attrition_rate = round((attrition_count / total_employees) * 100, 1) if total_employees > 0 else 0
avg_monthly_income = round(df_selection["MonthlyIncome"].mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", f"{total_employees:,}")
col2.metric("Attrition Count", f"{attrition_count:,}")
col3.metric("Attrition Rate", f"{attrition_rate}%")
col4.metric("Avg. Monthly Income", f"${avg_monthly_income:,.0f}")

st.markdown("---")

# --- PIE CHART ROW ---
st.header("Attrition Proportions by Demographics")
col1, col2 = st.columns(2)
df_attrition_only = df_selection[df_selection['Attrition'] == 'Yes']

with col1:
    st.subheader("Gender Distribution of Leavers")
    fig_pie_gender = px.pie(
        df_attrition_only, names='Gender', title='<b>Gender of Employees Who Left</b>',
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie_gender.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black')
    st.plotly_chart(fig_pie_gender, use_container_width=True)

with col2:
    st.subheader("Overtime Status of Leavers")
    fig_pie_overtime = px.pie(
        df_attrition_only, names='OverTime', title='<b>Overtime Status of Employees Who Left</b>',
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_pie_overtime.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black')
    st.plotly_chart(fig_pie_overtime, use_container_width=True)

st.markdown("---")


# --- CHART ROW: KEY DRIVERS & HIERARCHY ---
st.header("Key Drivers & Hierarchical Breakdown")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Predictive Factors of Attrition")
    df_model = df_selection.copy()
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])
    X = df_model.drop('Attrition', axis=1)
    y = df_model['Attrition']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(10)
    fig_importance = px.bar(importances, x='importance', y='feature', orientation='h', title='<b>Top 10 Most Important Features</b>', text='importance', color='importance', color_continuous_scale=px.colors.sequential.Plasma_r)
    fig_importance.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black', yaxis={'categoryorder': 'total ascending'})
    fig_importance.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_importance, use_container_width=True)

with col2:
    st.subheader("Attrition Count by Department & Job Role")
    fig_treemap = px.treemap(df_attrition_only, path=[px.Constant("All Employees"), 'Department', 'JobRole'], title='<b>Treemap of Employees Who Left</b>', color_discrete_sequence=px.colors.qualitative.T10)
    fig_treemap.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black')
    fig_treemap.update_traces(root_color="lightgrey")
    st.plotly_chart(fig_treemap, use_container_width=True)


st.markdown("---")

# --- CHART ROW: COMPENSATION & DEMOGRAPHICS ---
st.header("Compensation and Demographic Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Does Income Make a Difference?")
    fig_violin_income = px.violin(df_selection, y='MonthlyIncome', x='Attrition', box=True, title='<b>Income Distribution: Leavers vs. Stayers</b>', color='Attrition', color_discrete_map={'Yes': '#e63946', 'No': '#457b9d'})
    fig_violin_income.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black')
    st.plotly_chart(fig_violin_income, use_container_width=True)

with col2:
    st.subheader("How Does Overtime Affect Different Ages?")
    fig_box_age = px.box(df_selection, x='OverTime', y='Age', color='Attrition', title='<b>Age Distribution by Overtime & Attrition</b>', color_discrete_map={'Yes': '#e63946', 'No': '#457b9d'})
    fig_box_age.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='black')
    st.plotly_chart(fig_box_age, use_container_width=True)


