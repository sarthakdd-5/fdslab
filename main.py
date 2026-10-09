import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    'Patient_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Age': [45, 54, 63, np.nan, 52, 70, 39, 48, np.nan, 61],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Cholesterol': [230, 280, 210, 250, np.nan, 310, 190, 240, 290, 220],
    'Max_HR': [150, 142, 125, 168, 155, 110, 175, 160, 135, 130],
    'Target_Risk': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    'Department': ['Cardiology', 'Cardiology', 'ER', 'Cardiology', 'ER', 'Cardiology', 'ER', 'Cardiology', 'ER', 'Cardiology']
}
df = pd.DataFrame(data)

print("=== INFO ===")
df.info()
print("\n=== DESCRIPTION ===")
print(df.describe())

print("\n=== MISSING VALUES COUNT ===")
print(df.isnull().sum())

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)

top_cholesterol = df[df['Age'] > 45].sort_values(by='Cholesterol', ascending=False).head(5)

freq_gender = df['Gender'].value_counts()
freq_risk = df['Target_Risk'].value_counts()

explicit_slice = df.loc[0:2, ['Patient_ID', 'Age', 'Cholesterol']]
implicit_slice = df.iloc[0:3, [0, 1, 3]]

cond1 = df[(df['Age'] > 50) & (df['Target_Risk'] == 1)][['Patient_ID', 'Age', 'Max_HR']]
cond2 = df[(df['Department'] == 'Cardiology') & (df['Cholesterol'] > 230)][['Patient_ID', 'Cholesterol']]
cond3 = df[(df['Gender'] == 'Male') | (df['Max_HR'] < 130)][['Patient_ID', 'Gender', 'Max_HR']]

max_chol = df['Cholesterol'].max()
min_hr = df['Max_HR'].min()

gb1 = df.groupby('Target_Risk')['Cholesterol'].mean()
gb2 = df.groupby(['Department', 'Gender'])['Max_HR'].mean()

df['HR_Reserve'] = 220 - df['Age'] - df['Max_HR']

agg1 = df.groupby('Target_Risk').agg({'Age': ['mean', 'min', 'max'], 'Cholesterol': 'mean'})
agg2 = df.groupby('Department').agg({'Max_HR': ['mean', 'std'], 'HR_Reserve': 'sum'})

cardio_group = df.groupby('Department').get_group('Cardiology')

corr_age_hr = df['Age'].corr(df['Max_HR'])

df['Cholesterol_Norm'] = (df['Cholesterol'] - df['Cholesterol'].min()) / (df['Cholesterol'].max() - df['Cholesterol'].min())

vitals_df = pd.DataFrame({'Patient_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110], 'BP': [120, 135, 140, 128, 130, 150, 118, 122, 145, 138]})
merged_df = pd.merge(df, vitals_df, on='Patient_ID')