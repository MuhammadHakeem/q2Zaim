import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from yellowbrick.cluster import silhouette_visualizer
from sklearn.preprocessing import StandardScaler

st.title("Question 3")
st.subheader("Naive Bayes VS Decision Tree")

X = ["Naive Bayes", "Decision Tree"]
bfr_change = [50, 80]
aftr_change = [52, 81]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, bfr_change, 0.4, label = 'Naive Bayes')
plt.bar(X_axis + 0.2, aftr_change, 0.4, label = 'Decision Tree')

plt.xticks(X_axis, X)
plt.xlabel("Model")
plt.ylabel("Model")
plt.title("Accuracy Score")
plt.legend()
plt.show()
st.pyplot()

n_clusterChoice = st.selectbox("Select K Value", [2, 3, 4, 5])

#read dataset
df = pd.read_csv("Bank_CreditScoring.csv")
df2 = df.copy()

st.subheader("Clustering Analysis (K-Means)")
bckups_df = df2.copy()
X_CA = bckups_df.drop(["Decision"], axis=1)
Y_CA = bckups_df["Decision"]

X_CA = pd.get_dummies(X_CA, drop_first=True)
model_KM = KMeans(n_clusters=n_clusterChoice, init='random', n_init=30, random_state=1234, max_iter=400, tol=1e-09)
model_KM.fit(X_CA)
label=model_KM.fit_predict(X_CA)
X_CA['Label'] = label
st.write("X value is Monthly Salary")
st.write("Y value is Total Sum of Loan")

fig, axes = plt.subplots(1, 2, figsize=(13,6))

sns.relplot(x="Monthly_Salary", y="Total_Sum_of_Loan", hue="Decision", data = bckups_df, ax=axes[0])
sns.relplot(x="Monthly_Salary", y="Total_Sum_of_Loan", hue="Label", data = X_CA, ax=axes[1])

st.pyplot()

# st.write("Silhouette Score (n=2) = ", silhouette_score(X_CA, label))

conclusion = f"""
Silhouette Score (n=2) =  {silhouette_score(X_CA, label)} 
"""

st.markdown(conclusion)