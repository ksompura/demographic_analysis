# Demographic data analysis from 1994 Census data

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url)

data.head()
data.shape
data.info
col_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']

data.columns = col_names

# How many people of each race are represented in this dataset? This should be a Pandas series with race names as the index labels. 
df = data

df_race = df.groupby(['race'])['race'].count()
df_race.sort_values(ascending=False)
## We can see that race: 'white' is the large majority group being represented in this dataset, followed by 'Black', and 'Asian-Pac-Islander'.

# plot race representation




