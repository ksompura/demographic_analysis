# Demographic data analysis from 1994 Census data

#import libraries
from locale import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read in the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, skipinitialspace=True)

data.head()
data.shape
data.info
col_names = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','salary']

data.columns = col_names

# 1. How many people of each race are represented in this dataset? This should be a Pandas series with race names as the index labels. 
df = data

df_race = df.groupby(['race'])['race'].count()
df_race = df_race.sort_values(ascending=False)
print(df_race)
## We can see that race: 'white' is the large majority group being represented in this dataset, followed by 'Black', and 'Asian-Pac-Islander'.

# plot race representation
df_race.plot.bar()
sns.set_style('darkgrid')
#plt.xlabel('Race')
#plt.ylabel('Count of Individuals')
#plt.title('Race Representation')
#plt.legend()
plt.show()



# 2. What is the average age of men?

# make a mask to only select men
mask = df['sex'] == 'Male'
df_male = df[mask]
df_male['age'].mean()
## average male age is 39.434 years old.
# consider graphing age distribution


# 3. What is the percentage of people who have a Bachelor's degree?
# find number with bachelor/ total people
# groupby education
df_edu = df.education.value_counts()

# calculate percentage of people with bachelors degree
df.education.value_counts(normalize=True) * 100
## 16.44% of the people in the dataset have a Bachelor's degree.
# consider visualizing the education levels represented


# 4. What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50k?
# create datafram with only Bachelors, Masters, and Doctorate
mask_adv = (df['education'] == 'Bachelors') | (df['education'] == 'Masters') | (df['education'] == 'Doctorate')
df_adv = df[mask_adv]

df_adv.groupby(['education', 'salary'])['salary'].count()

df_adv.salary.value_counts(normalize=True) * 100
# Only 46.54% of people with advanced education (Bachelors, Masters, or Doctorate) made more than $50,000 at the time of the survey.
# That is also to say 53.46% of people with advance education made less than $50,000.

# consider graphing salary with age, sex, race, or home country to see if there is anything significant

# 5. What percentage of people without advanced education make more than 50K?

# invert the mask made for the advanced education in question 4
mask_und = ~mask_adv

df_und = df[mask_und]

df_und.groupby(['education', 'salary'])['salary'].count()
df_und.salary.value_counts(normalize=True) * 100

# Only 17.37% of people without advanced education (Bachelors, Masters, or Doctorate) made more than $50,000 at the time of the survey.
# # That is also to say 82.63% of people without advance education made less than $50,000.


# 6. What is the minimum number of hours a person works per week?
# find the min value 
df['hours_per_week'].min()

# the minimum number of hours a person worked per week was 1 hour.

# consider graphing a histogram of the hours per week to see how many hours people work


# 7. What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?
# create a mask to filter only people who work the minimum number of hours per week
mask_min = df['hours_per_week'] == df.hours_per_week.min()
df_min = df[mask_min]
df2_min = df_min.groupby('salary')['salary'].count()
df2_min['percentage'] = (df2_min / df2_min.sum()) * 100
print(df2_min)

# 10% of the people working the minimum number of hours per week are making a salary of more than $50,000.


# 8. What country has the highest percentage of people that earn >50K and what is that percentage?
df_country = df.value_counts(subset=['native_country','salary'], normalize=True) * 100
df2_country = df
print(df_country.to_string())


df_country = df.groupby(['native_country','salary'], as_index = False).count()

country = df_country[['native_country','salary','age']]
country.rename(columns={'age':'count'}, inplace=True)


country["percentage"] = 100 * country['count']/ country.groupby('native_country')['count'].transform('sum')
country = country.sort_values('percentage', ascending=False)

print(country.to_string())
# The country with the highest percentage of people that earn greater than $50,000 is Iran with 41.86%.

