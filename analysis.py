
# coding: utf-8

# # Does faculty salary vary by gender?

# ## Set up
# 
# Before getting started, the only addtional library you should have to install (that did not come with the anaconda python distribution) is `seaborn`, a package for visualization:
# 
# ```
# pip install seaborn
# ```
# 
# Let's begin by reading in some data from [this course website](http://data.princeton.edu/wws509/datasets/#salary). Columns included are:
# 
# - **sx** = Sex, coded 1 for female and 0 for male
# - **rk** = Rank, coded
#     - 1 for assistant professor,
#     - 2 for associate professor, and
#     - 3 for full professor
# - **yr** = Number of years in current rank
# - **dg** = Highest degree, coded 1 if doctorate, 0 if masters
# - **yd** = Number of years since highest degree was earned
# - **sl** = Academic year salary, in dollars.

# In[248]:

# Set up
import numpy as np
import pandas as pd
import seaborn as sns # for visualiation
import urllib2 # to load data
from scipy.stats import ttest_ind # t-tests
import statsmodels.formula.api as smf # linear modeling
import matplotlib.pyplot as plt # plotting
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().magic(u'matplotlib inline')


# In[249]:

# Read data from URL
file = urllib2.urlopen('http://data.princeton.edu/wws509/datasets/salary.dat')
headers = file.next()
df = pd.DataFrame(l.rstrip().split() for l in file)
df.columns = headers.rstrip().split()
df['sl'] = df['sl'].astype(float) # Make sure salary is float
df['yr'] = df['yr'].astype(int) # Make sure year is int is float
df['yd'] = df['yd'].astype(int) # Make sure salary is float


# ## Descriptive statistics

# Here, you should explore dimensions of your dataframe -- compute measure of interest such as column means, correlations, numbers of observations by group, etc.

# In[250]:

# Number of males/females in the dataset
df.groupby('sx').size()


# In[251]:

# Salary by sex
df[['sx', 'sl']].groupby('sx').agg('mean')


# # Test for a difference in means by gender
# Use a t-test to see if there is a significant difference in means

# In[252]:

# Separate into different arrays by sex
males = df[df['sx'] == 'male'] 
females = df[df['sx'] == 'female']


# In[253]:

# Test for difference
ttest_ind(males[['sl']], females[['sl']]) # not significant!


# ## Difference in means by rank (full v.s. not full)

# In[254]:

# Separate into different arrays by sex
full = df[df['rk'] == 'full']
not_full = df[df['rk'] != 'full']

# Test for difference
ttest_ind(full[['sl']], not_full[['sl']]) # significant!
print(np.mean(not_full['sl']), np.mean(full['sl']))


# In[297]:

# Use the ANOVA method to test for differences in means across multiple groups
from scipy import stats
stats.f_oneway(df[df.rk == 'full'].sl, df[df.rk == 'associate'].sl, df[df.rk == 'assistant'].sl)


# ## Explore salary distributions by sex, rank

# In[255]:

# Histograms of each distribution
import matplotlib.pyplot as plt
min = np.min(df['sl'])
max = np.max(df['sl'])
plt.figure(figsize=(10,5))
df['sl'].hist(by=df['sx'], range=[min, max])
df['sl'].hist(by=df['rk'], range=[min, max])
plt.show()


# In[256]:

# View distributions in a boxplot
df[['sx', 'sl']].boxplot(by='sx')


# In[257]:

# Do number of years vary by gender?
df[['sx', 'yr', 'yd']].boxplot(by='sx')


# ## Explore bivariate relationships visually
# 

# Number of years since degree versus salary

# In[258]:

df.plot('yd', 'sl', kind="scatter")


# Number of years in current position versus salary

# In[263]:

# Show relationship b/w each variable and the ourcome
sns.stripplot(x="sx", y="sl", data=df, jitter=True);


# In[259]:

df.plot('yr', 'sl', kind="scatter")


# In[264]:

# Categorical variables
g = sns.PairGrid(df,
                 x_vars=["sx",'rk'],
                 y_vars=["sl"],
                 aspect=.75, size=7)
g.map(sns.stripplot, palette="pastel");


# In[267]:

# Salary by continuous variables (yd, yr)
fig, axs = plt.subplots(1, 2, sharey=True)
df.plot(kind='scatter', x='yd', y='sl', ax=axs[0], figsize=(16, 8))
df.plot(kind='scatter', x='yr', y='sl', ax=axs[1])


# In[265]:

# Sex by rank!
sns.factorplot(x="sx", y="sl", 
               col="rk", data=df, kind="strip", jitter=True);


# ## Simple linear regression: what is the expected salary increase for each additional year in your current position (`yr`)

# In[260]:

# create a fitted model in one line
lm = smf.ols(formula='sl ~ yd', data=df).fit()
lm.summary()


# In[261]:

# Make predictions using the linear model
df['predictions'] = lm.predict()


# In[262]:

# How well does our line fit our data?
plt.scatter(df.yd, df.sl)
plt.plot(df.yd, df.predictions)
plt.show()


# ## Multiple Regression

# Predict using **multiple** independent variables

# In[266]:

# Just to check: yd and yr are correlated
print(df[['yd', 'yr']].corr())
df.plot(kind='scatter', x='yd', y='yr')


# In[270]:

lm_mult = smf.ols(formula='sl ~ yd + sx + rk + yr + dg', data=df).fit()
lm_mult.summary()


# In[271]:

df['mult_preds'] = lm_mult.predict()


# In[272]:

# How do our predictions compare
plt.scatter(df.predictions, df.mult_preds)


# In[273]:

# How do our predictions perform (over years since degree)
plt.scatter(df.yd, df.sl)
plt.scatter(df.yd, df.mult_preds, color='red')
plt.show()


# In[274]:

# Compare predictions to observations
plt.scatter(df.mult_preds, df.sl)
plt.plot(df.sl, df.sl)
plt.show()


# In[275]:

# What are the r-squared values of the models?
print(lm.rsquared, lm_mult.rsquared) # explain the same amount of variance?


# In[276]:

# Let's add a model that's just sex and years since graduation
lm_mult_sex_only = smf.ols(formula='sl ~ yd + C(sx)', data=df).fit()
lm_mult_sex_only.summary()


# In[277]:

print(lm.rsquared, lm_mult_sex_only.rsquared)


# In[278]:

lm_mult_sex_only.summary()


# In[283]:

# Plot the residuals with a center line
df['lm_resids'] = df.sl - df.predictions
df['lm_mult_resids'] = df.sl - df.mult_preds
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 6))
ax1.scatter(df.sl, df.lm_resids)
ax1.plot((np.min(df.sl), np.max(df.sl)), (0,0),color='darkblue', lw=2)
ax2.scatter(df.sl, df.lm_mult_resids)
ax2.plot((np.min(df.sl), np.max(df.sl)), (0,0),color='darkblue', lw=2)
plt.show()


# As shown above, this model clearly **over predicts** low salaries and **under predicts** higher salaries. Clearly, we're missing something that **explains variation** besides the number of years.

# In[ ]:



