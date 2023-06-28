#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ##### This project aims to explore the relationship between minimum wage, happiness, and social support in 31 countries. The primary focus is to analyze whether countries with relatively low minimum wages but high happiness scores also report higher levels of social support compared to countries with high happiness and high minimum wages, as well as the relationship between social support and happiness and the relationship between minimum wage, social support and happiness scores for the data as a whole. Additionally, the project investigates the connection between minimum wage and government corruption.
# 
# ##### Minimum wage, representing the lowest legally mandated remuneration for workers, has been associated with happiness and well-being globally. By examining the interplay between minimum wage, happiness scores, and social support, this project sheds light on the factors influencing happiness and well-being within different socioeconomic contexts. Moreover, it delves into the relationship between minimum wage and government corruption.
# 
# 
# ### Questions:
# 
# ##### * What is the relationship between the average social support score for countries with relatively low minimum wage but high happiness scores as compared to the average social support score for countries with relatively high minimum wage and high happiness scores as well as to the average social support score for all countries?
# 
# ##### * What is the relationship between average social support scores and average happiness scores? 
# 
# ##### * What is the relationship between social support, happiness scores and minimum wage? 
#  
# ##### * What is the relationship between minimum wage and government corruption.  Do countries with a relatively low minimum wage correlate significantly with higher levels of government corruption?
# 

# # Methodology

# ##### Data was taken from the website https://www.kaggle.com accessed on 3/7/2022
# 
# ##### 1. https://www.kaggle.com/datasets/frtgnn/minimum-wages-between-2001-2018
# 
# ##### CSV name: MINIMUM_WAGES.csv
# ##### The dataset provided by Firat Gonen gives the minimum wage of 31 countries. The minimum wage is yearly in US Dollars. The countries appear to have been selected randomly.
# 
# ##### 2. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2015.csv
# 
# ##### CSV name: 2015.csv, 2016.csv, 2017.csv, 2018.csv
# ##### The datasets provided by Sustainable Development Solutions Network gives the happiness score for 155-158 countries. The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll.  The dataset for the year 2019 was not included since the MINIMUM_WAGES dataset only goes up to the year 2018.
# 
# 

# # Results

# ### 1. Import Libraries

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats


# ### 2. Import Datasets 

# In[3]:


# Use pandas to read the csv files.

mini = pd.read_csv('data/MINIMUM_WAGES.csv')
hap15 = pd.read_csv('data/2015.csv')
hap16 = pd.read_csv('data/2016.csv')
hap17 = pd.read_csv('data/2017.csv')
hap18 = pd.read_csv('data/2018.csv')


# ### 3. Clean the data

# ##### Strip the string data of whitespace and rename columns and rows in the datasets that use different column names or text ids for the same values between the respective datasets to avoid generating any missing info once the merge function has been deployed. Columns for the years 2001-2014 in the MINIMUM_WAGE dataset have been dropped since those years are not present in the happiness datasets. Columns that are not mutually present across all respective happiness datasets are dropped. 

# In[4]:


# Clean the data.

mini = mini.applymap(lambda x: x.strip() if isinstance (x,str) else x)
hap15 = hap15.applymap(lambda x: x.strip() if isinstance (x, str) else x)
hap16 = hap16.applymap(lambda x: x.strip() if isinstance (x, str) else x)
hap17 = hap17.applymap(lambda x: x.strip() if isinstance (x, str) else x)
hap18 = hap18.applymap(lambda x: x.strip() if isinstance (x, str) else x)

mini['Country'] = mini['Country'].replace(['Korea', 'Slovak Republic', 'Russian Federation'], 
                                        ['South Korea', 'Slovakia', 'Russia'])
mini = mini.drop(['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
                 '2009', '2010', '2011', '2012', '2013', '2014'], axis=1)

hap15 = hap15.drop(['Standard Error', 'Region', 'Happiness Rank', 'Dystopia Residual'], axis=1)
hap15 = hap15.rename(columns={'Family':'Social support','Health (Life Expectancy)':'Healthy life expectancy',
                             'Trust (Government Corruption)':'Government Corruption'})

hap16 = hap16.drop(['Lower Confidence Interval','Upper Confidence Interval', 'Region',
                    'Happiness Rank', 'Dystopia Residual'],axis=1)
hap16 = hap16.rename(columns={'Family':'Social support', 'Health (Life Expectancy)':'Healthy life expectancy', 
                             'Trust (Government Corruption)':'Government Corruption'})

hap17 = hap17.drop(['Whisker.high', 'Whisker.low', 'Happiness.Rank', 'Dystopia.Residual'], axis=1)
hap17 = hap17.rename(columns={'Happiness.Score':'Happiness Score', 
                              'Economy..GDP.per.Capita.':'Economy (GDP per Capita)','Family':'Social support', 
                              'Health..Life.Expectancy.':'Healthy life expectancy',
                              'Trust..Government.Corruption.':'Government Corruption'})
hap17.insert(7, 'Generosity', hap17.pop('Generosity'))

hap18 = hap18.drop(['Overall rank'], axis=1)
hap18.insert(7, 'Generosity', hap18.pop('Generosity'))
hap18 = hap18.rename(columns={'Country or region':'Country', 'Score':'Happiness Score',
                             'GDP per capita':'Economy (GDP per Capita)', 'Freedom to make life choices':
                             'Freedom', 'Perceptions of corruption':'Government Corruption'})




# In[5]:


# Check a sample of the output and headings for minimum wage data.

mini.head()


# In[6]:


# Check data types, etc...
mini.info()


# In[7]:


# Check headings for happiness score datasets.
hap15.head()


# In[8]:


hap15.info()


# In[9]:


hap16.head()


# In[10]:


hap16.info()


# In[11]:


hap17.head()


# In[12]:


hap17.info()


# In[13]:


hap18.head()


# In[14]:


hap18.info()


# ##### There is a null value present in the 'Trust (Government Corruption)' column.

# In[15]:


# Remove null value by replacing it with a zero.
hap18['Government Corruption'] = hap18['Government Corruption'].fillna(0)
hap18.info()


# ##### The null value has been removed.

# ### 4. Make a Dataframe

# ##### Create a dataframe by merging the minimum wage dataset with the happiness datasets using countries that are present in all the respective data sets. 

# In[16]:


# Create data frame.

minihappy = pd.merge(mini, hap15, left_on='Country', right_on='Country', how='inner', suffixes=('', '_2015'))
minihappy = pd.merge(minihappy, hap16, left_on='Country', right_on='Country', how='inner', suffixes=('', '_2016'))
minihappy = pd.merge(minihappy, hap17, left_on='Country', right_on='Country', how='inner', suffixes=('', '_2017'))
minihappy = pd.merge(minihappy, hap18, left_on='Country', right_on='Country', how='inner', suffixes=('', '_2018'))

minihappy = minihappy.rename(columns={'Happiness Score':'Happiness Score_2015',
                                      'Economy (GDP per Capita)':'Economy (GDP per Capita)_2015',
                                      'Social support':'Social support_2015',
                                      'Healthy life expectancy':'Healthy life expectancy_2015',
                                      'Freedom':'Freedom_2015',
                                      'Government Corruption':'Government Corruption_2015',
                                      'Generosity':'Generosity_2015',
                                      })

# Display the maximum columns
pd.set_option('display.max_columns', None)

# Print data frame
minihappy.head()



# In[17]:


# Rename hap15 columns.  


minihappy = minihappy.rename(columns={'Happiness Score':'Happiness Score_2015',
                                      'Economy (GDP per Capita)':'Economy (GDP per Capita)_2015',
                                      'Social support':'Social support_2015',
                                      'Healthy life expectancy':'Healthy life expectancy_2015',
                                      'Freedom':'Freedom_2015',
                                      'Trust (Government Corruption)':'Trust (Government Corruption)_2015',
                                      'Generosity':'Generosity_2015',
                                      })

# Print data frame
minihappy.head()


# In[18]:


# Check data types, etc...
minihappy.info()


# In[19]:


# Check for null values
minihappy.isna().sum()


# ##### There are no null values

# ### 5. Compare countries with relatively low minimum wage, but high happiness scores to countries with high minimum wage and high happiness scores on the life evaluation variable of social support.

# In[20]:


# Write a function that creates a data frame that filters for countries with relatively 
# low minimum wage but high happiness scores.

def low_minimum_wage_high_happiness(minihappy):
    # Create a new dataframe with the Minimum Wage and Happiness Score columns for each year.
    minhap2 = pd.DataFrame(data=minihappy[['Country', '2015','2016','2017','2018',
                                'Happiness Score_2015', 'Happiness Score_2016',
                                'Happiness Score_2017', 'Happiness Score_2018']])
    
    # Calculate the average minimum wage and happiness scores for each country during 2015-2018
    minhap2['Average Minimum Wage 2015-2018'] = minhap2[['2015','2016','2017','2018']].mean(axis=1)
    minhap2['Average Happiness Score 2015-2018'] = minhap2[['Happiness Score_2015', 'Happiness Score_2016',
                                'Happiness Score_2017', 'Happiness Score_2018']].mean(axis=1)
    
    # Filter for countries with relatively low minimum wage and high happiness scores.
    low_wage_high_happiness = minhap2[(minhap2['Average Minimum Wage 2015-2018'] < 
                                       minhap2['Average Minimum Wage 2015-2018'].median()) & 
                                      (minhap2['Average Happiness Score 2015-2018'] > 
                                       minhap2['Average Happiness Score 2015-2018'].median())] 
    # Sort the result by happiness score in descending order.
    low_wage_high_happiness = low_wage_high_happiness.sort_values(by='Average Happiness Score 2015-2018',ascending=False)
    
    return low_wage_high_happiness

low_happy = low_minimum_wage_high_happiness(minihappy)
low_happy


# ##### Create a new data frame to compare the average social support scores for the five countries in the low_happy data frame with the average social support scores for other countries during the years 2015-2018.

# In[21]:


# Create a data frame for the average social support scores for the countries in the low_happy data frame.
social_support = minihappy.iloc[[27,16,28,3,4], [0,7, 14, 21, 28]]
social_support['Average Social Support 2015-2018'] = social_support.mean(axis=1)
social_support


# In[22]:


social_support.describe()


# In[23]:


# Find the average social support score for countries with low minimum wages but high happiness scores.
low_happy_ss_avg = social_support['Average Social Support 2015-2018'].mean()
low_happy_ss_avg


# ##### Make a dataframe for countries with relativley high minimum wage and high happiness scores.

# In[24]:


# Write a function that creates a data frame that filters for countries with relatively 
# high minimum wage and high happiness scores.

def high_minimum_wage_high_happiness(minihappy):
    # Create a new dataframe with the Minimum Wage and Happiness Score columns for each year.
    minhap3 = pd.DataFrame(data=minihappy[['Country', '2015','2016','2017','2018',
                                'Happiness Score_2015', 'Happiness Score_2016',
                                'Happiness Score_2017', 'Happiness Score_2018']])
    
    # Calculate the average minimum wage and happiness scores for each country during 2015-2018
    minhap3['Average Minimum Wage 2015-2018'] = minhap3[['2015','2016','2017','2018']].mean(axis=1)
    minhap3['Average Happiness Score 2015-2018'] = minhap3[['Happiness Score_2015', 'Happiness Score_2016',
                                'Happiness Score_2017', 'Happiness Score_2018']].mean(axis=1)
    
    # Filter for countries with relatively high minimum wage and high happiness scores.
    high_wage_high_happiness = minhap3[(minhap3['Average Minimum Wage 2015-2018'] > 
                                       minhap3['Average Minimum Wage 2015-2018'].median()) & 
                                      (minhap3['Average Happiness Score 2015-2018'] > 
                                       minhap3['Average Happiness Score 2015-2018'].median())] 
    # Sort the result by happiness score in descending order.
    high_wage_high_happiness = high_wage_high_happiness.sort_values(by='Average Happiness Score 2015-2018',ascending=False)
    
    return high_wage_high_happiness

high_happy = high_minimum_wage_high_happiness(minihappy)
high_happy


# In[25]:


# Create a data frame for the average social support scores for the countries in the high_happy data frame.
social_support2 = minihappy.iloc[[17,2,18,0,10,26,9,1,15,25], [0,7,14,21,28]]
social_support2['Average Social Support 2015-2018'] = social_support2.mean(axis=1)
social_support2


# In[26]:


social_support2.describe()


# In[27]:


# Find the average social support score for countries with high happiness and high minimum wage.
high_happy_ss_avg = social_support2['Average Social Support 2015-2018'].mean()
high_happy_ss_avg


# ##### Find the average social support score for all countries .

# In[28]:


# Create a data frame that returns the Average social support scores for each country. 
minihappy_ss = minihappy.iloc[:,[0, 7, 14, 21, 28]]
minihappy_ss['Average Social Support 2015-2018'] = minihappy_ss.mean(axis=1)
minihappy_ss


# In[29]:


# Find the average social support score for all countries.
minihappy_ss_avg = minihappy_ss['Average Social Support 2015-2018'].mean()
minihappy_ss_avg


# In[30]:


# Data
categories = ['Low Minimum Wage &\nHigh Happiness', 'High Minimum Wage &\nHigh Happiness', 'All Countries']
averages = [low_happy_ss_avg, high_happy_ss_avg, minihappy_ss_avg]

# Create bar chart
plt.bar(categories, averages)

# Add labels and title
plt.xlabel('')
plt.ylabel('Average Social Support Score')
plt.title('Average Social Support Score Comparison')

# Rotate x-axis labels at a diagonal angle and align them to the right
plt.xticks(rotation=45, ha='right')

# Adjust the y-axis position for stacked labels
plt.subplots_adjust(bottom=0.2)

# Display the chart
plt.show()


# ### Notes on Findings

# ##### *The countries with low minimum wage had a average social support score of 1.20934.
# ##### *The countries with high minimum wage and high happiness had a average social support score of 1.3408.
# ##### *The average social support score for all countries was 1.273

# ###  6. Compare average social support scores with average happiness scores for all countries.

# In[31]:


# Create a data frame for average social support scores and average happiness scores for all countries. 
# Begin by creating a series using the minihappy_ss data frame to return the average social support for each country.
minihappy_ss = minihappy_ss.set_index('Country')
minihappy_ss2 = minihappy_ss.loc[:,['Average Social Support 2015-2018']]

# Sort values by countries with highest social support to lowest.
minihappy_ss2 = minihappy_ss2.sort_values(by='Average Social Support 2015-2018', ascending=False)


minihappy_ss2.head()


# ##### Find the average happiness score for each country

# In[32]:


# Create a data frame of the happiness scores for countries during
# the years 2015-2018



happiness_score = minihappy.loc[:,['Country', 'Happiness Score_2015', 'Happiness Score_2016', 'Happiness Score_2017',
                                  'Happiness Score_2018']]


# Find the average happiness score for each country during the years 2015-2018

happiness_score['Average Happiness Score 2015-2018'] = happiness_score.mean(axis=1)

# Create a series using the average happiness scores for each contry during the year 2015-2018

happiness_score = happiness_score.set_index('Country')
happiness_score = happiness_score.loc[:,['Average Happiness Score 2015-2018']]

# Sort values by countries with highest happiness score to lowest
happiness_score = happiness_score.sort_values('Average Happiness Score 2015-2018', ascending=False)

happiness_score.head()


# In[33]:


# Concatenate minihappy_ss2 series with happiness_score series
social_happy = pd.concat([minihappy_ss2, happiness_score], axis=1)
social_happy


# In[34]:


# reset index to facilitate creation of charts and plots
social_happy = social_happy.reset_index()
social_happy.head()


# ### 7. Graph 3: Scatterplot with Regression Line for Average Social Support and Happiness Scores 2015-2018

# In[35]:


sns.regplot(x='Average Social Support 2015-2018', y='Average Happiness Score 2015-2018', data=social_happy)

#Set the x and y axis labels.
plt.xlabel('Average Social Support 2015-2018')
plt.ylabel('Average Happiness Score 2015-2018')

#Set the title of the plot.
plt.title('Average Social Support and Happiness Score 2015-2018')

#Calculate the regression equation, correlation coefficient, p-value, and standard error.
slope, intercept, r_value, p_value, std_err = stats.linregress(social_happy['Average Social Support 2015-2018'],
social_happy['Average Happiness Score 2015-2018'])
correlation_coefficient = np.corrcoef(social_happy['Average Social Support 2015-2018'], social_happy['Average Happiness Score 2015-2018'])[0,1]

#Display the regression equation, correlation coefficient, p-value, and standard error.
#Set the slope, intercept, correlation coefficient, and p_value to .3f to round the numbers to three decimal places.
print(f"Regression Equation: y = {slope:.3f}x + {intercept:.3f}")
print(f"Correlation Coefficient: {correlation_coefficient:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Standard Error: {std_err:.3f}")

#Display the plot.
plt.show()


# ### Notes on Findings

# ##### * There is a positive correlation between average social support and average happiness scores. As the average social support increases, the average happiness scores also tend to increase. This is indicated by the positive slope of the regression line.
# 
# ##### *The correlation coefficient of 0.444 suggests a moderate positive relationship between average social support and average happiness scores.
# 
# ##### *The p-value of 0.012 indicates that the observed correlation between average social support and average happiness scores is statistically significant.
# 
# ##### *The regression equation of y = 2.983x + 2.583 shows that, on average, for every unit increase in the average social support, the average happiness score is expected to increase by approximately 2.983 units..
# 
# 

# ### 8. Create a 3D plot comparing average social support, average minimum wage and average happiness scores.
# 
# ##### Find the average minimum wage for each country

# In[36]:


# Create a data frame using minimum wage data.
mini2 = minihappy.loc[:,['Country', '2015','2016','2017','2018']]

# Find the average minimum wage for each country over the years 2015-2018.
mini2['Average Minimum Wage Yearly in USD 2015-2018']=mini2.mean(axis=1)

# Create a series featuring the average minimum wage for the years 2015-2018 for all the countries present
# across the repsective datasets.

mini2 = mini2.set_index('Country')
mini2 = mini2.loc[:,['Average Minimum Wage Yearly in USD 2015-2018']]

# Sort the data from countries with the highest minum wage to the lowest.
mini2 = mini2.sort_values('Average Minimum Wage Yearly in USD 2015-2018', ascending=False)

mini2


# ##### Create a data frame to faciltiate the merging of average social support, average minimum wage and average happiness scores for each country in order to faciltate creation of 3D plot

# In[37]:


# Create dataframe
minhap = pd.concat([mini2, happiness_score], axis=1)

# Sorting values for rows by average minimum wage from highest to lowest

minhap = minhap.sort_values('Average Minimum Wage Yearly in USD 2015-2018', ascending=False)

minhap.head()


# ##### Merge the minhap and social_happy data frames 

# In[38]:


# Merge the data frames
wage_social_happy = pd.merge(minhap, social_happy)
wage_social_happy = wage_social_happy.set_index('Country')
wage_social_happy


# ##### Reset the index to facilitae creation of plot

# In[39]:


wage_social_happy = wage_social_happy.reset_index()
wage_social_happy.head()


# In[40]:


# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(
    wage_social_happy['Average Minimum Wage Yearly in USD 2015-2018'],
    wage_social_happy['Average Happiness Score 2015-2018'],
    wage_social_happy['Average Social Support 2015-2018'],
    c='b',  # Color of the data points
    marker='o'  # Marker style
)

# Add country labels to the plot
for i, txt in enumerate(wage_social_happy['Country']):
    ax.text(
        wage_social_happy.loc[i, 'Average Minimum Wage Yearly in USD 2015-2018'],
        wage_social_happy.loc[i, 'Average Happiness Score 2015-2018'],
        wage_social_happy.loc[i, 'Average Social Support 2015-2018'],
        txt,  # Country label
        size=8,  # Text size
        zorder=1,  # Priority over data points
        color='k'  # Text color
    )

# Set labels and title
ax.set_xlabel('Average Minimum Wage (USD)')
ax.set_ylabel('Average Happiness Score')
ax.set_zlabel('Average Social Support')
ax.set_title('Relationship between Minimum Wage, Happiness Score, and Social Support')

# Calculate correlation coefficients
corr_min_wage_happiness = wage_social_happy['Average Minimum Wage Yearly in USD 2015-2018'].corr(wage_social_happy['Average Happiness Score 2015-2018'])
corr_min_wage_social_support = wage_social_happy['Average Minimum Wage Yearly in USD 2015-2018'].corr(wage_social_happy['Average Social Support 2015-2018'])

# Perform hypothesis testing
pvalue_min_wage_happiness = stats.pearsonr(wage_social_happy['Average Minimum Wage Yearly in USD 2015-2018'], wage_social_happy['Average Happiness Score 2015-2018'])[1]
pvalue_min_wage_social_support = stats.pearsonr(wage_social_happy['Average Minimum Wage Yearly in USD 2015-2018'], wage_social_happy['Average Social Support 2015-2018'])[1]

# Display correlation coefficients and p-values
print(f"Correlation between Minimum Wage and Happiness Score: {corr_min_wage_happiness:.3f}")
print(f"Correlation between Minimum Wage and Social Support: {corr_min_wage_social_support:.3f}")
print(f"P-value for correlation between Minimum Wage and Happiness Score: {pvalue_min_wage_happiness:.3f}")
print(f"P-value for correlation between Minimum Wage and Social Support: {pvalue_min_wage_social_support:.3f}")

# Show the plot
plt.show()


# ##### *The correlation coefficient between the average minimum wage and happiness score is 0.450. This indicates a moderate positive correlation between these two variables. 
# 
# ##### *The correlation coefficient between the average minimum wage and social support is 0.363. This suggests a moderate positive correlation between these two variables. Higher minimum wages tend to be associated with higher levels of social support, but the relationship is not as strong as with the happiness score.
# 
# ##### * The p-value associated with the correlation between minimum wage and happiness score is 0.011. This indicates that the observed correlation is statistically significant at a significance level of 0.05. Therefore, we can reject the null hypothesis that there is no correlation between minimum wage and happiness score.
# 
# ##### * The p-value associated with the correlation between minimum wage and social support is 0.044. This suggests that the observed correlation is statistically significant at a significance level of 0.05. 
# 
# 

# ### 9.  Compare levels of government corruption with minimum wage

# In[41]:


# Use minihappy data frame to make new data frame comparing Government Corruption with Minimum Wage
minihappy.head()


# In[42]:


minihappy.info()


# In[43]:


# Create new data frame that returns average government_corruption score data.
corrupt_wage = minihappy.iloc[:,[0,10,17,24,31]]
corrupt_wage['Average Government Corruption 2015-2018']=corrupt_wage.mean(axis=1)
corrupt_wage.head()


# In[44]:


corrupt_wage.info()


# In[45]:


# Create series that returns Average Government_Corruption 2015-2018.
corrupt_wage2 = corrupt_wage.iloc[:,[0,5]]
corrupt_wage2 = corrupt_wage2.set_index('Country')
corrupt_wage2 = corrupt_wage2.sort_values('Average Government Corruption 2015-2018', ascending=False)
corrupt_wage2.head()


# In[46]:


# Merge the mini2 series and corrupt_wage2 series to create data frame 
# comparing a country's Average Minimum Wage with Government Corruption.
wages_corrupt = pd.concat([mini2, corrupt_wage2], axis=1)
wages_corrupt.head()


# In[47]:


# Reset the index to facilitate the creation of a scatterplot.
wages_corrupt = wages_corrupt.reset_index()
wages_corrupt.head()


# In[48]:


# Create scatterplot comparing Average Minimum Wage with Government Corruption.
sns.regplot(x='Average Minimum Wage Yearly in USD 2015-2018', y='Average Government Corruption 2015-2018', data=wages_corrupt)

#Set the x and y axis labels.
plt.xlabel('Average Minimum Wage Yearly in USD 2015-2018')
plt.ylabel('Average Government Corruption 2015-2018')

#Set the title of the plot.
plt.title('Average Minimum Wage Yearly in USD and Average Government Corruption 2015-2018')

#Calculate the regression equation, correlation coefficient, p-value, and standard error.
slope, intercept, r_value, p_value, std_err = stats.linregress(wages_corrupt['Average Minimum Wage Yearly in USD 2015-2018'],
wages_corrupt['Average Government Corruption 2015-2018'])
correlation_coefficient = np.corrcoef(wages_corrupt['Average Minimum Wage Yearly in USD 2015-2018'], wages_corrupt['Average Government Corruption 2015-2018'])[0,1]

#Display the regression equation, correlation coefficient, p-value, and standard error.
#Set the slope, intercept, correlation coefficient, and p_value to .3f to round the numbers to three decimal places.
print(f"Regression Equation: y = {slope:.3f}x + {intercept:.3f}")
print(f"Correlation Coefficient: {correlation_coefficient:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Standard Error: {std_err:.3f}")

#Display the plot.
plt.show()


# ##### * The regression equation indicates that there is a positive slope (0.000) but with a very small value, suggesting that there is almost no linear relationship between the average minimum wage and government corruption.
# 
# ##### *The correlation coefficient is 0.728, which indicates a moderately strong positive correlation between average minimum wage and government corruption.
# 
# ##### *The p-value is 0.000, which is below the typical significance level of 0.05. This suggests that the correlation between average minimum wage and government corruption is statistically significant.
# 
# ##### *The standard error of 0.000 suggests that the estimate of the regression coefficient is very precise. However, since the slope coefficient is close to zero, the standard error is not particularly informative in this case.
# 
# 

# # Conclusion

# ##### *It's interesting to note that countries with low minimum wage, but high happiness had lower average social support scores than countries with high minimum wage and high happiness as well as the average social supprt scores for all countries.  It's possible these reults are due to the small sample size used in this study.  A larger sample sizae would be more appropriate.  
# 
# ##### * While there is a positive relationship between social support and happiness scores, this analysis does not establish a causal relationship. Other factors and variables not considered in this analysis could influence happiness scores.
# 
# ##### * The results indicate that there is a moderate positive correlation between minimum wage and both happiness score and social support. Higher minimum wages are associated with higher levels of happiness and social support, although the strength of the relationships is not extremely strong.
# 
# ##### *Overall, there is a moderate positive correlation between average minimum wage and government corruption, but the slope of the relationship is almost negligible.

# ## Suggestions for Further Study

# ##### It may be useful to expand the study to include more countries and cover a larger time period.
# 
# ##### Explore the relationships between other variables in the datasets:
# 
# #####  Compare happiness scores and minimum wage with generosity.
# 
# 
# 
# ##### Explore the relationship between happiness and other socioeconomic factors not present in these datasets such as income inequlity, education level and employment rate and how they interact with minimum wage.
# 
# ##### Explore the relationship between minimum wage and different demographic groups, such as age, gender and race to understand how it impacts happiness among these different groups.
# 
# 

# In[ ]:




