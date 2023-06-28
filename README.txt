Introduction

This project aims to explore the relationship between minimum wage, happiness, and social support in 31 countries. The primary focus is to analyze whether countries with relatively low minimum wages but high happiness scores also report higher levels of social support compared to countries with high happiness and high minimum wages, as well as the relationship between social support and happiness and the relationship between minimum wage, social support and happiness scores for the data as a whole. Additionally, the project investigates the connection between minimum wage and government corruption.

Minimum wage, representing the lowest legally mandated remuneration for workers, has been associated with happiness and well-being globally. By examining the interplay between minimum wage, happiness scores, and social support, this project sheds light on the factors influencing happiness and well-being within different socioeconomic contexts. Moreover, it delves into the relationship between minimum wage and government corruption.

Questions:
* What is the relationship between the average social support score for countries with relatively low minimum wage but high happiness scores as compared to the average social support score for countries with relatively high minimum wage and high happiness scores as well as to the average social support score for all countries?
* What is the relationship between average social support scores and average happiness scores?
* What is the relationship between social support, happiness scores and minimum wage?
* What is the relationship between minimum wage and government corruption? Do countries with a relatively low minimum wage correlate significantly with higher levels of government corruption?




REQUIREMENTS

The necessary libraries to be installed are listed in the 'Requirements.txt' folder.

This project was created working in Jupyter Notebook with Python 3.9.13 (ipykernel).

To run this project it is necessary to do the following:

Clone the repository to your local machine.

Navigate to the cloned repo folder in the terminal and run:

	pip install -r requirements

In case there is a problem with this method I have included a list of the libraries below to be installed individually:

	import matplotlib
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import numpy as np
	import pandas as pd
	import seaborn as sns
	import scipy.stats as stats

The datasets analyzed in this project are included in the `data` folder:
	
	`MINIMUM_WAGES.csv`
	`2015.csv`
	`2016.csv`
	`2017.csv`
	`2018.csv`

In order to make them run from the `data` folder in `Minimujm_Wage_Happiness.ipynb`, in the second code block you will need to change the names to:
	
	`data/MINIMUM_WAGES.csv`
	`data/2015.csv`
	`data/2016.csv`
	`data/2017.csv`
	`data/2018.csv`

Otherwise you can download the .csv files from the links listed below and run them from your local machine.

The main file in this project is called 'Wages_Social_Support_Corruption_Happiness
.ipynb' and contains the data analysis conducted in this project.


HOW TO RUN THE PROGRAM IN JUPYTER NOTEBOOK

	1. Clone the repository.
	2. Save the Folder.
	3. Open `Jupyter Notebook` from the command line or start menu.
	4. Go to the saved location of the repo.
	5. Open 'Wages_Social_Support_Corruption_Happiness.ipynb'.
	6. Open the `Cell` tab and click `Run All`.

HOW TO RUN THE PROGRAM IN PYTHON

	1. Clone the Repository.
	2. Save the folder.
	3. Open the saved repository in your terminal or IDE.
	4. Run the `Wages_Social_Support_Corruption_Happiness.py` file.


METHODOLOGY

Data was taken from the website https://www.kaggle.com accessed on 3/7/2022

1. https://www.kaggle.com/datasets/frtgnn/minimum-wages-between-2001-2018

CSV name: MINIMUM_WAGES.csv

The dataset provided by Firat Gonen gives the minimum wage of 31 countries. The minimum wage is yearly in US Dollars. The countries appear to have been selected randomly.

2. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2015.csv

3. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2016.csv
   
4. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2017.csv

5. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2018.csv


CSV names: 2015.csv
	   2016.csv
           2017.csv
           2018.csv

The datasets provided by Sustainable Development Solutions Network gives the happiness score for 155-158 countries. The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. The dataset for 2019.csv was not included since the MINIMUM_WAGES dataset only goes up to the year 2018.



FEATURES UTILIZED

1. Read data:
I read in five CSV data files from the website Kaggle.com.

2. Cleaning data and performing a pandas merge and then calculate new values:
I used '.strip()', '.drop()', '.rename()', '.insert()', '.fillna()', '.merge()', etc... to clean and manipulate the data. 
I used '.head()', '.info()', ','.describe()', '.isna().sum()', '.fillna()', '.sort_values()', '.mean(), etc... to analyze the data.

3. Visualize your data:
I used matplotlib, mpl_toolkits.mplot3d, seaborn, and scipy.stats to create a bar chart, multiple scatterplots and to calculate the regression equation and p value, etc...

4. Instructions for how to utilize a virtual environment are included in the Requirements.txt folder.

5. Interpret your data:
The interpretation of the data is included in the Markdown cells.  

CONCLUSION:

It's interesting to note that countries with low minimum wage, but high happiness had lower average social support scores than countries with high minimum wage and high happiness as well as the average social support scores for all countries.  It's possible these results are due to the small sample size used in this study.  A larger sample size would be more appropriate.  

While there is a positive relationship between social support and happiness scores, this analysis does not establish a causal relationship. Other factors and variables not considered in this analysis could influence happiness scores.

The results indicate that there is a moderate positive correlation between minimum wage and both happiness score and social support. Higher minimum wages are associated with higher levels of happiness and social support, although the strength of the relationships is not extremely strong.

Overall, there is a moderate positive correlation between average minimum wage and government corruption, but the slope of the relationship is almost negligible.



CITATIONS


1. https://www.kaggle.com/datasets/frtgnn/minimum-wages-between-2001-2018

2. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2015.csv

3. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2016.csv
   
4. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2017.csv

5. https://www.kaggle.com/datasets/unsdsn/world-happiness?select=2018.csv