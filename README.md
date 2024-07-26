# Project-3--Women-Wellbeing

# Project Team 
Raisha Rawal, Elia Porter, Robb Stenman

# Executive Summary
This project aims to develop a comprehensive Global Women's Professional Well-Being Index using key datasets that reflect various aspects of women's professional well-being across countries. The index will provide valuable insights into women's professional well-being in countries and globally.

# Project Description
The project aims to provide insight on the professional development of women globally by using a collection of economic and social indicators for various entities, with each row representing a single entity. The columns represent different variables, including codes, years, and various metrics such as average hours worked, employment population ratio, gender wage gap, labor force, GDP per capita, school years, paid leave, agriculture, industry, and female share of employment in services.
The goal of the project is to build a professional wellbeing score app where one can enter an entity name and is able to view the professional wellbeing score as well as suggestions and solutions for the entity to improve the score. 
In order to reach our goal, we used the standard data science process flow. 

1.	Data Collection: We sourced our data from https://ourworldindata.org/women-rights. Each of the dataset that we used were indicators of professional wellbeing. We used 7 datasets for this project, each of the dataset would be used to predict the professional wellbeing score. Each dataset had three common columns: Entity, Code and Year. Rest of the columns were unique features which would be used to build the professional wellbeing score. The dataset shape were as follows,
    -	Shape of average_hours DataFrame: (971, 4)
    -	Shape of employment_ratio DataFrame: (3919, 4)
    -	Shape of wage_gap DataFrame: (636, 4)
    -	Shape of percapita_labor DataFrame: (7185, 6)
    -	Shape of school_years DataFrame: (1502, 4)
    -	Shape of maternity_leave DataFrame: (10206, 4)
    -	Shape of labor_sector DataFrame: (2781, 6)
2.	Data Cleaning: We first worked on renaming each of the feature columns. We then moved to drop some null values from two dataset before merging all the dataset to created our DataFrame merged_df. We used a series of outer merges using the “merge” function from pandas. Then we checked for some duplicate values and removed a column called “Continent” as it did not add any value to our project goals. 

3.	Exploratory Data Analysis: For EDA, we first began by check the datatypes for each of the columns in the DataFrame.     
    All of our features had the datatype = float64 and the common columns Entity and Code = Object and Year= Int. 
    We then used the describe function to view the generative descriptive of the dataset which provided the summary of the central tendency, dispersion and shape of the dataset’s distribution. 
    After looking at the datasets distribution, we worked on handling the null values. We opted to use the KNNimputer to take care of our null values. This is a module used to impute the missing values in a dataset using the K-Nearest Neighbors algorithm.  To prepare for this module we identified the numerical columns and the categorical columns so that we could train only on the numerical columns. We used n_neighbors = 5. For each row with a missing value in a numerical column, the imputer finds the 5 most similar rows. We also changed the number to 15 but it didn’t make much of a change. Hence, we moved back to 5. 
    After imputing the data and ensuring the null values to be 0, we then moved on to some basic EDA of looking at top 10 and bottom 10 countries for each of the features.
    To prepare for the models we had to first create our professional wellbeing score 0-1. Once we successfully created the score, we then used to move to model building. 

4.	Model Building: 


# Next Steps / Future Considerations