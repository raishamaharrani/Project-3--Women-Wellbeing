# Project-3--Women-Wellbeing

# Project Team 
Raisha Rawal, Elia Porter, Robb Stenman

# Executive Summary
This project aims to develop a comprehensive Global Women's Professional Well-Being Index using key datasets that reflect various aspects of women's professional well-being across countries. The index will provide valuable insights into women's professional well-being in countries and globally.

# Project Description
The project aims to provide insight on the professional development of women globally by using a collection of economic and social indicators for various entities, with each row representing a single entity. The columns represent different variables, including codes, years, and various metrics such as average hours worked, employment population ratio, gender wage gap, labor force, GDP per capita, school years, paid leave, agriculture, industry, and female share of employment in services.
The goal of the project is to build a professional wellbeing score app where one can enter an entity name and is able to view the professional wellbeing score. The user will be able to view explaination for the score and view suggestions and solutions for the entity to improve the score. 
In order to reach our goal, we used the standard data science process flow as follows:

1.	Data Collection: We sourced our data from https://ourworldindata.org/women-rights. Each of the dataset that we used were indicators of professional wellbeing. We used 7 datasets for this project, each of the dataset would be used to predict the professional wellbeing score. Each dataset had three common columns: Entity, Code and Year. Rest of the columns were unique features which would be used to build the professional wellbeing score. The dataset shape were as follows,
    -	Shape of average_hours DataFrame: (971, 4)
    -	Shape of employment_ratio DataFrame: (3919, 4)
    -	Shape of wage_gap DataFrame: (636, 4)
    -	Shape of percapita_labor DataFrame: (7185, 6)
    -	Shape of school_years DataFrame: (1502, 4)
    -	Shape of maternity_leave DataFrame: (10206, 4)
    -	Shape of labor_sector DataFrame: (2781, 6)
2.	Data Cleaning: We first worked on renaming each of the feature columns. We then moved to drop some null values from two dataset before merging all the dataset to created our DataFrame merged_df. We used a series of outer merges using the “merge” function from pandas. Then we checked for some duplicate values and removed a column called “Continent” as it did not add any value to our project goals. 

3. Exploratory Data Analysis: For EDA, we first began by checking the datatypes for each column in the DataFrame. Our features had the datatype = float64 and the standard columns Entity and Code = Object and Year= Int.
We then used the describe function to view the generative descriptive of the dataset, which provided the summary of the central tendency, dispersion, and shape of the dataset’s distribution.
After looking at the dataset distribution, we worked on handling the null values. We opted to use the KNNimputer to take care of our null values. This is a module used to impute the missing values in a dataset using the K-Nearest Neighbors algorithm.  To prepare for this module, we identified the numerical and categorical columns so that we could train only on the numerical columns. We used n_neighbors = 5. The imputer finds the five most similar rows for each row with a missing value in a numerical column. We also changed the number to 15, but it didn’t make much of a change. Hence, we moved back to 5.
After imputing the data and ensuring the null values are 0, we moved on to some basic EDA by looking at each feature's top 10 and bottom ten countries.
Creating target variable: To prepare for the models, we first created our professional well-being score of 0-1. Once we successfully developed the score, we moved on to the model building.

4.	Model Building: For model building, we scaled and split the data into training and testing sets and built some simple models to check the performance of our dataset. Each model results in MSE and the R², which determine the performance of these models on our dataset.
-     The MSE measures the average squared difference between the predicted and actual values. A lower MSE indicates better performance, with a value of 0 indicating perfect prediction.
-     The R² measures the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R² indicates better performance, with a value of 1 indicating perfect prediction.
 Model #1 Random Forest Regressor model: The GridSearchCV identified a set of optimal hyperparameters for the Random Forest Regressor model, which resulted in a highly accurate and explanatory model. The model uses a large number of trees, is allowed to split internal nodes with few samples, and uses a subset of features at each split. The model is not limited by a maximum depth, which allows it to grow deeper trees and improve predictions. The results suggest that the model is well-suited for the task and can be used to make accurate predictions.
-          Mean Squared Error:  5.9229050140828465e-05
-          R² Score:  0.9889872679319625
 Model #2 LightGBM Regressor model: The results suggest that the LightGBM model performs exceptionally well on the dataset. The MSE is very low, indicating that the model makes accurate predictions with minimal error. The R² Score is very high, indicating that the model explains almost all of the variance in the dependent variable. Overall, the results suggest that the LightGBM model is a good fit for the dataset and can be used to make accurate predictions.
 
-          LightGBM MSE: 3.9479925593837536e-05
-          LightGBM R2 Score: 0.9926593142791044R2: 0.9841876509160756
 
 Model #3 Linear Regression Model: The linear regression result suggests that the model fits the training data almost perfectly, which indicates the problem of overfitting.
-          MSE: 6.22747124967587e-32
-          R2: 1.0
 Model #4 Gradient Boosting Regressor: This is a type of supervised learning algorithm that combines multiple weak models to create a robust predictive model. In this case, the model is used for regression tasks, predicting a continuous output variable. Based on these evaluation metrics, the Gradient Boosting Regressor is performing exceptionally well. The low MSE indicates accurate predictions, and the high R² indicates a strong relationship between the variables. This suggests that the model is a good fit for the data and can be used to make reliable predictions.
-          MSE: 6.958849219496495e-05
-          R2: 0.9870610888113232
    Hyperparameter tuning on Gradient Boosting Regressor results show that tuning the parameters helped the model perform extremely well. The very low MSE indicates highly accurate predictions, and the high R² indicates a strong relationship between the variables. This suggests that the model is a good fit for the data and can be used to make highly reliable predictions.
 
-          MSE: 3.0609370876092964e-05
-          R2: 0.9943086576700437
 
 Model #5 XGBoost Regressor: Extreme Gradient Boosting is a popular and powerful supervised learning algorithm widely used for regression and classification tasks. It's implementing the gradient boosting framework designed to be highly efficient and scalable.
This model also performed very well for our project. The very low MSE indicates highly accurate predictions, and the very high R² indicates a strong relationship between the variables. This suggests that the model is a good fit for the data and can be used to make highly reliable predictions.
-          MSE: 4.445643726121712e-05
-          R2: 0.9917340083777603
 
    We also performed hyperparameter tuning on the XGBoost Model to see if the performance would get better, and it did. Here are the results
-           MSE: 2.2772498559815083e-05
-           R2: 0.9957658036966198
 
 Model #6 Neural Network Model: We began by using sequential and dense modules to create linear stacks and dense, fully connected neural network layers.
We then defined our input features (X) and our target variable (Y) and split them into test and train sets using the train_test_split function from Scikit-Learn. For reproducibility, we set the test size to 20%, the train size to 80%, and the random state parameter to 42. We used the neural network model “Sequential,” which consists of 64 neurons on the first layer, 32 on the second layer, and a single neuron on the third, which is the output layer.

We used the ReLU activation function as it allows the model to learn more complex relationships between the input features and the target variable. Other activation functions are more suitable for classification problems.

The results of the neural network model training process started with 50 epochs. The loss reported for each epoch decreased over time, indicating that the model is learning and improving its predictions. The following snip is from epoch 44-50
- Epoch 44/50
- 270/270 - 0s - 858us/step - loss: 9.9018e-06
- Epoch 45/50
- 270/270 - 0s - 845us/step - loss: 6.8179e-06
- Epoch 46/50
- 270/270 - 0s - 850us/step - loss: 7.1100e-06
- Epoch 47/50
- 270/270 - 0s - 795us/step - loss: 9.6552e-06
- Epoch 48/50
- 270/270 - 0s - 835us/step - loss: 1.2141e-05
- Epoch 49/50
- 270/270 - 0s - 889us/step - loss: 5.4070e-06
- Epoch 50/50
- 270/270 - 0s - 925us/step - loss: 5.7331e-06
 
Building Gradio App: To view the professional well-being score for each entity in our dataset, we utilized Gradio, a Python library that allows you to create a user-friendly interface for machine learning models. This helped us visualize the professional well-being score for each entity entered.

To make the project goals and results more interactive, we incorporated explanations and suggestions functions to display on the Gradio app by calling LLM-OPENAI through an API key.
The explanation function generates explanations for an entity's professional well-being score based on its feature values compared to the average values across all entities. The suggestions function generates suggestions to improve an entity's professional well-being based on its explanations.


# Things we tried: 
- Additional Models: 
    - Randomized Search Cv, however we got the following error, The error you're encountering indicates that the target variable (y) has continuous values, which is not suitable for a RandomForestClassifier because it expects categorical labels for classification tasks. If your task involves predicting continuous values, you should use a regression model instead, like RandomForestRegressor

    - Valence Aware Dictionary for Sentiment Reasoning (VADER) is a powerful tool for sentiment analysis and it’s widely used in natural language processing applications. VADER is primarily designed for short texts and is not a machine learning classifier and was not suitable for our project as its primary function is sentiment analysis and we were looking at scores for overall professional well-being.

    - OpenAI Whisper is a powerful and flexible tool for transcribing and analyzing audio files, we tried to connect it with our Gradio App for the explanations and suggestions features, however there were audio connection issues. 

# Next Steps /Future Considerations
- Refining the models
- Expanding the dataset to look at other factors such as women personal and social wellbeing globally. 

Future Considerations
- Use more advance machine learning techniques such as deep learning and transfer learning 
- Incorporating additional data sources
- Developing a more sophisticated explanations function that can provide more detailed and actionable insights into the factors influencing an entity's professional wellbeing score.
- Creating a more user-friendly interface such as integrating whisper 

# References 
- Xpert Learning 
- ChatGpt 
- Google Search 
- Peer Support 
- Instructor Support 
