##### Heart Disease Prediction Using Machine Learning



This project predicts the likelihood of heart disease using patient data. It demonstrates how machine learning algorithms can be applied to healthcare data to make predictions. Users can run the Jupyter Notebook to see the full workflow, from data exploration to model evaluation.



##### Features



**Data Exploration**: Understand the dataset and visualize key patterns.



**Model Training**: Train multiple machine learning models on patient data.



**Model Evaluation**: Compare models using accuracy, precision, recall, and F1-score.



**Prediction**: Make predictions on unseen data using the best-performing model.



##### Data and Methodology

##### 

###### Dataset:



The dataset used includes critical medical parameters such as:



**cp (chest pain type)**: Strongly linked to heart attack prediction.



**oldpeak**: Reflects ST Depression in ECG, indicating myocardial ischemia.



**ca (number of major vessels)**: Related to blood flow conditions.



Other parameters include age, cholesterol levels, blood pressure, and more.



##### Machine Learning Models



The following models were used to predict heart disease:



Logistic Regression



Random Forest Classifier



K-Nearest Neighbours (KNN) Classifier



Among these, Logistic Regression achieved the highest accuracy of **88.18%** and was selected as the final model.



##### Model performance:



&nbsp;                                        **Model**                      | **Accuracy**

&nbsp;                                        ----------------------------|------------

&nbsp;                                        KNN                        | 0.6885

&nbsp;                                        Logistic Regression   | 0.8852

&nbsp;                                        Random Forest         | 0.8361



##### Steps in the Project



**Data Loading \& Exploration**: Load the dataset and perform exploratory data analysis (EDA).



**Data Pre-processing**: Handle missing values, separate the target variable, and split data into training and testing sets.



**Model Training**: Train Logistic Regression, Random Forest, and KNN classifiers.



**Model Evaluation**: Evaluate each model and select the best one based on accuracy and other metrics.



**Prediction**: Use the best-performing model to make predictions on new data.



##### Visualizations:



###### **Model Accuracy Comparisons:**





##### Conclusion:



Logistic Regression is the most effective model for predicting heart disease in this dataset. Features like chest pain type, maximum heart rate, and ST depression are strong indicators for the presence of heart 



disease. Machine learning models can help doctors and healthcare professionals make early predictions and provide better patient care.



Further improvements can be made by adding more data, tuning hyperparameters, or trying other algorithms like Support Vector Machines or Neural Networks.



###### Getting Started



**Prerequisites**



Python 3.8 or later



Libraries: NumPy , pandas, scikit-learn, matplotlib, seaborn



**Installation:**



Clone the repository:



https://github.com/Sreejaswee-kantepalli/Heart-Disease-Prediction-Using-Machine-Learning



pip install -r requirements.txt python main.py



Copy this code and save it as README.md in the root directory of your GitHub repository. Adjust the dataset link and any other specific project details as needed.

 



                                                              





