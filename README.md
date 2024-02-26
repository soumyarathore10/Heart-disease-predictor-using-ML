# Heart-disease-predictor-using-ML
The project is a heart disease prediction system that uses machine learning to analyze user input and predict the likelihood of a person having heart disease. It is built using Python, scikit-learn for machine learning, and Tkinter for the graphical user interface (GUI).

Here's a step-by-step description of the project:

Data Collection: The project uses the Heart Disease UCI dataset, which contains various attributes related to heart health, such as age, sex, cholesterol levels, and more.

Data Preprocessing: The dataset is preprocessed to handle missing values. Any missing values in the dataset are replaced with the mean of the respective columns.

Model Training: A Random Forest Classifier is trained on the preprocessed dataset. The Random Forest algorithm is an ensemble learning method that builds multiple decision trees during training and merges them together to get a more accurate and stable prediction.

Graphical User Interface (GUI): The project includes a GUI built using Tkinter, a standard Python library for creating GUI applications. The GUI allows users to input their information, such as age, sex, cholesterol levels, etc., and click a button to get a prediction.

Prediction: When the user clicks the "Predict" button, the input data is passed to the trained Random Forest model, which predicts whether the user is likely to have heart disease or not. The prediction result is displayed on the GUI along with a probability score.

Confusion Matrix Visualization: Additionally, the project displays a confusion matrix as a bar chart to visualize the performance of the model. The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions made by the model.

Overall, this project demonstrates how machine learning can be used to predict heart disease based on user input and provides a user-friendly interface for interacting with the prediction system
