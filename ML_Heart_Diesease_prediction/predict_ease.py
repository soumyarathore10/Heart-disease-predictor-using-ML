import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the Heart Disease UCI dataset

data = pd.read_csv("C:/Users/soumy/Desktop/python/ml/heart_data.csv", na_values='?')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Preprocess the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Feature scaling with feature names
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Function to make predictions
def predict():
    try:
        # Get user inputs
        age = float(entries[0].get())
        sex = float(entries[1].get())
        cp = float(entries[2].get())
        trestbps = float(entries[3].get())
        chol = float(entries[4].get())
        fbs = float(entries[5].get())
        restecg = float(entries[6].get())
        thalach = float(entries[7].get())
        exang = float(entries[8].get())
        oldpeak = float(entries[9].get())
        slope = float(entries[10].get())
        ca = float(entries[11].get())
        thal = float(entries[12].get())

        # Scale the user inputs
        user_inputs = scaler.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make prediction probability
        probability = lr_classifier.predict_proba(user_inputs)[0, 1]  # Probability of having heart disease
        prediction = 1 if probability > 0.5 else 0 

        # Display result in the GUI
        result_str = f"The predicted probability of having heart disease is: {probability:.2%}\n"
        result_str += f"The predicted result is: {'Positive' if prediction == 1 else 'Negative'}"
        result_label.config(text=result_str)

        # Calculate and display accuracy
        y_pred = lr_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_label.config(text=f"Accuracy on Test Data: {accuracy:.2%}")

        # Display confusion matrix in the GUI
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all inputs")

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.clf()  
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)  
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

   
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    
    canvas.draw()

# Create GUI window
root = tk.Tk()
root.title("Heart Disease Predictor")


background_color = "#efc69b"
text_color = "#473144"  
button_color = "#af1b3f"  
accent_color = "#df9b6d"  
result_color = "#ccb69b"  


style = ttk.Style(root)
style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), foreground=text_color)
style.configure("Subtitle.TLabel", font=("Helvetica", 12, "italic"), foreground=text_color)
style.configure("TButton", font=("Helvetica", 12), foreground=text_color)


title_label = ttk.Label(root, text="HEART DISEASE PREDICTOR", style="Title.TLabel")
title_label.grid(row=0, column=0, columnspan=2, pady=10)


labels = [
    "Age:", "Sex (0 for female, 1 for male):", "Chest Pain Type (1-3):",
    "Resting Blood Pressure:", "Cholesterol:", "Fasting Blood Sugar (0 for False, 1 for True):",
    "Resting Electrocardiographic Results:", "Maximum Heart Rate Achieved:",
    "Exercise Induced Angina (0 for No, 1 for Yes):", "ST Depression Induced by Exercise Relative to Rest:",
    "Slope of the Peak Exercise ST Segment:", "Number of Major Vessels Colored by Fluoroscopy:", "Thalassemia:"
]


input_frame = ttk.Frame(root, style="TFrame")
input_frame.grid(row=1, column=0, padx=10, pady=10)

entries = []

for i, label in enumerate(labels):
    ttk.Label(input_frame, text=label, style="Title.TLabel").grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
    entry = ttk.Entry(input_frame, font=("Helvetica", 12))
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry.configure(style="TEntry")
    entries.append(entry)


predict_button = ttk.Button(root, text="Predict", command=predict, style="TButton")
predict_button.grid(row=2, column=0, columnspan=2, pady=10)


result_label = ttk.Label(root, text="", style="Subtitle.TLabel", background=result_color)
result_label.grid(row=3, column=0, columnspan=2, pady=5)


accuracy_label = ttk.Label(root, text="", style="Subtitle.TLabel", background=result_color)
accuracy_label.grid(row=4, column=0, columnspan=2, pady=5)


graph_frame = ttk.Frame(root, style="TFrame")
graph_frame.grid(row=1, column=1, padx=10, pady=10)

figure, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(figure, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


root.mainloop()
