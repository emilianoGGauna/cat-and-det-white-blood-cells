#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
# FOLD 1
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

# Métricas de entrenamiento
train_loss_FOLD1 = [0.3950, 0.1788, 0.1518, 0.1109, 0.0654, 0.0569, 0.0716, 0.0898, 0.0625, 0.0466, 0.0748, 0.0829]
train_accuracy_FOLD1 = [0.8864, 0.9419, 0.9571, 0.9637, 0.9812, 0.9798, 0.9792, 0.9726, 0.9805, 0.9871, 0.9785, 0.9742]

# Métricas de validación
validation_loss_FOLD1 = [1.1485, 0.4332, 0.3126, 0.3259, 0.1594, 0.3322, 0.4789, 0.2821, 0.3445, 0.1090, 0.3668, 0.1607]
validation_accuracy_FOLD1 = [0.7341, 0.8241, 0.9087, 0.9193, 0.9616, 0.9206, 0.9233, 0.9312, 0.9286, 0.9722, 0.9233, 0.9616]

# Tiempo por época (en segundos, asumiendo que los tiempos están en formato MM:SS)
time_per_epoch_FOLD1 = [
    29*60 + 25, 29*60 + 32, 29*60 + 32, 29*60 + 27, 29*60 + 19, 29*60 + 19, 29*60 + 20, 29*60 + 19, 29*60 + 21, 29*60 + 14, 28*60 + 38, 29*60 + 22
]

# Calcular el tiempo total (suma de los tiempos de entrenamiento y validación)
total_time_FOLD1 = sum(time_per_epoch_FOLD1)

import pandas as pd

# Matriz de confusión
conf_matrix_data_FOLD1 = [
    [106, 1, 0, 5, 2, 1, 0],
    [0, 128, 1, 0, 0, 0, 0],
    [0, 10, 101, 0, 0, 0, 0],
    [0, 0, 0, 91, 1, 1, 0],
    [1, 0, 0, 0, 101, 1, 0],
    [0, 4, 0, 0, 1, 97, 0],
    [0, 0, 0, 0, 0, 0, 103]
]
conf_matrix_df_FOLD1 = pd.DataFrame(conf_matrix_data_FOLD1, columns=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO', 'PLATELET'], index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO', 'PLATELET'])

# Informe de clasificación
class_report_data_FOLD1 = {
    'Precision': [0.99, 0.90, 0.99, 0.95, 0.96, 0.97, 1.00, " ", 0.97],
    'Recall': [0.92, 0.99, 0.91, 0.98, 0.98, 0.95, 1.00, " ", 0.96],
    'F1-Score': [0.95, 0.94, 0.95, 0.96, 0.97, 0.96, 1.00, 0.96 , 0.96],
    'Support': [115, 129, 111, 93, 103, 102, 103, 756 , 756]
}

classification_report_FOLD1 = pd.DataFrame(class_report_data_FOLD1, index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO', 'PLATELET', "accuracy", "macro_avg"])

#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
## FOLD 2
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

# Métricas de entrenamiento
train_loss_FOLD2 = [0.4012, 0.2512, 0.1771, 0.1826, 0.1345, 0.0724, 0.1735, 0.0674, 0.0874, 0.0981, 0.2148, 0.0918]
train_accuracy_FOLD2 = [0.8793, 0.9300, 0.9475, 0.9452, 0.9562, 0.9764, 0.9467, 0.9779, 0.9737, 0.9749, 0.9399, 0.9684]

# Métricas de validación
validation_loss_FOLD2 = [27.5277, 1.0049, 0.2285, 0.4490, 0.2319, 0.3477, 0.4313, 0.3166, 0.1970, 0.2450, 3.0911, 0.3477]
validation_accuracy_FOLD2 = [0.5549, 0.7180, 0.9253, 0.8521, 0.9345, 0.9177, 0.8872, 0.9253, 0.9466, 0.9223, 0.6723, 0.9192]

# Tiempo por época (en segundos, asumiendo que los tiempos están en formato MM:SS)
time_per_epoch_FOLD2 = [
    26*60 + 18, 25*60 + 49, 25*60 + 46, 25*60 + 45, 25*60 + 48, 25*60 + 46, 25*60 + 51, 25*60 + 53, 25*60 + 27, 25*60 + 28, 25*60 + 22
]

# Calcular el tiempo total (suma de los tiempos de entrenamiento y validación)
total_time_FOLD2 = sum(time_per_epoch_FOLD2)

# Matriz de confusión
conf_matrix_data_FOLD2 = [
    [113, 0, 0, 0, 2, 0],
    [1, 117, 1, 0, 0, 4],
    [3, 5, 107, 0, 0, 0],
    [0, 0, 0, 86, 2, 3],
    [3, 1, 0, 0, 90, 27],
    [0, 1, 0, 0, 0, 90]
]
conf_matrix_df_FOLD2 = pd.DataFrame(conf_matrix_data_FOLD2, columns=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'], index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'])

# Informe de clasificación
class_report_data_FOLD2 = {
    'Precision': [0.94, 0.94, 0.99, 1.00, 0.96, 0.73, " ", 0.92], 
    'Recall': [0.98, 0.95, 0.93, 0.95, 0.74, 0.99, " ", 0.92],
    'F1-Score': [0.96, 0.95, 0.96, 0.97, 0.84, 0.84, 0.92, 0.92],
    'Support': [115, 123, 115, 91, 121, 91, 656, 656 ]
}
classification_report_FOLD2 = pd.DataFrame(class_report_data_FOLD2, index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO', 'accuracy', 'macro avg'])

#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
### FOLD 3
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

# Métricas de entrenamiento
train_loss_FOLD3 = [0.4499, 0.2539, 0.1638, 0.1181, 0.1528, 0.0950, 0.1067, 0.1301, 0.0795, 0.0455, 0.1243, 0.0973]
train_accuracy_FOLD3 = [0.8622, 0.9284, 0.9475, 0.9650, 0.9532, 0.9726, 0.9684, 0.9589, 0.9749, 0.9844, 0.9673, 0.9688]

# Métricas de validación
validation_loss_FOLD3 = [4.4156, 1.3277, 0.4697, 0.3392, 0.1624, 0.1898, 0.3629, 1.3960, 0.1879, 0.1503, 0.3479, 0.1134]
validation_accuracy_FOLD3 = [0.6692, 0.7546, 0.8872, 0.8994, 0.9482, 0.9527, 0.9131, 0.8232, 0.9375, 0.9619, 0.9573, 0.9588]

# Tiempo por época (en segundos, asumiendo que los tiempos están en formato MM:SS)
time_per_epoch_FOLD3 = [
    25*60 + 30, 25*60 + 27, 25*60 + 42, 25*60 + 50, 25*60 + 6, 25*60 + 5, 25*60 + 23, 25*60 + 45, 25*60 + 20, 25*60 + 35, 25*60 + 33, 25*60 + 26
]

# Calcular el tiempo total (suma de los tiempos de entrenamiento y validación)
total_time_FOLD3 = sum(time_per_epoch_FOLD3)

# Matriz de confusión
conf_matrix_data_FOLD3 = [
    [118, 0, 1, 0, 0, 0],
    [1, 113, 5, 0, 0, 0],
    [0, 4, 98, 0, 0, 0],
    [4, 1, 0, 104, 1, 0],
    [3, 0, 0, 2, 108, 2],
    [1, 2, 0, 0, 0, 88]
]
conf_matrix_df_FOLD3 = pd.DataFrame(conf_matrix_data_FOLD3, columns=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'], index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'])

# Informe de clasificación
class_report_data_FOLD3 = {
    'Precision': [0.93, 0.94, 0.94, 0.98, 0.99, 0.98, 0, 0.96],
    'Recall': [0.99, 0.95, 0.96, 0.95, 0.94, 0.97, 0, 0.96],
    'F1-Score': [0.96, 0.95, 0.95, 0.96, 0.96, 0.97, 0.96, 0.96],
    'Support': [119, 119, 102, 110, 115, 91, 656, 656]
}
classification_report_FOLD3 = pd.DataFrame(class_report_data_FOLD3, index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO', 'accuracy', 'acro avg'])

#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
#### FOLD 4 
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

train_loss_FOLD4 = [0.4145, 0.2765, 0.1810, 0.1006, 0.1978, 0.0779, 0.1088, 0.1060, 0.0887, 0.0928, 0, 0.0517]
train_accuracy_FOLD4 = [0.8736, 0.9216, 0.9478, 0.9650, 0.9467, 0.9756, 0.9657, 0.9673, 0.9772, 0.9760, 0, 0.9840]
train_time_FOLD4 = [25*60 + 18, 25*60 + 24, 26*60 + 3, 26*60 + 8, 26*60, 25*60 + 55, 25*60 + 54, 25*60 + 54, 25*60 + 49, 25*60 + 56, None, 26*60 + 41]  # in seconds

validation_loss_FOLD4 = [5.0863, 5.4389, 22.7575, 0.8502, 0.3389, 0.1932, 0.1430, 0.3396, 4.0289, 0.3594, 0, 0.1043]
validation_accuracy_FOLD4 = [0.6662, 0.5976, 0.7530, 0.8491, 0.9360, 0.9451, 0.9604, 0.9223, 0.8125, 0.9070, 0, 0.9695]
validation_time_FOLD4 = [1*60 + 48, 1*60 + 54, 1*60 + 51, 1*60 + 51, 1*60 + 50, 1*60 + 49, 1*60 + 49, 1*60 + 50, 1*60 + 50, 1*60 + 49, 0, 1*60 + 57]  # in seconds

total_time_FOLD4 = sum([t for t in train_time_FOLD4 if t is not None]) + sum([t for t in validation_time_FOLD4 if t is not None])


conf_matrix_data_FOLD4 = [
    [96, 0, 0, 0, 1, 0],
    [0, 126, 1, 0, 1, 0],
    [0, 4, 115, 0, 0, 0],
    [0, 0, 0, 100, 6, 1],
    [0, 0, 0, 1, 98, 1],
    [0, 0, 0, 0, 4, 101]
]

conf_matrix_df_FOLD4 = pd.DataFrame(conf_matrix_data_FOLD4, columns=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'], index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'])
classification_report_FOLD4  = pd.DataFrame({
    "precision": [1.00, 0.97, 0.99, 0.99, 0.89, 0.98, None, 0.97],
    "recall": [0.99, 0.98, 0.97, 0.93, 0.98, 0.96, None, 0.97],
    "f1-score": [0.99, 0.98, 0.98, 0.96, 0.93, 0.97, None, 0.97],
    "support": [97, 128, 119, 107, 100, 105, 656, 656]
}, index=["BAS", "BNE", "EO", "ERB", "LY", "MO", "accuracy", "macro avg"])

#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
##### FOLD 5
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

# Training Loss for each epoch
train_loss_FOLD5 = [0.4559, 0.2681, 0.2090, 0.1704, 0.1052, 0.1190, 0.0645, 0.1624, 0.1113, 0.1007, 0.0544, 0.0465]

# Training Accuracy for each epoch (in percentages)
train_accuracy_FOLD5 = [86.79, 92.04, 93.11, 94.52, 96.38, 96.04, 98.52, 95.05, 96.35, 97.26, 98.33, 98.71]

# Validation Loss for each epoch
validation_loss_FOLD5 = [3.4383, 5.2090, 0.5294, 0.4676, 0.1397, 0.1562, 0.2448, 0.6243, 0.1471, 0.3984, 0.2427, 0.1529]

# Validation Accuracy for each epoch (in percentages)
validation_accuracy_FOLD5 = [62.35, 59.30, 84.76, 89.63, 96.19, 96.65, 92.53, 79.73, 95.58, 89.18, 94.05, 95.43]

# Training and Validation Time for each epoch (in minutes:seconds)
# Assuming each training epoch took about 25 minutes and each validation epoch took about 1 minute and 50 seconds
train_time_FOLD5 = ["25:53", "24:53", "24:54", "25:07", "25:09", "26:29", "25:54", "25:44", "25:42", "25:42", "25:42", "25:42"]
validation_time_FOLD5 = ["1:48", "1:47", "1:51", "1:49", "1:50", "1:55", "1:50", "1:50", "1:50", "1:49", "1:50", "1:50"]

# Confusion Matrix
conf_matrix_data_FOLD5 = [
    [108, 1, 0, 0, 2, 0],
    [0, 124, 7, 0, 0, 0],
    [0, 2, 100, 0, 0, 0],
    [0, 0, 0, 96, 0, 0],
    [3, 0, 0, 4, 96, 0],
    [2, 4, 0, 3, 2, 102]
]

conf_matrix_df_FOLD5 = pd.DataFrame(conf_matrix_data_FOLD5, columns=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'], index=['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO'])

# Classification Report Data
# Classification report data for Fold 5
classification_report_data_FOLD5 = {
    "precision": [0.96, 0.95, 0.93, 0.93, 0.96, 1.00, 0.95, 0.96],
    "recall": [0.97, 0.95, 0.98, 1.00, 0.93, 0.90, 0.95, 0.95],
    "f1-score": [0.96, 0.95, 0.96, 0.96, 0.95, 0.95, 0.95, 0.95],
    "support": [111, 131, 102, 96, 103, 113, 656, 656]
}

# Creating DataFrame
classification_report_FOLD5 = pd.DataFrame(classification_report_data_FOLD5, index=["BAS", "BNE", "EO", "ERB", "LY", "MO", "accuracy", "macro avg"])

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming your accuracy and loss data are stored in lists
train_accuracy_data = [train_accuracy_FOLD1, train_accuracy_FOLD2, train_accuracy_FOLD3, train_accuracy_FOLD4, train_accuracy_FOLD5]
validation_accuracy_data = [validation_accuracy_FOLD1, validation_accuracy_FOLD2, validation_accuracy_FOLD3, validation_accuracy_FOLD4, validation_accuracy_FOLD5]
train_loss_data = [train_loss_FOLD1, train_loss_FOLD2, train_loss_FOLD3, train_loss_FOLD4, train_loss_FOLD5]
validation_loss_data = [validation_loss_FOLD1, validation_loss_FOLD2, validation_loss_FOLD3, validation_loss_FOLD4, validation_loss_FOLD5]

# Assuming your confusion matrices are stored in a list
conf_matrix_data = [conf_matrix_data_FOLD1, conf_matrix_data_FOLD2, conf_matrix_data_FOLD3, conf_matrix_data_FOLD4, conf_matrix_data_FOLD5]

# Assuming your classification reports are stored in variables
classification_reports = [classification_report_FOLD1, classification_report_FOLD2, classification_report_FOLD3, classification_report_FOLD4, classification_report_FOLD5]

# Initialize lists to store total values and counts for averaging
total_train_accuracy = [0.0] * len(train_accuracy_data[0])
total_validation_accuracy = [0.0] * len(validation_accuracy_data[0])
total_train_loss = [0.0] * len(train_loss_data[0])
total_validation_loss = [0.0] * len(validation_loss_data[0])

# Initialize counts to calculate the average later
count_train_accuracy = [0] * len(train_accuracy_data[0])
count_validation_accuracy = [0] * len(validation_accuracy_data[0])
count_train_loss = [0] * len(train_loss_data[0])
count_validation_loss = [0] * len(validation_loss_data[0])

# Loop to accumulate sums and counts
for i in range(5):
    for j in range(len(train_accuracy_data[0])):
        if train_accuracy_data[i][j] is not None:
            total_train_accuracy[j] += train_accuracy_data[i][j]
            count_train_accuracy[j] += 1
        if validation_accuracy_data[i][j] is not None:
            total_validation_accuracy[j] += validation_accuracy_data[i][j]
            count_validation_accuracy[j] += 1
        if train_loss_data[i][j] is not None:
            total_train_loss[j] += train_loss_data[i][j]
            count_train_loss[j] += 1
        if validation_loss_data[i][j] is not None:
            total_validation_loss[j] += validation_loss_data[i][j]
            count_validation_loss[j] += 1

# Calculate the averages, avoiding division by zero
avg_train_accuracy = [total / count if count != 0 else None for total, count in zip(total_train_accuracy, count_train_accuracy)]
avg_validation_accuracy = [total / count if count != 0 else None for total, count in zip(total_validation_accuracy, count_validation_accuracy)]
avg_train_loss = [total / count if count != 0 else None for total, count in zip(total_train_loss, count_train_loss)]
avg_validation_loss = [total / count if count != 0 else None for total, count in zip(total_validation_loss, count_validation_loss)]


# Creating subplots for accuracy, loss, and confusion matrices
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(30, 25))

for i in range(5):
    # Plotting accuracy
    ax_acc = axes[i, 0]
    ax_acc.plot(train_accuracy_data[i], 'o-b', label='Train Accuracy')
    ax_acc.plot(validation_accuracy_data[i], 'o-g', label='Validate Accuracy')
    ax_acc.set_title(f'Fold {i+1} Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # Update average accuracy
    avg_train_accuracy = np.add(avg_train_accuracy, train_accuracy_data[i])
    avg_validation_accuracy = np.add(avg_validation_accuracy, validation_accuracy_data[i])

    # Plotting loss
    ax_loss = axes[i, 1]
    ax_loss.plot(train_loss_data[i], 'o-k', label='Train Loss')
    ax_loss.plot(validation_loss_data[i], 'o-y', label='Validate Loss')
    ax_loss.set_title(f'Fold {i+1} Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()

    # Update average loss
    avg_train_loss = np.add(avg_train_loss, train_loss_data[i])
    avg_validation_loss = np.add(avg_validation_loss, validation_loss_data[i])

    # Plotting confusion matrix
    ax_cm = axes[i, 2]
    sns.heatmap(conf_matrix_data[i], annot=True, fmt='g', ax=ax_cm, cmap='Blues')
    ax_cm.set_title(f'Fold {i+1} Confusion Matrix')
    ax_cm.set_xlabel('Predicted Labels')
    ax_cm.set_ylabel('True Labels')

# Calculate average values
avg_train_accuracy /= 10
avg_validation_accuracy /= 10
avg_train_loss /= 10
avg_validation_loss /= 10

plt.tight_layout()
plt.show()

# Plotting average accuracy and loss
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Average Accuracy
ax1.plot(avg_train_accuracy, 'o-c', label='Average Train Accuracy')  # 'o-c' for circle markers with cyan color
ax1.plot(avg_validation_accuracy, 'o-m', label='Average Validate Accuracy')  # 'o-m' for circle markers with magenta color
ax1.set_title('Average Accuracy Across Folds')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Average Loss
ax2.plot(avg_train_loss, 'o-', label='Average Train Loss')  # 'o-r' for circle markers with red color
ax2.plot(avg_validation_loss, 'o-', label='Average Validate Loss')  # 'o-p' for circle markers with purple color
ax2.set_title('Average Loss Across Folds')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# Print classification reports
for i, report in enumerate(classification_reports):
    print(f"Classification Report for Fold {i+1}:\n{report}\n")

# Initialize dictionaries to store total metrics for averaging
total_metrics = {
    'Precision': [0] * len(classification_reports[0]),
    'Recall': [0] * len(classification_reports[0]),
    'F1-Score': [0] * len(classification_reports[0])
}

# Accumulate sums for each metric
for report in classification_reports:
    total_metrics['Precision'] += report['Precision']
    total_metrics['Recall'] += report['Recall']
    total_metrics['F1-Score'] += report['F1-Score']

# Number of folds
num_folds = len(classification_reports)

# Calculate the average for each metric
avg_metrics = {
    metric: [total / num_folds for total in totals]
    for metric, totals in total_metrics.items()
}

# Create a DataFrame for the average report
avg_classification_report = pd.DataFrame(avg_metrics, index=classification_reports[0].index)

# Print the average classification report
print(f"Average Classification Report Across All Folds:\n{avg_classification_report}\n")