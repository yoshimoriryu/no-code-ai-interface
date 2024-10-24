import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'uploaded_files/wine.csv'
data = pd.read_csv(file_path)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and validation datasets to CSV files
train_data.to_csv('wine_training.csv', index=False)
validation_data.to_csv('wine_validation.csv', index=False)

print("Training and validation datasets have been created.")
