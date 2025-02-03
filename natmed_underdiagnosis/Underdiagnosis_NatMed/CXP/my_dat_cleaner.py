import pandas as pd
import os

SOURCE_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid.csv"
CLEANED_CSV_PATH = "/workspace/persistent/code/MEDFAIR/chexpert/chexpertchestxrays-u20210408/train_cheXbert-myCopy1_with_valid_clean.csv"


def clean_it():
    data = pd.read_csv(SOURCE_CSV_PATH)

    print("sample entries from csv file: ")
    print(data.head)

    # replace nan with 0.0
    data.fillna(0.0, inplace=True)
    
    # replace non-positive cases (0.0, -1.0, NaN) with 0.0 in numeric columns
    numeric_columns = data.select_dtypes(include='number').columns
    data[numeric_columns] = data[numeric_columns].apply(lambda x: x.where(x >= 1.0, 0.0))
    
    # investigate anomalous entries in columns
    print("anomalous 'Sex' column entries: ")
    anomalous_entries = data[~data['Sex'].isin(['Male', 'Female'])]
    print(anomalous_entries)
    
    # remove rows with 'Unknown' value in the 'Sex' column
    data = data[data['Sex'] != 'Unknown']

    print("Anomalous 'Age' column entries: ")
    anomalous_entries = data[~data['Age'].astype(str).str.isdigit()]
    print(anomalous_entries)

    #print("Anomalous 'Atelectasis' column entries: ")
    #anomalous_entries = data[~data['Atelectasis'].astype(str).str.isdigit()]
    #print(anomalous_entries)
    #print("Anomalous 'Cardiomegaly' column entries: ")
    #anomalous_entries = data[~data['Cardiomegaly'].astype(str).str.isdigit()]
    #print(anomalous_entries)
    #print("Anomalous 'Consolidation' column entries: ")
    #anomalous_entries = data[~data['Consolidation'].astype(str).str.isdigit()]
    #print(anomalous_entries)
    #print("Anomalous 'Edema' column entries: ")
    #anomalous_entries = data[~data['Edema'].astype(str).str.isdigit()]
    #print(anomalous_entries)
    #print("Anomalous 'Pleural Effusion' column entries: ")
    #anomalous_entries = data[~data['Pleural Effusion'].astype(str).str.isdigit()]
    #print(anomalous_entries)
    
    print("Anomalous 'Atelectasis' column entries: ")
    anomalous_entries = data[(data['Atelectasis'] != 0) & (data['Atelectasis'] != 1)]
    print(anomalous_entries)

    print("Anomalous 'Cardiomegaly' column entries: ")
    anomalous_entries = data[(data['Cardiomegaly'] != 0) & (data['Cardiomegaly'] != 1)]
    print(anomalous_entries)

    print("Anomalous 'Consolidation' column entries: ")
    anomalous_entries = data[(data['Consolidation'] != 0) & (data['Consolidation'] != 1)]
    print(anomalous_entries)

    print("Anomalous 'Edema' column entries: ")
    anomalous_entries = data[(data['Edema'] != 0) & (data['Edema'] != 1)]
    print(anomalous_entries)

    print("Anomalous 'Pleural Effusion' column entries: ")
    anomalous_entries = data[(data['Pleural Effusion'] != 0) & (data['Pleural Effusion'] != 1)]
    print(anomalous_entries)


    # Define age ranges
    age_ranges = [(0, 20), (20, 40), (40, 80), (80, float('inf'))]
    labels = ['0-20', '20-40', '40-80', '80+']

    # add new column 'Age Group' mapping age to age ranges
    data['Age Group'] = pd.cut(data['Age'], bins=[0, 20, 40, 80, float('inf')], labels=labels, right=False)    

    # final check print before saving
    print("sample entries from cleaned csv file: ")
    print(data.head)
    
    
    #data.to_csv(CLEANED_CSV_PATH, index=False)

    print("Data cleaned and saved to: ", CLEANED_CSV_PATH)


if __name__ == "__main__":
    clean_it()

