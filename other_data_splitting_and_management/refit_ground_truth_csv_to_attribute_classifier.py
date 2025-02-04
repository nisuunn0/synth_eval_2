import pandas as pd


#input_csv = './ground_truth.csv'  # Replace with the actual path to your file
input_csv = './split_test_for_attribute_classifier/test.csv'
#output_csv = './ground_truth_refit/refit_for_attribute_classifier_ground_truth.csv'  # The new CSV file
#output_csv = './test_refit/refit_for_attribute_classifier_test.csv'
output_csv = './test_refit_latest/refit_for_attribute_classifier_test.csv'

df = pd.read_csv(input_csv)

# Set halfway point to only process the first half of the CSV
#halfway_point = len(df) // 2
#df = df.iloc[:halfway_point]



df['Sex bin'] = df['Sex'].apply(lambda x: 0 if x == 'Female' else 1)


df['frontlat bin'] = df['Frontal/Lateral'].apply(lambda x: 0 if x == 'Frontal' else 1)


df['Age Group 0-20'] = df['Age Group'].apply(lambda x: 1 if x == '0-20' else 0)
df['Age Group 20-40'] = df['Age Group'].apply(lambda x: 1 if x == '20-40' else 0)
df['Age Group 40-80'] = df['Age Group'].apply(lambda x: 1 if x == '40-80' else 0) # ADDED BY ME LATER!!!
df['Age Group 80+'] = df['Age Group'].apply(lambda x: 1 if x == '80+' else 0)


df['AP'] = df['AP/PA'].apply(lambda x: 1 if x == 'AP' else 0)
df['PA'] = df['AP/PA'].apply(lambda x: 1 if x == 'PA' else 0)


df.to_csv(output_csv, index=False)



