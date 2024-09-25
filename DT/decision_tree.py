import pandas as pd
import numpy as np
import random
from math import log2

def generate_dataset(num_samples):
    genders = ['M', 'F']
    ages = ['young', 'adult', 'senior']
    countries = ['USA', 'Canada', 'Germany', 'France', 'India', 'Japan']
    apps = ['atom count', 'check mate mate', 'beehive finder', 'puzzle master', 'code game']

    data = {
        'Gender': [random.choice(genders) for _ in range(num_samples)],
        'Age': [random.choice(ages) for _ in range(num_samples)],
        'Country': [random.choice(countries) for _ in range(num_samples)],
        'App': [random.choice(apps) for _ in range(num_samples)]
    }
    
    return pd.DataFrame(data)

dataset = generate_dataset(100)

def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    H = 0
    for count in counts:
        p = count / len(labels)
        H -= p * log2(p) if p > 0 else 0  # prevent log(0)
    return H

def information_gain(data, labels, feature):
    values = np.unique(data[feature])
    H = entropy(labels)
    H_feature = 0
    for value in values:
        sub_labels = labels[data[feature] == value]
        H_feature += (len(sub_labels) / len(labels)) * entropy(sub_labels)
    IG = H - H_feature
    return IG

H = entropy(dataset['App'])

features = ['Gender', 'Age', 'Country']
information_gains = {feature: information_gain(dataset, dataset['App'], feature) for feature in features}

print(f'Entropy: {H}')
for feature, IG in information_gains.items():
    print(f'Information Gain - {feature}: {IG}')

best_split = max(information_gains, key=information_gains.get)
print(f'Best feature to split: {best_split}')

def classify_customers(new_customers, best_split, dataset):
    predictions = []
    
    for customer in new_customers:
        customer_data = {col: val for col, val in zip(dataset.columns[:-1], customer)}
        if best_split == 'Gender':
            if customer_data['Gender'] == 'F':
                prediction = 'check mate mate'
            else:
                prediction = 'atom count'
        elif best_split == 'Age':
            if customer_data['Age'] == 'young':
                prediction = 'atom count'
            elif customer_data['Age'] == 'adult':
                prediction = 'beehive finder'
            else:
                prediction = 'puzzle master'
        elif best_split == 'Country':
            if customer_data['Country'] == 'USA' or customer_data['Country'] == 'Canada':
                prediction = 'code game'
            else:
                prediction = 'puzzle master'
        predictions.append(prediction)
    
    return predictions

new_customers = [
    ['F', 'young', 'USA'],
    ['M', 'adult', 'France'],
    ['F', 'senior', 'India']
]

predictions = classify_customers(new_customers, best_split, dataset)

for i, prediction in enumerate(predictions):
    print(f'Customer {i+1}: {prediction}')
