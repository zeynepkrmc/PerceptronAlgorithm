import pandas as pd 
import numpy as np 

def load_data(filename): 
    train_data = pd.read_excel(filename, sheet_name='TRAINData') 
    test_data = pd.read_excel(filename, sheet_name='TESTData') 
    return train_data, test_data 

# Perceptron Learning Algorithm 
def perceptron_train(X_train, y_train, learning_rate=0.1, max_iter=1000): 
    weights = np.zeros(X_train.shape[1]) 
    for _ in range(max_iter): 
        for i in range(len(y_train)): 
            prediction = np.dot(X_train[i], weights) 
            # if misclassified 
            if y_train[i] * prediction <= 0:   
                weights += learning_rate * y_train[i] * X_train[i] 
    
    return weights 

def perceptron_predict(X_test, weights): 
    predictions = np.dot(X_test, weights) 
    return np.where(predictions >= 0, 1, -1) 

 

def preprocess_data(data): 
    X = data.drop(columns=['SubjectID', 'Class']).values 
    y = data['Class'].apply(lambda x: 1 if x == 4 else -1).values   
    X = np.c_[np.ones(X.shape[0]), X] 
    
    return X, y 

def main(): 
    filename = 'DataForPerceptron.xlsx' 
    train_data, test_data = load_data(filename) 

    X_train, y_train = preprocess_data(train_data) 
    X_test, _ = preprocess_data(test_data)   

    weights = perceptron_train(X_train, y_train) 
    predictions = perceptron_predict(X_test, weights) 

    # Map 
    y_predMap = np.where(predictions == -1 ,2, 4) 
    test_data['Predicted Class'] = y_predMap 
    test_data.to_excel('TestDataWithPredictions.xlsx', index=False) 

    print("Predictions saved to 'TestDataWithPredictions.xlsx'.") 
    print(test_data[['SubjectID', 'Predicted Class']]) 

if __name__ == "__main__": 
    main() 