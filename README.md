# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.

In many real-world applications—such as house price prediction, temperature forecasting, sales estimation, or demand prediction—the relationship between input variables and the output is non-linear and complex. Traditional statistical models often fail to capture these patterns effectively. Deep Learning models, particularly Artificial Neural Networks (ANNs), are capable of learning such complex relationships through multiple hidden layers and non-linear activation functions.

In this experiment, a dataset containing multiple independent variables (features) and a dependent variable (target) is provided. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand how network architecture, learning rate, number of epochs, and activation functions affect the accuracy of regression predictions and to demonstrate the effectiveness of deep learning in solving regression problems.
## Neural Network Model

<img width="1170" height="777" alt="image" src="https://github.com/user-attachments/assets/c83e0188-4884-4dc3-821a-b447668afd92" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Austin Aro A

### Register Number: 212224040038

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/DL1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: RESHMA R
# Register Number: 212224040274
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.Adam(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())



        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information

<img width="166" height="431" alt="Screenshot 2026-07-25 172423" src="https://github.com/user-attachments/assets/f84bbe9d-814f-45c5-8609-fa15276049e3" />


### OUTPUT

<img width="146" height="310" alt="Screenshot 2026-07-25 173419" src="https://github.com/user-attachments/assets/4bfaa549-5e02-461e-9af8-3e7fb433d8d4" />


<img width="228" height="385" alt="Screenshot 2026-07-25 172438" src="https://github.com/user-attachments/assets/36d11040-0b61-4660-9464-4a6777625892" />





### Training Loss Vs Iteration Plot

<img width="607" height="469" alt="Screenshot 2026-07-25 172448" src="https://github.com/user-attachments/assets/4c5d9dbd-8480-4d5a-b3b4-fa13885bedfc" />



### New Sample Data Prediction

<img width="792" height="208" alt="Screenshot 2026-07-25 172830" src="https://github.com/user-attachments/assets/0daff0bc-a0e9-4f07-b2fc-19008b1746c5" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
