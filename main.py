import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test)

# Define the model
class NaiveBayesModel(nn.Module):
    def __init__(self):
        super(NaiveBayesModel, self).__init()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = NaiveBayesModel()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

# Predicting for age=30, estimated salary=87000
test_data = torch.FloatTensor(sc.transform([[30, 87000]]))
predicted = model(test_data)
print(torch.round(predicted).item())

# Predicting for test data
model.eval()
test_outputs = model(X_test_tensor)
y_pred_torch = torch.round(test_outputs).detach().numpy().astype(int)
print(np.concatenate((y_pred_torch.reshape(len(y_pred_torch), 1), Y_test.reshape(len(y_pred_torch), 1)), 1))

# Confusion Matrix using PyTorch
conf_matrix_torch = torch.tensor(confusion_matrix(Y_test, y_pred_torch))
print(conf_matrix_torch)

# Accuracy Score using PyTorch
accuracy_torch = accuracy_score(Y_test, y_pred_torch)
print(accuracy_torch)

# Plotting using Matplotlib
import matplotlib.pyplot as plt

# Confusion Matrix Plot using Seaborn
import seaborn as sns

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_torch, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
