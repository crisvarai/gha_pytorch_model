import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == "__main__":

    # Parameters
    inputDim = 1        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    learningRate = 0.01 
    epochs = 100

    # Create dummy data for training
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    # Model
    model = linearRegression(inputDim, outputDim)
    if torch.cuda.is_available():   # For GPU
        model.cuda()

    # Loss function and Optimizer
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    # Training
    for epoch in range(1, epochs+1):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, 
        # dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        logging.info('Epoch {} Loss {}'.format(epoch, loss.item()))

    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        # logging.info(predicted)

    # MAE and MSE Metric
    mae = mean_absolute_error(y_train, predicted)
    mse = mean_squared_error(y_train, predicted)
    logging.info('Mean Absolute Error {}, Mean Squared Error {}'.format(mae, mse))

    # plt.clf()
    # plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    # plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    # plt.legend(loc='best')
    # plt.show()