# Set random seed for reproducibility
set.seed(46)

# Define the network structure
inputSize <- 1   # One input (x value)
hiddenSize <- 100  # Number of hidden neurons
outputSize <- 1  # One output (log(x))

# Initialize weights randomly
weightsInputHidden <- matrix(runif(inputSize * hiddenSize, -1, 1), nrow = hiddenSize, ncol = inputSize)
biasHidden <- runif(hiddenSize, -1, 1)
weightsHiddenOutput <- runif(hiddenSize, -1, 1)
biasOutput <- runif(1, -1, 1)

# Activation function - tanh
tanh <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

# tanh derivative for backpropagation
tanh_derivative <- function(x) {
  1 - tanh(x)^2
}

# Forward pass function
forward <- function(x) {
  hiddenInput <- weightsInputHidden %*% x + biasHidden
  hiddenOutput <- tanh(hiddenInput)
  output <- sum(hiddenOutput * weightsHiddenOutput) + biasOutput
  return(list(output = output, hiddenOutput = hiddenOutput))
}

# Define learning rate
learningRate <- 0.0001  # Increase learning rate for faster convergence

# Backpropagation function
backpropagation <- function(x, y) {
  forwardPass <- forward(x)
  predicted <- forwardPass$output
  hiddenOutput <- forwardPass$hiddenOutput

  # Calculate the error
  error <- predicted - y

  # Gradients for output layer
  dOutput <- error  # Since we have linear output, no activation derivative

  # Gradients for hidden layer
  dHidden <- dOutput * weightsHiddenOutput * (1 - hiddenOutput^2)  # Apply chain rule for tanh derivative

  # Update weights and biases using gradient descent
  weightsHiddenOutput <<- weightsHiddenOutput - learningRate * dOutput * hiddenOutput
  biasOutput <<- biasOutput - learningRate * dOutput
  weightsInputHidden <<- weightsInputHidden - learningRate * dHidden %*% t(x)
  biasHidden <<- biasHidden - learningRate * dHidden
}

# Generate training data using logarithmic spacing for better coverage
trainData <- exp(seq(0, 2, length.out = 1000))  # Exponentially spaced data to cover more range
trainLabels <- log(trainData)  # Labels: natural log of input data

# Training loop
epochs <- 1000  # Number of epochs
for (epoch in 1:epochs) {
  totalLoss <- 0
  for (i in 1:length(trainData)) {
    x <- matrix(trainData[i], nrow = inputSize)  # Reshape input for matrix multiplication
    y <- trainLabels[i]  # Target value (log(x))
    backpropagation(x, y)  # Update weights and biases
    prediction <- forward(x)$output  # Get prediction
    totalLoss <- totalLoss + (prediction - y)^2  # Squared error loss
  }

  # Print progress every 10 epochs
  if (epoch) {
    cat("Epoch:", epoch, "Loss:", totalLoss / length(trainData), "\n")
  }
}

# Testing the network with more diverse test data
testData <- 50 # Test data: More diverse range
testPredictions <- sapply(testData, function(x) forward(matrix(x, nrow = inputSize))$output)
actualLog <- log(testData)  # Actual log values

# Display results
data.frame(TestValue = testData, PredictedLog = testPredictions, ActualLog = actualLog)
