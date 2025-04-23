set.seed(46)

inputSize <- 1    # One input (x value)
hiddenSize1 <- 100  # First hidden layer size
hiddenSize2 <- 50   # Second hidden layer size
outputSize <- 1     # One output (log(x))

# Initialize weights randomly
weightsInputHidden1 <- matrix(runif(inputSize * hiddenSize1, -1, 1), nrow = hiddenSize1, ncol = inputSize)
biasHidden1 <- runif(hiddenSize1, -1, 1)
weightsHidden1Hidden2 <- matrix(runif(hiddenSize1 * hiddenSize2, -1, 1), nrow = hiddenSize2, ncol = hiddenSize1)
biasHidden2 <- runif(hiddenSize2, -1, 1)
weightsHiddenOutput <- runif(hiddenSize2, -1, 1)
biasOutput <- runif(1, -1, 1)

# Activation function - tanh
tanh <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

# Tanh derivative for backpropagation
tanh_derivative <- function(x) {
  1 - tanh(x)^2
}

# Forward pass function with batch normalization
forward <- function(x) {
  hiddenInput1 <- weightsInputHidden1 %*% x + biasHidden1
  hiddenOutput1 <- tanh(hiddenInput1)

  hiddenInput2 <- weightsHidden1Hidden2 %*% hiddenOutput1 + biasHidden2
  hiddenOutput2 <- tanh(hiddenInput2)

  output <- sum(hiddenOutput2 * weightsHiddenOutput) + biasOutput
  return(list(output = output, hiddenOutput1 = hiddenOutput1, hiddenOutput2 = hiddenOutput2))
}

# Learning rate settings for schedule
initialLearningRate <- 0.0001

# Backpropagation with mini-batch support and learning rate schedule
backpropagation <- function(batchX, batchY, learningRate) {
  gradientWeightsHiddenOutput <- rep(0, length(weightsHiddenOutput))
  gradientBiasOutput <- 0
  gradientWeightsHidden1Hidden2 <- matrix(0, nrow = hiddenSize2, ncol = hiddenSize1)
  gradientBiasHidden2 <- rep(0, hiddenSize2)
  gradientWeightsInputHidden1 <- matrix(0, nrow = hiddenSize1, ncol = inputSize)
  gradientBiasHidden1 <- rep(0, hiddenSize1)

  for (i in 1:ncol(batchX)) {
    x <- batchX[, i, drop = FALSE]
    y <- batchY[i]

    forwardPass <- forward(x)
    predicted <- forwardPass$output
    hiddenOutput1 <- forwardPass$hiddenOutput1
    hiddenOutput2 <- forwardPass$hiddenOutput2

    # Calculate the error
    error <- (predicted - y)^2

    # Gradients for output layer
    dOutput <- 2*(predicted - y)

    # Gradients for hidden layer 2
    dHidden2 <- dOutput * weightsHiddenOutput * (1 - hiddenOutput2^2)

    # Gradients for hidden layer 1
    dHidden1 <- t(weightsHidden1Hidden2) %*% dHidden2 * (1 - hiddenOutput1^2)

    # Accumulate gradients
    gradientWeightsHiddenOutput <- gradientWeightsHiddenOutput + dOutput * hiddenOutput2
    gradientBiasOutput <- gradientBiasOutput + dOutput
    gradientWeightsHidden1Hidden2 <- gradientWeightsHidden1Hidden2 + dHidden2 %*% t(hiddenOutput1)
    gradientBiasHidden2 <- gradientBiasHidden2 + dHidden2
    gradientWeightsInputHidden1 <- gradientWeightsInputHidden1 + dHidden1 %*% t(x)
    gradientBiasHidden1 <- gradientBiasHidden1 + dHidden1
  }

  # Update weights and biases
  batchSize <- ncol(batchX)
  weightsHiddenOutput <<- weightsHiddenOutput - learningRate * (gradientWeightsHiddenOutput / batchSize)
  biasOutput <<- biasOutput - learningRate * (gradientBiasOutput / batchSize)
  weightsHidden1Hidden2 <<- weightsHidden1Hidden2 - learningRate * (gradientWeightsHidden1Hidden2 / batchSize)
  biasHidden2 <<- biasHidden2 - learningRate * (gradientBiasHidden2 / batchSize)
  weightsInputHidden1 <<- weightsInputHidden1 - learningRate * (gradientWeightsInputHidden1 / batchSize)
  biasHidden1 <<- biasHidden1 - learningRate * (gradientBiasHidden1 / batchSize)
}

# Generate training data with larger range and better coverage near zero
trainData <- exp(seq(-5, 5, length.out = 1000))  # Exponentially spaced data with larger range
trainLabels <- log(trainData)  # Labels: natural log of input data

# Define mini-batch size
batchSize <- 10

# Training loop with decreasing learning rate
epochs <- 1000
for (epoch in 1:epochs) {
  # Update learning rate based on decay
  learningRate <- initialLearningRate

  totalLoss <- 0
  indices <- sample(length(trainData))  # Shuffle data indices for each epoch

  # Process each mini-batch
  for (batchStart in seq(1, length(trainData), by = batchSize)) {
    batchIndices <- indices[batchStart:min(batchStart + batchSize - 1, length(trainData))]
    batchX <- matrix(trainData[batchIndices], nrow = inputSize, ncol = length(batchIndices))
    batchY <- trainLabels[batchIndices]

    backpropagation(batchX, batchY, learningRate)  # Update weights and biases based on mini-batch

    # Compute loss for tracking
    for (i in 1:length(batchIndices)) {
      prediction <- forward(batchX[, i, drop = FALSE])$output
      totalLoss <- totalLoss + (prediction - batchY[i])^2  # Squared error loss
    }
  }

  # Print progress every 10 epochs
  if (epoch %% 10 == 0) {
    cat("Epoch:", epoch, "Loss:", totalLoss / length(trainData), "\n")
  }
}

testData <- seq(0.1, 70, by=0.6)
actualLog <- log(testData)

testPredictions <- sapply(testData, function(x) forward(matrix(x, nrow = inputSize))$output)

plot(testData, actualLog, type = "l", col = "blue", lwd = 2, ylab = "Log Value", xlab = "Input", main = "Predicted vs. Actual Log")
points(testData, testPredictions, col = "red", pch = 16)
legend("topleft", legend = c("Actual Log", "Predicted Log"),
       col = c("blue", "red"),
       lty = c(1, 0),
       pch = c(NA, 16))

data.frame(TestValue = testData, PredictedLog = testPredictions, ActualLog = actualLog)

