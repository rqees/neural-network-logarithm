# neural-network-logarithm
A neural network which can solve logarithms! I created this project for a first-year computational biology class (CSB195) at UofT. The report created for this project is below.

Report 2
========

Authors: Raees Kabir [✉](mailto:r.kabir@mail.utoronto.ca) , ChatGPT-4o

11 November 2024

CSB195

Objective
---------

My aim is to create a simple neural network in the R to approximate the natural logarithmic function. This task was approached using a feedforward neural network with two hidden layers and a backpropagation learning algorithm.

Model
-----

### Architecture

The neural network consists of:

-   Input Layer: One input node representing the input value x (the value for which the logarithm is calculated)

-   First Hidden Layer: 100 neurons, applying the hyperbolic tangent (tanh) activation function.

-   Second Hidden Layer: 50 neurons, also using the tanh activation function.

-   Output Layer: One neuron that outputs the predicted value of ln(x).

The architecture was chosen to balance between simplicity and the ability to capture the non-linear relationship between x and ln(x). I also tested out the ReLU and sigmoid functions but the results were not as accurate.

### Training Data

Training data was generated by creating a sequence of exponentially spaced values from 0 to 5 and taking the natural logarithm of these values This range ensures the model is exposed to both smaller and larger values, providing a well-rounded dataset for learning.

### Loss Function

This model uses the squared error loss function, which computes the square of the difference between the predicted logarithmic value and the actual logarithmic value for each sample.

### Training Process with Mini-Batches

Inspired by a technique I learned about called "stochastic gradient descent", I trained the model using mini-batches. The training data is split into smaller batches to improve the efficiency of the learning process.

-   Mini-Batch Size: The mini-batch size was set to 10. This means that for each update, 10 random samples are selected from the training data, and the weights are updated based on the average error across these 10 samples.

-   Epochs: The entire training dataset is passed through the network 1000 times.

-   Shuffling: The data is shuffled at the beginning of each epoch to prevent the model from learning patterns based solely on the order of the data.

During each epoch, the mini-batches are processed sequentially, and the weights are updated after every batch. The loss for each batch is accumulated and used to track the model's performance over time.  

### Backpropagation and Learning

The model uses backpropagation with mini-batch gradient descent to minimize the mean squared error between the predicted and actual logarithmic values. The gradients are computed for each layer's weights and biases using the chain rule to identify the sensitivity of each weight and bias to the error, and the weights and biases are then updated iteratively.

A learning rate is implemented with a value of 0.0001 to smoothly locate optimal values. I had originally also implemented a decay rate to gradually reduce the learning rate over time. However, I did not see a significant change.

Observations
------------

Throughout the training process, the loss is tracked and printed every epoch to monitor the model's progress

A summary of the tracked loss is shown below with set.seed(46)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeYXZsqcqoAYc3ikC1XqeC7BNk2VRKpo8vorYefg2bfawStNoDDGQQ9ii2RT-sqfZkw4_OuEWInibVMitxXWSwgXtZB40RBnWWT84G3apbcM7N6Koy0Z863-jLHYkUy-RKNDG4xsQ?key=yXk6lC_ZskG_IyNkx6jkhaIC)

*Image produced from data by Gemini*

I found it surprising that the model was able to adjust for the loss so quickly at the start.

Testing
-------

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXeXMPpz4IM8Z7WZElJ4G4v2VL2rEPWcmsZDGwJfxK-EplJHrFfueuZQ1dnWm0xTHbnloBc74nWyw_Xsy9G3GkG6X9cBN0spz-Re7_RxRqitr9QkEcq9HeVvatoV50jyKf0sqTROWw?key=yXk6lC_ZskG_IyNkx6jkhaIC)

The mean absolute error (MAE) is calculated by:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| \text{Predicted}_i - \text{Actual}_i \right|$$

Where:

*   n is the number of data points.
*   Predicted<sub>i</sub> is the predicted value for the i-th data point.
*   Actual<sub>i</sub> is the actual value for the i-th data point.

From the above tested dataset, the mean absolute error is 0.0434, which is quite impressive for such a basic model.

The model, however, struggles with values that are significantly different from its trained data, such as very large and small numbers.

For example:

| Test Value | Predicted Log | Actual Log |
|---|---|---|
| 500 | 5.530542 | 6.214608 |

Where the percent error is approximately 11.04%, calculated by:

$$\text{Percent Error} = \left( \frac{|\text{Predicted} - \text{Actual}|}{\text{Actual}} \right) \times 100$$

Conclusion
----------

The neural network successfully approximates the natural logarithm function, achieving a Mean Absolute Error (MAE) of 0.0434 on test data. The model performed well for values within the training range, but struggled with extreme values, showing a percent error of 11.04% for x = 500. While the model is effective for typical inputs, expanding the training data range could improve generalization to out-of-range values.

Appendix 1: ChatGPT Conversation
--------------------------------

<https://chatgpt.com/share/67344e59-30c4-8001-b5cc-cd6a15bd05f2>

<https://chatgpt.com/share/67344e22-45c8-8001-9bd1-404ab9d10f69>

<https://chatgpt.com/share/67344d62-5804-8003-ab1e-ceab373f6cb7>

[END]
