# GLM Second Order Polynomial Regression with Pseudo Inverse and Gradient Descent Weight Vectors

## Introduction 

Given an initial set of training data: {[1,0], 1}, {[2,1], 0}, {[3,2], 2} and {[2,6], 1} the code above utilizes both psuedo inverse and gradient descent to predict the y-output for testing data x = [1,1]. The data plot shows that a basic linear regression model wouldn't adequetely fit the data, and thus a second order polynomial regression model was used instead. This model contains six basis equations and requires the calculation of six coefficients or "weights" to be crossed with the input phi matrix. Gradient descent iterations vs loss function graphs are also provided for both batch and stochastic optimization.

### Simple First Order Linear Regression

First order regression consists of the sum of the bias term and the dot product of the weight and training feature vectors.
<br/>
<br/>
![linear_reg](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/linear_reg.png "linear_reg")

### Second Order Polynomial Regression

Second order regression consists of six basis equations along with the bias term. For multivariate least square estimation, the input phi matrix yields k + 1 basis vectors as the number of columns, and n training instances as the number of rows. From the given training data, our phi dimensions should be 6 x 4.
<br/>
<br/>
![order2](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/order2.png "order2")

### Regression with Neural Networks

Regression from the point of view of a Neural Network with two hidden layers. Each perceptron takes in a weighted sum depending on the strengths of the connections between the layers. 
<br/>
<br/>
![nn](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/nn.png "nn")

## Pseudo Inverse Method
In order to find the optimal weight vector, we need to derive the minimum of the error function. The above model is an example of multivariate least squares estimation where the objective function can be expressed as:
<br/>
<br/>
![objective](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/objective.png "objective")
<br/>
<br/>
The partial derivative of the objective function with respect to the weight vector is expressed below:
<br/>
<br/>
![partial_derivative](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/partial_derivative.png "partial_derivative")
<br/>
<br/>
which yields the following:
<br/>
<br/>
![left_inverse_expression](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/left_inverse_expression.png "left_inverse_expression")
<br/>
<br/>
After setting the expression equal to 0, we have derived the pseudo inverse weight vector:
<br/>
<br/>
![left_inverse](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/left_inverse.png "left_inverse") 
<br/>
<br/>
The columns of our phi matrix are dependent and is thus uninvertible. The pseudo inverse weight vector for the training data was calculated using the alternate right inverse for row independence.
<br/>
<br/>
![right_inverse](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/right_inverse.png "right_inverse") 
<br/>
<br/>

## Gradient Descent

### Mean Squared Error

The error for our training data is calculated as the difference between the desired and computed output. For a large batch of data as in our phi matrix, we can introduce mean squared error by taking the expectation of the square of the training data error:
<br/>
<br/>
![mse](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/mse.png "mse") 
<br/>
<br/>
Because of the multivariate nature of the training data, the graph of the MSE equation represents a bowl shaped function in x1 and x2 cross space. The optimal weight vector is represented by the global minimum of this error function.
![mse_graph](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/mse_graph.PNG "mse_graph")
The gradient descent algorithm finds the optimal minimum by traversing through the error function a set number of times (denoted by pmax iterations). At each iteration, an initialized weight vector is updated by the sum of its previous iteration value and a specified learning rate multiplied by the gradient of the MSE function. The gradient is defined as the first-order derivative of the error function with respect to the weight vector. 
<br/>
<br/>
![weight_update](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/weight_update.png "weight_update")
<br/>
<br/>
where p = number of iterations, i = vector dimensions, and eta = the learning rate.

### Batch Gradient Descent

Batch gradient descent calculates the exact input and cross correlations at each iteration using the entire batch of given training data. The following is derived from differentiating the MSE objective function. Note that P and R remain fixed at each iteration.
<br/>
<br/>
![gradient](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/gradient.png "gradient")
<br/>
<br/>
Where P denotes the cross correlation matrix and R denotes the input correlation matrix. Batch gradient descent is rather computationally complex for large datasets because P and R calculations are too costly for high dimension feature vectors. However, since BGD calculates true gradient at each iteration, convergence is generally reached in a fewer number of iterations. 

### Stochastic Gradient Descent

Stochastic gradient descent estimates the gradient using a randomly selected subset of the data at each iteration. Thus, R and P are also estimated at each iteration until the long term average of the locally optimized weight vector approaches the true optimal solution. The weight update equation is expressed below:
<br/>
<br/>
![sgd_weight_update](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/sgd_weight_update.png "sgd_weight_update")
<br/>
<br/>
Where d and s respectively denote desired vs computed outputs. Stochastic gradient descent significantly reduces the computational complexity during weight training, but generally takes a larger number of iterations to reach convergence of an optimal minimum. For deep learning neural networks, stochastic is generally the default optimization method. The picture below shows the difference in convergence paths for SGD vs BGD
![sgdvsbgd](https://raw.github.com/wnam98/Machine-Learning-Regression-and-Gradient-Descent-Models/master/imgs/sgdvsbgd.png "sgdvsbgd")

## Running the script in your terminal

This script is only compatible with Python 3. Your system should preferably have version 3.5 or above. 
For MacOS/Linux users, cd into the directory containing the script and enter the following command:
```
python3 data.py
```
or for Windows:
```
python data.py
```
Make sure your Python 3 interpreter is properly configured in your environment variables.

## Tools/Frameworks

* [Jupyter Notebook](https://jupyter.org/) - Web environment
* [Numpy](https://numpy.org/) - Third party library for linear algebra/vector calculations
* [Matplotlib](https://matplotlib.org/) -  Third party library for graphing and visualizations

## Author

* **Walter Nam** - *Initial work* - [profile](https://github.com/wnam98)
