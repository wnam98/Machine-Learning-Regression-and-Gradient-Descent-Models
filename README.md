# Implementing Stochastic and Batch Gradient Descent for a GLM Second Order Polynomial Regression Model 

## Introduction 

Given an initial set of training data: {[1,0], 1}, {[2,1], 0}, {[3,2], 2} and {[2,6], 1} the code above utilizes both psuedo inverse and gradient descent to predict the y-output for testing data x = [1,1]. The data plot shows that a basic linear regression model wouldn't adequetely fit the data, and thus a second order polynomial regression model was used instead. This model contains six basis equations and requires the calculation of six coefficients or "weights" to be crossed with the input phi matrix. 

### Simple First Order Linear Regression

First order regression consists of the sum of the bias term and the dot product of the weight and training feature vectors.
<img src="https://www.latex4technics.com/l4ttemp/r9qvo5.png?1586181594021" /> 

### Second Order Polynomial Regression

Second order regression consists of six basis equations along with the bias term. From the given training data, this model yields a 6 x 4 input correlation matrix.
<br/>
<img src="https://www.latex4technics.com/l4ttemp/r9qvo5.png?1586182059773" /> 

### Pseudo Inverse Method
Left Inverse
<br/>
<img src="https://www.latex4technics.com/l4ttemp/r9qvo5.png?1586182702985" /> 

Right Inverse

### Gradient Descent

### Input Correlation

### Cross Correlation

### Mean Squared Error

### Weight Update

## Running the script in your terminal

For MacOS/Linux users, cd into the directory containing the script and enter the following command:
```
python3 data.py
```
or for Windows:
```
python data.py
```

## Tools/Frameworks

* [Jupyter Notebook](https://jupyter.org/) - The web environment used to write the python code
* [Numpy](https://numpy.org/) - Third party framework for linear algebra/vector calculations
* [Matplotlib](https://matplotlib.org/) - Python library for graphing and visualizations

## Authors

* **Walter Nam** - *Initial work* - [profile](https://github.com/wnam98)
