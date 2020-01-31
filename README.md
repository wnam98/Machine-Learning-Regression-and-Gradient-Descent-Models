# Regression-and-Gradient-Descent-Models
Pseudo-Inverse vs Gradient Descent weight calculations

3rd Party python modules were limited to numpy, matplotlib, and pandas. This model predicts y-values given several multivariate training data sets, with the regression model of choice being 2nd order polynomial. Despite only being given four instances, we had to calculate six coefficients, or "weights" using the appropriate basis equations for our phi matrix. Due to the dimensions or our polynomial model matrix, we were unable to utilize left-inverse, as our singular matrix would not be invertable. Since we had row-independence, we used right-inverse, which yielded our 1 x 6 weight vector. These are the ideal weights, whose dot product with the phi matrix from the x1 and x2 inputs would yield the predicted y value. We then compared this y-value with our Gradient Descent weight vectors. Note that feature scaling was unnecessary, and convergence was observed for iteration values >= 10000. 
