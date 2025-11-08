import random
import numpy as np
import matplotlib.pyplot as plt
import math
import util

class Model:
    """
    Abstract class for a machine learning model.
    """
    
    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree = 1, learning_rate = 1e-3):
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = [random.random() for i in range(self.degree + 1)]
 
    def get_features(self, x):
        return [x**p for p in range(self.degree + 1)]

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)

        return sum(w * f for w, f in zip(self.weights, features))

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        return (self.hypothesis(x) - y)**2

    def gradient(self, x, y):
        featuresX = self.get_features(x)
        
        return [2 * (self.hypothesis(x) - y) * featuresX[i] for i in range(self.degree + 1)]
    
    def train(self, dataset : util.Dataset, evalset = None):
        nIterations = 2000
        k = 0
        eval_iters = []
        losses = []
        testLosses = []
        alpha = self.learning_rate/(2 * dataset.get_size())
        max_weight = 1e6

        while k < nIterations:
            randomSample = dataset.get_sample()

            eval_iters.append(k)
            losses.append(dataset.compute_average_loss(self))
            if evalset is not None:
                testLosses.append(evalset.compute_average_loss(self))

            featuresX = self.get_features(randomSample[0])
  
            for j in range(self.degree + 1):
                updateValue = alpha * 2 * dataset.get_size() * (self.hypothesis(randomSample[0]) - randomSample[1])
                self.weights[j] = self.weights[j] - (updateValue * featuresX[j])
                if self.weights[j] > max_weight:
                    self.weights[j] = max_weight
                elif self.weights[j] < -max_weight:
                    self.weights[j] = -max_weight

            k += 1
        
        return [eval_iters, losses, testLosses]


# PA4 Q2
def linear_regression():

    # Helpful functions:
    # util.get_dataset, util.Dataset.compute_average_loss, util.RegressionDataset.plot_data
    # util.RegressionDataset.plot_loss_curve

    # (a)
    sine_train = util.get_dataset("sine_train")
    sine_test = util.get_dataset("sine_test")
    sine_model = PolynomialRegressionModel(1, 1e-4)

    sine_train_result = sine_model.train(sine_train, sine_test)

    print('average loss of sine_train:', sine_train.compute_average_loss(sine_model))
    sine_train.plot_data(sine_model)

    # (b)
    sine_train.plot_loss_curve(sine_train_result[0], sine_train_result[1])

    # (c)
    degrees = [1, 2]
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    sine_val = util.get_dataset("sine_val")
    results = []

    for degree in degrees:
        for lr in learning_rates:
            model = PolynomialRegressionModel(degree, lr)
            model.train(sine_train)

            train_loss = sine_train.compute_average_loss(model)
            val_loss = sine_val.compute_average_loss(model)

            results.append((degree, lr, train_loss, val_loss))
            
    best = min(results, key=lambda x: x[3])

    print('best combination:', f"degree={best[0]}, lr={best[1]:.0e}, train_loss={best[2]:.4f}, val_loss={best[3]:.4f}")

# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate = 1e-2):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = [0 for i in range(self.num_features + 1)]

    def get_features(self, x):
        features = [1] + [z for y in x for z in y]
        return features 

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        sum = 0

        for j in range(self.num_features + 1):
            sum = sum + (self.weights[j] * features[j])

        return 1 / (1 + math.exp(-1 * sum))

    def predict(self, x):
        if self.hypothesis(x) > 0.5:
            return 1
        else:
            return 0

    def loss(self, x, y):
        return (-1) * ((y * math.log(self.hypothesis(x))) + ((1 - y) * math.log((1 - self.hypothesis(x)))))

    def gradient(self, x, y):
        featuresX = self.get_features(x)
        
        return [(y - self.hypothesis(x)) * featuresX[i] for i in range(self.num_features + 1)]

    def train(self, dataset : util.Dataset, evalset : util.Dataset = None):
        nIterations = 2000
        k = 0
        eval_iters = []
        accuracies = []
        testAccuracies = []
        alpha = self.learning_rate/(2 * dataset.get_size())

        while k < nIterations:
            if k % 100 == 0:
                print(f'blr iteration: {k}/{nIterations}')
            randomSample = dataset.get_sample()

            if k % 20 == 0:
                eval_iters.append(k)
                accuracies.append(dataset.compute_average_accuracy(self, 20))
                testAccuracies.append(evalset.compute_average_accuracy(self, 20))

            featuresX = self.get_features(randomSample[0])
  
            for j in range(self.num_features + 1):
                updateValue = alpha * 2 * dataset.get_size() * (self.hypothesis(randomSample[0]) - randomSample[1])
                self.weights[j] = self.weights[j] - (updateValue * featuresX[j])

            k += 1
        
        return [eval_iters, accuracies, testAccuracies, self.weights[1:]]


# PA4 Q4
def binary_classification():

    # Helpful functions:
    # util.Dataset.compute_average_accuracy, util.MNISTDataset.plot_accuracy_curve
    # util.MNISTDataset.plot_confusion_matrix
    # util.MNISTDataset.plot_image
    
    # (a)
    mnist_binary_train = util.get_dataset("mnist_binary_train")
    mnist_binary_test = util.get_dataset("mnist_binary_test")
    mnist_binary_model = BinaryLogisticRegressionModel(28**2)

    mnist_binary_train_result = mnist_binary_model.train(mnist_binary_train, mnist_binary_test)

    mnist_binary_train.compute_average_accuracy(mnist_binary_model)
    mnist_binary_train.plot_accuracy_curve(mnist_binary_train_result[0], mnist_binary_train_result[1])
    mnist_binary_test.plot_accuracy_curve(mnist_binary_train_result[0], mnist_binary_train_result[2])
    
    # (b)
    mnist_binary_train.plot_confusion_matrix(mnist_binary_model)

    # (c)
    mnist_binary_train.plot_image(mnist_binary_train_result[3])

    # (d)
    misclassified = []

    for i in range(mnist_binary_test.get_size()):
        x = mnist_binary_test.xs[i]
        y = mnist_binary_test.ys[i]

        pred = mnist_binary_model.predict(x)

        if pred != y:
            flat = [pixel for row in x for pixel in row]
            misclassified.append(flat)
            if len(misclassified) >= 10:
                break
    
    for error in misclassified:
        mnist_binary_test.plot_image(error)


# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights = [[0 for i in range(self.num_features + 1)] for k in range(num_classes)]

    def get_features(self, x):
        features = [1] + [z for y in x for z in y]
        return features 

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        
        logits = []
        for k in range(self.num_classes):
            logit = sum(self.weights[k][j] * features[j] for j in range(self.num_features + 1))
            logits.append(logit)
        
        max_logit = max(logits)
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        
        sum_exp = sum(exp_logits)
        probabilities = [exp_val / sum_exp for exp_val in exp_logits]
        
        return probabilities

    def predict(self, x):
        predictions = self.hypothesis(x)
        maxValue = max(predictions)

        prediction = predictions.index(maxValue)
        return prediction

    def loss(self, x, y):
        features = self.get_features(x)
        sum = 0
        mainSum = 0
            
        for j in range(self.num_features + 1):
            sum = sum + (self.weights[y][j] * features[j])

        for k in range(self.num_classes):
            classSum = 0

            for j in range(self.num_features + 1):
                classSum = classSum + (self.weights[k][j] * features[j])

            mainSum = mainSum + math.exp(classSum)

        return sum - math.log(mainSum)

    def gradient(self, x, y, probs=None):
        features = self.get_features(x)
        if probs is None:
            probs = self.hypothesis(x)
        
        gradients = []
        for k in range(self.num_classes):
            grad = (self.indicator(y, k) - probs[k])
            gradients.append([grad * feat for feat in features])

        return gradients
        
    def indicator(self, y, k):
        if y == k:
            return 1
        else:
            return 0

    def train(self, dataset : util.Dataset, evalset : util.Dataset = None):
        nIterations = 2000
        k = 0
        eval_iters = []
        accuracies = []
        testAccuracies = []
        alpha = self.learning_rate
        batch_size = 32
        max_weight = 10.0

        while k < nIterations:
            if k % 100 == 0:
                print(f'mlr iteration {k}/{nIterations}')
            
            batch_gradients = [[0 for _ in range(self.num_features + 1)] for _ in range(self.num_classes)]
            
            for _ in range(batch_size):
                sample = dataset.get_sample()
                probs = self.hypothesis(sample[0])
                sample_grads = self.gradient(sample[0], sample[1], probs)
                
                for c in range(self.num_classes):
                    for j in range(self.num_features + 1):
                        batch_gradients[c][j] += sample_grads[c][j]

            for c in range(self.num_classes):
                for j in range(self.num_features + 1):
                    update = alpha * batch_gradients[c][j] / batch_size
                    self.weights[c][j] = max(min(
                        self.weights[c][j] + update, 
                        max_weight
                    ), -max_weight)

            if k % 50 == 0:
                eval_iters.append(k)
                accuracies.append(dataset.compute_average_accuracy(self, 50))
                if evalset is not None:
                    testAccuracies.append(evalset.compute_average_accuracy(self, 50))

            k += 1
        
        feature_weights = [[w for w in class_weights[1:]] for class_weights in self.weights]
        return [eval_iters, accuracies, testAccuracies, feature_weights]

# PA4 Q6
def multi_classification():

    # Helpful functions:
    # util.Dataset.compute_average_accuracy, util.MNISTDataset.plot_accuracy_curve
    # util.MNISTDataset.plot_confusion_matrix
    # util.MNISTDataset.plot_image
    
    mnist_train = util.get_dataset("mnist_train")
    mnist_test = util.get_dataset("mnist_test")
    
    train_subset_size = 1000
    mnist_train.xs = mnist_train.xs[:train_subset_size]
    mnist_train.ys = mnist_train.ys[:train_subset_size]
    
    mnist_model = MultiLogisticRegressionModel(28**2, 10, learning_rate=1e-3)
    mnist_train_result = mnist_model.train(mnist_train, mnist_test)
    
    mnist_train.compute_average_accuracy(mnist_model)
    mnist_train.plot_accuracy_curve(mnist_train_result[0], mnist_train_result[1], title="Training Accuracy")
    
    mnist_test.compute_average_accuracy(mnist_model)
    mnist_test.plot_accuracy_curve(mnist_train_result[0], mnist_train_result[2], title="Test Accuracy")
    
    mnist_test.plot_confusion_matrix(mnist_model)
    
    for digit in range(10):
        mnist_test.plot_image(mnist_train_result[3][digit])

# PA4 Q7 EX
class RidgeRegressionModel(Model):
    """
    Ridge regression model with polynomial features and L2 regularization.
    x and y are real numbers. The goal is to fit y = hypothesis(x) while minimizing
    both the squared error and the L2 norm of the weights.
    
    The objective function is:
    J(w) = (1/2m) * sum((h(x) - y)^2) + (lambda/2) * sum(w_j^2)
    where lambda is the regularization parameter controlling the strength of regularization
    """

    def __init__(self, degree = 1, learning_rate = 1e-3, lambda_reg = 0.1):
        self.degree = degree
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.weights = [0.1 * random.random() for i in range(self.degree + 1)]
 
    def get_features(self, x):
        return [x**p for p in range(self.degree + 1)]

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        return sum(w * f for w, f in zip(self.weights, features))

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        mse_loss = (self.hypothesis(x) - y)**2

        l2_loss = self.lambda_reg * sum(w**2 for w in self.weights[1:])
        return mse_loss + l2_loss

    def gradient(self, x, y):
        featuresX = self.get_features(x)
        pred_error = 2 * (self.hypothesis(x) - y)
        
        gradients = [pred_error * featuresX[0]]
        
        for i in range(1, self.degree + 1):
            grad = pred_error * featuresX[i] + 2 * self.lambda_reg * self.weights[i]
            gradients.append(grad)
            
        return gradients
    
    def train(self, dataset : util.Dataset, evalset = None):
        nIterations = 2000
        k = 0
        eval_iters = []
        losses = []
        testLosses = []
        alpha = self.learning_rate/(2 * dataset.get_size())
        max_weight = 1e6

        while k < nIterations:
            randomSample = dataset.get_sample()

            if k % 100 == 0:
                eval_iters.append(k)
                losses.append(dataset.compute_average_loss(self))
                if evalset is not None:
                    testLosses.append(evalset.compute_average_loss(self))

            grads = self.gradient(randomSample[0], randomSample[1])
            
            for j in range(self.degree + 1):
                self.weights[j] -= alpha * grads[j]
                if self.weights[j] > max_weight:
                    self.weights[j] = max_weight
                elif self.weights[j] < -max_weight:
                    self.weights[j] = -max_weight

            k += 1
        
        return [eval_iters, losses, testLosses]

def regularized_regression():

    # Helpful functions:
    # util.get_dataset, util.Dataset.compute_average_loss, util.RegressionDataset.plot_data
    # util.RegressionDataset.plot_loss_curve

    # (a)
    sine_train = util.get_dataset("sine_train")
    sine_test = util.get_dataset("sine_test")
    sine_model = RidgeRegressionModel(1, 1e-4, 0.1)

    sine_train_result = sine_model.train(sine_train, sine_test)

    print('average loss of sine_train:', sine_train.compute_average_loss(sine_model))
    sine_train.plot_data(sine_model)

    # (b)
    sine_train.plot_loss_curve(sine_train_result[0], sine_train_result[1])

    # (c)
    degrees = [1, 2]
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    lambda_regs = [0.0, 0.01, 0.1, 1.0]

    sine_val = util.get_dataset("sine_val")
    results = []

    for degree in degrees:
        for lr in learning_rates:
            for lam in lambda_regs:
                model = RidgeRegressionModel(degree, lr, lam)
                model.train(sine_train)

                train_loss = sine_train.compute_average_loss(model)
                val_loss = sine_val.compute_average_loss(model)

                results.append((degree, lr, lam, train_loss, val_loss))
            
    best = min(results, key=lambda x: x[4])

    print('best combination:', 
          f"degree={best[0]}, lr={best[1]:.0e}, lambda={best[2]:.2e}, "
          f"train_loss={best[3]:.4f}, val_loss={best[4]:.4f}")

# PA4 Q8 EX

class DatabaseLogisticRegressionModel(Model):
    """
    Binary logistic regression model for classifying 2D point data.
    
    Input:
    - x: A list of two numbers [x1, x2] representing coordinates in 2D space
    - y: Binary label (0 or 1) representing the class of the point
    
    The model learns a decision boundary in the 2D plane that separates
    points labeled 0 from points labeled 1. It outputs P(y = 1 | x),
    the probability that a point x belongs to class 1, and makes predictions
    by thresholding this probability at 0.5.
    """

    def __init__(self, learning_rate = 1e-2):
        self.learning_rate = learning_rate
        self.weights = [0 for _ in range(3)]

    def get_features(self, x):
        return [1] + x

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        logit = sum(w * f for w, f in zip(self.weights, features))
        return 1 / (1 + math.exp(-logit))

    def predict(self, x):
        if self.hypothesis(x) > 0.5:
            return 1
        else:
            return 0

    def loss(self, x, y):
        h = self.hypothesis(x)
        return (-1) * (y * math.log(h) + (1 - y) * math.log(1 - h))

    def gradient(self, x, y):
        featuresX = self.get_features(x)
        pred_error = self.hypothesis(x) - y
        return [pred_error * f for f in featuresX]

    def train(self, dataset : util.Dataset, evalset : util.Dataset = None):
        nIterations = 5000
        k = 0
        eval_iters = []
        accuracies = []
        testAccuracies = []
        alpha = self.learning_rate
        max_weight = 10.0

        while k < nIterations:

            randomSample = dataset.get_sample()

            if k % 500 == 0:
                eval_iters.append(k)
                accuracies.append(dataset.compute_average_accuracy(self, 50))
                if evalset is not None:
                    testAccuracies.append(evalset.compute_average_accuracy(self, 50))

            grads = self.gradient(randomSample[0], randomSample[1])
            for j in range(3):
                self.weights[j] -= alpha * grads[j]
                self.weights[j] = max(min(self.weights[j], max_weight), -max_weight)

            k += 1

        return [eval_iters, accuracies, testAccuracies]

def database_classification():
    ex4q7i = util.get_dataset("ex4q7i")
    ex4q7ii = util.get_dataset("ex4q7ii")
    ex4q7iii = util.get_dataset("ex4q7iii")
    ex4q7iv = util.get_dataset("ex4q7iv")
    
    for i, dataset in enumerate([ex4q7i, ex4q7ii, ex4q7iii, ex4q7iv], 1):
        database_model = DatabaseLogisticRegressionModel(learning_rate=1e-2)
        database_model.train(dataset)

        acc = dataset.compute_average_accuracy(database_model)
        print(f"dataset {i} accuracy after training: {acc:.4f}")

def main():
    linear_regression()
    binary_classification()
    multi_classification()
    regularized_regression()
    database_classification()

if __name__ == "__main__":
    main()
