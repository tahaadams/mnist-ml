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
        sum = 0

        for j in range(len(features)):
            sum = sum + (self.weights[j] * features[j])

        return sum

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        return (self.hypothesis(x) - y)**2

    def gradient(self, x, y):
        featuresX = self.get_features(x)
        
        return [2 * (self.hypothesis(x) - y) * featuresX[i] for i in range(self.degree + 1)]
    
    def train(self, dataset : util.Dataset, evalset = None):
        nIterations = 10000
        k = 0
        eval_iters = []
        losses = []
        testLosses = []
        alpha = self.learning_rate/(2 * dataset.get_size())

        while k < nIterations:
            randomSample = dataset.get_sample()

            eval_iters.append(k)
            losses.append(dataset.compute_average_loss(self))
            testLosses.append(evalset.compute_average_loss(self))

            featuresX = self.get_features(randomSample[0])
  
            for j in range(self.degree + 1):
                updateValue = alpha * 2 * dataset.get_size() * (self.hypothesis(randomSample[0]) - randomSample[1])
                self.weights[j] = self.weights[j] - (updateValue * featuresX[j])

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

    print('average loss of sine_train', sine_train.compute_average_loss(sine_model))
    sine_train.plot_data(sine_model)

    # (b)
    sine_train.plot_loss_curve(sine_train_result[0], sine_train_result[1])

    # (c)
    # mix and match 10 times, (maybe do a for loop) and compute average loss
    sine_val = util.get_dataset("sine_val")
    sine_model0 = PolynomialRegressionModel(1, 1e-4/2)
    print('average loss of sine_model0', sine_val.compute_average_loss(sine_model0))
    sine_model1 = PolynomialRegressionModel(1, 1e-4/3)
    print('average loss of sine_model1', sine_val.compute_average_loss(sine_model1))
    sine_model2 = PolynomialRegressionModel(1, 1e-4/4)
    print('average loss of sine_model2', sine_val.compute_average_loss(sine_model2))
    sine_model3 = PolynomialRegressionModel(1, 1e-5)
    print('average loss of sine_model3', sine_val.compute_average_loss(sine_model3))
    sine_model4 = PolynomialRegressionModel(1, 1e-5/2)
    print('average loss of sine_model4', sine_val.compute_average_loss(sine_model4))
    sine_model5 = PolynomialRegressionModel(1, 1e-5/3)
    print('average loss of sine_model5', sine_val.compute_average_loss(sine_model5))
    sine_model6 = PolynomialRegressionModel(1, 1e-6)
    print('average loss of sine_model6', sine_val.compute_average_loss(sine_model6))
    sine_model7 = PolynomialRegressionModel(1, 1e-6/2)
    print('average loss of sine_model7', sine_val.compute_average_loss(sine_model7))
    sine_model8 = PolynomialRegressionModel(1, 1e-7)
    print('average loss of sine_model8', sine_val.compute_average_loss(sine_model8))
    sine_model9 = PolynomialRegressionModel(1, 1e-7/2)
    print('average loss of sine_model9', sine_val.compute_average_loss(sine_model9))

    print('model 8 has the best results from degree 1, learning rate 1e-7.')

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
    
    # # (b)
    mnist_binary_train.plot_confusion_matrix(mnist_binary_model)

    # # (c)
    mnist_binary_train.plot_image(mnist_binary_train_result[3])

    # # (d)
    misclassified = []
    mis_info = []
    for i in range(mnist_binary_test.get_size()):
        x = mnist_binary_test.xs[i]
        y = mnist_binary_test.ys[i]
        pred = mnist_binary_model.predict(x)
        if pred != y:
            prob = mnist_binary_model.hypothesis(x)
            flat = [pixel for row in x for pixel in row]
            misclassified.append(flat)
            mis_info.append((i, y, pred, prob))
        if len(misclassified) >= 10:
            break

    if len(misclassified) == 0:
        print("No misclassified examples found in the test set.")
    else:
        for idx, (i, actual, pred, prob) in enumerate(mis_info):
            print(f"Error {idx+1}: index={i}, actual={actual}, predicted={pred}, prob={prob:.4f}")
            mnist_binary_test.plot_image(misclassified[idx])

    # 8. Extra credit
    ex4q7i = util.get_dataset("ex4q7i")
    ex4q7ii = util.get_dataset("ex4q7ii")
    ex4q7iii = util.get_dataset("ex4q7iii")
    ex4q7iv = util.get_dataset("ex4q7iv")
    ex4q7_model = BinaryLogisticRegressionModel(28**2)

    ex4q7i_result = ex4q7_model.train(ex4q7i, mnist_binary_test)
    ex4q7ii_result = ex4q7_model.train(ex4q7ii, mnist_binary_test)
    ex4q7iii_result = ex4q7_model.train(ex4q7iii, mnist_binary_test)
    ex4q7iv_result = ex4q7_model.train(ex4q7iv, mnist_binary_test)

    ex4q7i.compute_average_accuracy(ex4q7_model)
    ex4q7i.plot_confusion_matrix(ex4q7_model)
    ex4q7ii.compute_average_accuracy(ex4q7_model)
    ex4q7ii.plot_confusion_matrix(ex4q7_model)
    ex4q7iii.compute_average_accuracy(ex4q7_model)
    ex4q7iii.plot_confusion_matrix(ex4q7_model)
    ex4q7iv.compute_average_accuracy(ex4q7_model)
    ex4q7iv.plot_confusion_matrix(ex4q7_model)

    ex4q7i.plot_image(ex4q7i_result[3])
    ex4q7ii.plot_image(ex4q7ii_result[3])
    ex4q7iii.plot_image(ex4q7iii_result[3])
    ex4q7iv.plot_image(ex4q7iv_result[3])


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
        classHypothesis = []
        features = self.get_features(x)
        mainSum = 0

        for k in range(self.num_classes):
            classSum = 0

            for j in range(self.num_features + 1):
                classSum = classSum + (self.weights[k][j] * features[j])

            mainSum = mainSum + math.exp(classSum)
        
        for k in range(self.num_classes):
            sum = 0

            for j in range(self.num_features + 1):
                sum = sum + (self.weights[k][j] * features[j])
            
            classValue = math.exp(sum) / mainSum
            classHypothesis.append(classValue)

        return classHypothesis

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

    def gradient(self, x, y, k):
        featuresX = self.get_features(x)
        sum = 0
        for i in range(self.num_features + 1):
            sum = sum + ((self.indicator(y, k) - self.hypothesis(x)[k]) * featuresX[i])
        
        return sum
        
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
        alpha = self.learning_rate/(2 * dataset.get_size())

        while k < nIterations:
            print('iteration:', k)
            randomSample = dataset.get_sample()

            if k % 20 == 0:
                eval_iters.append(k)
                accuracies.append(dataset.compute_average_accuracy(self, 20))
                testAccuracies.append(evalset.compute_average_accuracy(self, 20))

            for c in range(self.num_classes):
                for j in range(self.num_features + 1):
                    updateValue = alpha * 2 * dataset.get_size() * self.gradient(randomSample[0], randomSample[1], c)
                    self.weights[c][j] = self.weights[c][j] - updateValue

            k += 1
        
        return [eval_iters, accuracies, testAccuracies, self.weights[1:]]


# PA4 Q6
def multi_classification():

    # Helpful functions:
    # util.Dataset.compute_average_accuracy, util.MNISTDataset.plot_accuracy_curve
    # util.MNISTDataset.plot_confusion_matrix
    # util.MNISTDataset.plot_image

    "*** YOUR CODE HERE ***"
    
    # (a)
    mnist_train = util.get_dataset("mnist_train")
    mnist_train.xs = mnist_train.xs[:10]
    mnist_train.ys = mnist_train.ys[:10]
    mnist_test = util.get_dataset("mnist_test")
    mnist_model = MultiLogisticRegressionModel(28**2, 10)

    mnist_train_result = mnist_model.train(mnist_test)

    mnist_test.compute_average_accuracy(mnist_model)
    mnist_test.plot_accuracy_curve(mnist_train_result[0], mnist_train_result[1])
    mnist_test.plot_accuracy_curve(mnist_train_result[0], mnist_train_result[2])
    
    # (b)
    mnist_train.plot_confusion_matrix(mnist_model)

    # (c)
    mnist_train.plot_image(mnist_train_result[3])


def main():
    linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()
