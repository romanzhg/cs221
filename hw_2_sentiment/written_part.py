#!/usr/bin/env python

import collections

eta = float(1)

# Helper functions.
def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    rtn = float(0)
    for key in v1:
        if key in v2:
            rtn += v1[key] * v2[key]

    return rtn

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    for key in v2:
        v1[key] += v2[key] * scale

def sparseVectorMultiplication(v, scale) :
    for key in v:
        v[key] = v[key] * scale

def getMargin(weight, feature, y):
    # y is the label, the value is {-1, 1}.
    return sparseVectorDotProduct(weight, feature) * y

def sgd(weight, feature, label):
    # Updates weight.
    # label has value {-1, 1}.
    global eta
    gradient = collections.defaultdict(float)
    if (1 - getMargin(weight, feature, label)) > 0 :
        gradient = feature
        sparseVectorMultiplication(gradient, -label)
    else:
        # gradient is all 0 in this case.
        pass

    incrementSparseVector(weight, (-1) * eta, gradient)

def printDefaultDict(d):
    for key in d:
        print("key: " + str(key) + " value: " + str(d[key]))

# Main.
def main():
    weight = collections.defaultdict(float)

    # The training process, run for each input.
    feature = collections.defaultdict(float)
    
    feature.clear()
    feature["pretty"] = 1
    feature["bad"] = 1
    sgd(weight, feature, -1)

    feature.clear()
    feature["good"] = 1
    feature["plot"] = 1
    sgd(weight, feature, 1)

    feature.clear()
    feature["not"] = 1
    feature["good"] = 1
    sgd(weight, feature, -1)

    feature.clear()
    feature["pretty"] = 1
    feature["scenery"] = 1
    sgd(weight, feature, 1)

    printDefaultDict(weight)

if __name__ == "__main__":
    main()
