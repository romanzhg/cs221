#!/usr/bin/python

import random
import collections
import copy
import math
import sys
from util import *

'''

Notes on python copy
a = list()
b = list()
b = a # creates an alias
b = a.copy() # creates an shallow copy, b and a points to the same objects, but the pointer array is separate
b = copy.deepcopy(a) # create an deep copy, copy objects recursively

Written part in programming questions

3.d. The errors were mainly due to negation words, like "never boring".
"never" is a negative word with gives a negative score, same as boring, but
together they are a compliment to the movie.
The classifier would need some non-linearity to get the right answer, for example,
after "never" the words will have their feature value negated. 

3.f. Using n-gram combines words, to some extend it solves the "never boring"
problem. And "not good" would be another example. Data attached below.

4.a
1. u1 = [0.5, 2], u2 = [1, 0]
2. u1 = [0, 1], u2 = [1.5, 1]

4.c At the finding closest centroid setp, 
assignment_group = arg min_k sum_group ||phi(x) - uk||

The training data is attached,
using word feature
11221 weights
Official: train error = 0.0284186831739, dev error = 0.27687113112.

using character feature
train with n-gram 10
273325 weights
Official: train error = 0.000844119302195, dev error = 0.337647720878
----- END PART 3b-2-basic [took 0:00:14.364842 (max allowed 20 seconds), 0/2 points]

train with n-gram 8
243346 weights
Official: train error = 0.000562746201463, dev error = 0.292909397862
----- END PART 3b-2-basic [took 0:00:16.076096 (max allowed 30 seconds), 2/2 points]

train with n-gram 6
166495 weights
Official: train error = 0.0, dev error = 0.271525042206
----- END PART 3b-2-basic [took 0:00:21.833555 (max allowed 60 seconds), 2/2 points]

train with n-gram 5
107224 weights
Official: train error = 0.0, dev error = 0.270680922904
----- END PART 3b-2-basic [took 0:00:18.611444 (max allowed 60 seconds), 2/2 points]

'''
############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features = collections.defaultdict(int)
    wordList = x.split()
    for word in wordList:
        features[word] += 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def getMargin(weights, feature, y):
    # y is the label, the value is {-1, 1}.
    return dotProduct(weights, feature) * y

def sparseVectorMultiplication(v, scale) :
    for key in v:
        v[key] = v[key] * scale

def sgd(weights, feature, label, eta):
    # Updates weight.
    # label has value {-1, 1}.
    gradient = collections.defaultdict(float)
    if (1 - getMargin(weights, feature, label)) > 0 :
        gradient = feature
        sparseVectorMultiplication(gradient, -label)
    else:
        # gradient is all 0 in this case.
        pass

    increment(weights, (-1) * eta, gradient)

def getPredictor(weights, featureExtractor):
    # A linear -1/1 predictor, the decision boundary is 0.
    return lambda x : 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    for curIter in range(numIters):
        for example in trainExamples:
            sgd(weights, featureExtractor(example[0]), example[1], eta)
        errRateTrain = evaluatePredictor(trainExamples, getPredictor(weights, featureExtractor))
        errRateTest = evaluatePredictor(testExamples, getPredictor(weights, featureExtractor))
        # print("train error: " + str(errRateTrain))
        # print("test error: " + str(errRateTest))

    # evaluatePredictor(example, predictor)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        dataSize = random.randint(2, 101)
        x = collections.defaultdict(int)
        keysList = weights.keys()
        for _ in range(dataSize):
            x[random.choice(keysList)] = random.randint(1, 15)
        return (x, 1 if dotProduct(weights, x) >= 0 else -1)
        # END_YOUR_CODE

    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        tmpStr = x.replace(" ", "")
        features = collections.defaultdict(int)
        for startIndex in range(0, len(tmpStr) - n + 1):
            features[tmpStr[startIndex : startIndex + n]] += 1
        return features
        
    return extract

############################################################
# Problem 4: k-means
############################################################

def initListOfDict(size):
    rtn = list()
    for i in range(0,size):
        rtn.append(collections.defaultdict(float))
    return rtn

# An optimization on getting distance, centroid is dense, while dataPoint is sparse.
def getDistSqr(dataPoint, centroids, centroidSqrSums, centroidIndex):
    rtn = centroidSqrSums[centroidIndex]
    for k, v in dataPoint.items():
        rtn += v * v - 2 * centroids[centroidIndex].get(k, 0) * v
    return rtn

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    centroids = initListOfDict(K)
    assignments = [0 for _ in range(len(examples))]
    prevLoss = sys.float_info.max / 2
    curLoss = float(0)

    centroidSqrSum = [float(0) for _ in range(K)]

    initialCentroids = random.sample(xrange(0, len(examples)), K)
    for index in range(0, K):
        centroids[index] = examples[initialCentroids[index]].copy()

    for _ in range(maxIters):
        # calculate assignments and the curLoss # for k-mean the dist is L2 norm
        # (take square of the diff and sqrt).
        curLoss = float(0)
        
        for centroidIndex in range(0, len(centroids)):
            centroidSqrSum[centroidIndex] = float(
                dotProduct(centroids[centroidIndex], centroids[centroidIndex]))

        for elemIndex in range(0, len(examples)):
            tmpDict = dict()
            for centroidIndex in range(0, len(centroids)):
                distSqr = getDistSqr(examples[elemIndex], centroids, centroidSqrSum, centroidIndex)
                tmpDict[centroidIndex] = distSqr
            tmpAssignment = min(tmpDict, key = tmpDict.get)
            assignments[elemIndex] = tmpAssignment
            curLoss += tmpDict[tmpAssignment]

        # see if we can stop early
        if ((prevLoss - curLoss) <= 0) or (prevLoss - curLoss) / curLoss <= 0.01:
            break
        prevLoss = curLoss

        # calculate new centroids
        centroidExampleCount = [0 for _ in range(K)]
        for centroidIndex in range(0, len(centroids)):
            centroids[centroidIndex].clear()

        for elemIndex in range(0, len(examples)):
            increment(centroids[assignments[elemIndex]], float(1), examples[elemIndex])
            centroidExampleCount[assignments[elemIndex]] += 1

        for centroidIndex in range(0, len(centroids)):
            sparseVectorMultiplication(centroids[centroidIndex], 1 / float(centroidExampleCount[centroidIndex]))

    return (centroids, assignments, curLoss)
    # END_YOUR_CODE
