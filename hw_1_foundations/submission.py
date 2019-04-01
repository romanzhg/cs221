import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    wordList = text.split()
    wordList.sort()
    return max(wordList)
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt(math.pow(loc1[0] - loc2[0], 2) + math.pow(loc1[1] - loc2[1], 2))
    # END_YOUR_CODE

############################################################
# Problem 3c

def initListOfSet(size):
    # Initialize a list of list. The outer list index starts from 0.
    rtn = list()
    for i in range(0,size):
        rtn.append(set())
    return rtn

def maybeInsertWord(word, wordToId, wordCounter):
    if word in wordToId:
        return wordCounter
    else:
        wordToId[word] = wordCounter
        return wordCounter + 1

def buildGraph(sentence):
    words = sentence.split()
    sentenceLen = len(words)

    wordToId = dict()
    wordCount = 0

    for word in words:
        wordCount = maybeInsertWord(word, wordToId, wordCount)

    idToWord = [None] * wordCount
    for key, value in wordToId.iteritems():
        idToWord[value] = key

    graph = initListOfSet(wordCount)

    for index in range(0, len(words) - 1):
        wordFrom = words[index]
        wordTo = words[index + 1]
        graph[wordToId[wordFrom]].add(wordToId[wordTo])

    return idToWord, graph, wordCount, sentenceLen

def helper(rtnList, idToWord, graph, curVertexId, curSentence, remainingWords):
    # TODO: if the curSentence is empty, should not return it.
    if remainingWords == 1:
        curSentence.append(idToWord[curVertexId])
        rtnList.append(" ".join(curSentence))
        curSentence.pop()
        return
    
    curSentence.append(idToWord[curVertexId])
    for adj in graph[curVertexId]:
        helper(rtnList, idToWord, graph, adj, curSentence, remainingWords - 1)
    curSentence.pop()

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    idToWord, graph, uniqueWords, sentenceLen = buildGraph(sentence)
    rtnList = list()
    curSentence = list()

    for curId in range(uniqueWords):
        helper(rtnList, idToWord, graph, curId, curSentence, sentenceLen)

    return rtnList
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    rtn = float(0)
    for key in v1:
        if key in v2:
            rtn += v1[key] * v2[key]

    return rtn
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v2:
        v1[key] += v2[key] * scale
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    dd = collections.defaultdict(int)
    for word in text.split():
        dd[word] += 1
    
    rtn = set()
    for key in dd:
        if dd[key] == 1:
            rtn.add(key)
    return rtn
    # END_YOUR_CODE

############################################################
# Problem 3g

def getDp(text, dp, start, end):
    if start == end:
        return 1
    if start > end:
        return 0
    if dp[start][end] != 0:
        return dp[start][end]
    if text[start] == text[end]:
        rtn = 2 + getDp(text, dp, start + 1, end - 1)
        dp[start][end] = rtn
        return rtn
    else:
        rtn = max(getDp(text, dp, start + 1, end),
                  getDp(text, dp, start, end - 1))
        dp[start][end] = rtn
        return rtn

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    textLen = len(text)
    dp = [[0 for col in range(textLen)] for row in range(textLen)]
    return getDp(text, dp, 0, textLen - 1)
    # END_YOUR_CODE
