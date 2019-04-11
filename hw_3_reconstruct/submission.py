import shell
import util
import wordsegUtil

############################################################
'''
TODO: think of a framework for parameter tuning.
TODO: generate some test cases.

Python Notes
A python dictionary requires its key to be hashable.
A tuple is hashable, the program hash each elements and accumulates
the result,
https://github.com/python/cpython/blob/master/Objects/tupleobject.c

A list is not hashable, a list is mutable.
Python tuple is immutable.

############################################################

Written Part
1a. Consider 1 gram, with weight {at: 0.01, tend: 0.2, attend: 0.1}.
For input "attend", the greedy algorithm will get result "at tend",
but "attend" has the lower total cost.

2a. Consider "have some drink", and suppose the bigram give higher weight
to "have seem" than "have some", then if we use the greedy algorithm, "have seem"
will be chosen, the since "drnk" doesn't expand to many words, it is likely to be
"drink", "seem drink" doesn't make sense.

3a.
In this problem the state would have the semantic (endPosition, word), this
means the current vertex corresponds to a word(with vowels missing) ending
at endPosition in the query string.
Part 1 try all the possible end positions for a word.
Part 2 try all the possible fills for a word.
The solution to this problem combines part 1 and part 2, each vertex in part
one is expended to a set of vertexes(the possible fills), and if there is an
edge between two vertexes in part 1, there will be all to all edges for the
vertexes in these two sets.
A special ending state is added, with endPosition "len(query) + 1".
All vertexes with endPosition equal to len(query)
(which means the vertex corresponds to the last word) will be connected to this
state with a zero cost edge.

3c.
(Draw the graph.)
When using unigram model to solve this question, the only information we need to
maintain in a state is endPosition. In Problem 1, there is only one edge from one
state/vertex to the other, in this problem, there would be many, each edge corresponds
to a filled word.

Choose ub(w) = min_w'(b(w', w)), range over all w'(if there is no such w' then use value log(VOCAB_SIZE).
Build a graph with ub(w), then between each vertex, keep only the min cost edge.
On this graph, solve the reverse problem, assign each vertex a cost, and this cost will be
the heuristic value.

The heuristic will be consistent because we only reduce costs in the original(used in problem 3b) graph.


A few test cases,
Problem 1
Weareonamissiontomakeeveryacademicpaperpublishedavailableforfree

Problem 2
Why he should have captivated Scarlett when his mind was a stranger to hers she did not know

Problem 3
mgnllthppl
thtsmnthcrnr
(
thats me in the corner
)
thrgnllnfthrssnfrcslngthrvrklchhdbndslctdbythcptrfthshvrdn
( 
the original line of the Russian forces along the river Kolocha had been dislocated by the capture of the Shevardino
)
'''
############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

# state has the semantic (endPosition).
class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost
        self.queryLen = len(query)

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state == self.queryLen
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        rtn = []
        # Assumes the longest word has length 25.
        # Think of what would happen if we allow a very large number here: there will be a 
        # edge from current state to end, and if the cost of this edge is not set properly, sometimes
        # this edge may be chosen, which is never what we want.
        for wordLen in range(1, 25):
            if state + wordLen <= self.queryLen:
                tmpWord = self.query[state : state + wordLen]
                rtn.append([tmpWord, state + wordLen, self.unigramCost(tmpWord)])
        return rtn
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

# State has the semantic (index, word).
# The end state is has index len(input array), which doesn't correspond to any element in the array.
class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills
        self.queryWordLen = len(queryWords)
        self.endState = (len(queryWords), '')

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state == self.endState
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        tmpWord = state[1]
        tmpIndex = state[0]
        rtn = []
        # Connect every filled word for the last word to the end state.
        if tmpIndex == (self.queryWordLen - 1):
            rtn.append([None, self.endState, 0])
        else:
            toBeFilledWord = self.queryWords[tmpIndex + 1]
            fills = self.possibleFills(toBeFilledWord)
            if len(fills) == 0:
                rtn.append([toBeFilledWord, (tmpIndex + 1, toBeFilledWord), self.bigramCost(tmpWord, toBeFilledWord)])
            else:
                for nextWord in fills:
                    rtn.append([nextWord, (tmpIndex + 1, nextWord), self.bigramCost(tmpWord, nextWord)])
        return rtn
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    queryWords.insert(0, wordsegUtil.SENTENCE_BEGIN)

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    return ' '.join(ucs.actions[:-1])
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

# State has the semantic (endPosition, word).
# The end state is has index len(input query), which doesn't correspond to any element in the query.
class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills
        self.queryLen = len(query)
        self.endState = (len(query) + 1, '')

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state == self.endState
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        rtn = []
        tmpWord = state[1]
        # The index previous word ends and new word starts.
        tmpEndIndex = state[0]

        if tmpEndIndex == self.queryLen:
            rtn.append([None, self.endState, 0])
            return rtn

        # Assume words has at most 15 consonant chars.
        for wordLen in range(1, 16):
            nextEndIndex = tmpEndIndex + wordLen
            if nextEndIndex <= self.queryLen:
                toBeFilledWord = self.query[tmpEndIndex : nextEndIndex]
                fills = self.possibleFills(toBeFilledWord)
                for nextWord in fills:
                    rtn.append([nextWord, (nextEndIndex, nextWord), self.bigramCost(tmpWord, nextWord)])
        return rtn
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    return ' '.join(ucs.actions[:-1])
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
