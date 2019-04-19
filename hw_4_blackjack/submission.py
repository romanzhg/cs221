import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Version 1
# TODO: current version pass the tests. Modify blackjackFeatureExtractor in next version
# so the peek result(next card) is included in the feature.
#
# Written part
# 4b. Value iteration has access to the entire model of MDP, so it will always give the
# the optimal solution. 
# Q-Learning with identity feature extraction is like rote learning, it remembers a state
# only after there is a path that go through it, when the state space is large, it is hard to
# visit far away states(hard to explore), so it will return sub-optimal policies.
# In this problem, we can see for the small mdp, Q-Learning works good, for the large mdp,
# Q-Learning gives bad policies. The Vopt(start_state) is only 8.63, while Value iteration
# gives 35.52.
# With the feature extractor implemented in 4c, Q-Learning gives a policy with value 33.70,
# which is close to the optimal.
#
# 4d. Using policy trained from the original problem, the performance is only half as good.
# rl0(policy from value iteration on the original problem) 
# value of start state(through experiment): 6.8244
# rl1(policy from q-learning trained on the modified problem)
# value of start state(through experiment): 12.0
# rl2(policy from q-learning trained on the original problem)
# value of start state(through experiment): 6.8358
#
############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return ['right']
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0:
            return [(0, 0.9, 0), (1, 0.1, 1000000)]
        else:
            return []
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost
        self.cardTypeCount = len(cardValues)

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # drawCard cannot simply return None when the drawn card stack is empty,
    # because of the requirement of "quit when no cards".
    def drawCard(self, cardStack, cardIndex):
        newCardStack = list(cardStack)
        newCardStack[cardIndex] = newCardStack[cardIndex] - 1
        return tuple(newCardStack)

    def isEndState(self, state):
        return state[2] is None

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        
        # In case of end state, we actually don't care about the parameter 'action', it could be
        # None.
        total, nextCard, counts = state
        if self.isEndState(state):
            return []

        totalCardCount = sum(counts)
        pickedInPrevRound = nextCard is not None

        if action == 'Take':
            if pickedInPrevRound:
                newTotalValue = total + self.cardValues[nextCard]
                if newTotalValue > self.threshold:
                    return [((newTotalValue, None, None), 1, 0)]
                elif totalCardCount == 1:
                    return [((newTotalValue, None, None), 1, newTotalValue)]
                else:
                    return [((newTotalValue, None, self.drawCard(counts, nextCard)), 1, 0)]

            rtn = []
            for cardIndex in range(0, self.cardTypeCount):
                cardCount = counts[cardIndex]
                if cardCount == 0:
                    continue

                prob = cardCount / float(totalCardCount)
                newTotalValue = total + self.cardValues[cardIndex]
                
                if newTotalValue > self.threshold:
                    rtn.append(((newTotalValue, None, None), prob, 0))
                elif totalCardCount == 1:
                    return [((newTotalValue, None, None), 1, newTotalValue)]
                else:
                    rtn.append(((newTotalValue, None, self.drawCard(counts, cardIndex)), prob, 0))
            return rtn

        if action == 'Peek':
            if pickedInPrevRound:
                return []

            rtn = []
            for cardIndex in range(0, self.cardTypeCount):
                cardCount = counts[cardIndex]
                if cardCount == 0:
                    continue

                prob = cardCount / float(totalCardCount)
                rtn.append(((total, cardIndex, counts), prob, -self.peekCost))
            return rtn

        if action == 'Quit':
            return [((total, None, None), 1, total)]
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    return BlackjackMDP(cardValues=[3, 15], multiplicity=100, threshold=20, peekCost=1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    def getVopt(self, state):
        return max(self.getQ(state, ac) for ac in self.actions(state))

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

        # Nothing to update for this case. The Q-value will always be 0.
        vOpt = 0
        if newState is not None:
            vOpt = self.getVopt(newState)

        qPredict = self.getQ(state, action)
        # vOpt = self.getVopt(newState)
        target = float(reward) + float(self.discount) * vOpt
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - self.getStepSize() * (qPredict - target) * v

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE    
    mdp.computeStates()

    vi = ValueIteration()
    vi.solve(mdp)

    trails = 30000
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, 1)
    util.simulate(mdp, rl, numTrials=trails, maxIterations=1000, verbose=False)

    rl.explorationProb = 0

    differentActionCount = 0
    totalActionCount = 0
    for k, v in vi.pi.iteritems():
        rlAction = rl.getAction(k)
        totalActionCount += 1
        if rlAction != v:
            differentActionCount += 1
            # print("diff: " + str(k) + " action vi: " + v + " action rl: " + rlAction)

    totalReward = util.simulate(mdp, rl, numTrials=trails, maxIterations=1000, verbose=False)

    print("different action count: " + str(differentActionCount))
    print("difference rate: " + str(differentActionCount / float(totalActionCount)))

    print("vi value of start state: " + str(vi.V[mdp.startState()]))
    print("rl value of start state: " + str(rl.getVopt(mdp.startState())))
    print("rl value of start state(through experiment): " + str(sum(totalReward) / float(trails)))

    print("")
    # END_YOUR_CODE

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    rtn = []
    # rtn.append(((action, total, nextCard), 1))

    rtn.append(((action, total), 1))

    if counts is not None:
        k1 = tuple([0 if elem == 0 else 1 for elem in counts])
        # Change this to "(action,) + k1" will collide with the feature below.
        rtn.append(((action, k1), 1))

    if counts is not None:
        for i, v in enumerate(counts):
            rtn.append(((action, i, v), 1))

    return rtn

    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE

    # TODO: remove these two lines, not necessary.
    original_mdp.computeStates()
    modified_mdp.computeStates()

    trails = 30000

    vi0 = ValueIteration()
    vi0.solve(original_mdp)

    rl0 = util.FixedRLAlgorithm(vi0.pi)
    totalReward0 = util.simulate(modified_mdp, rl0, numTrials=trails)

    rl1 = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, 1)
    util.simulate(modified_mdp, rl1, numTrials=trails)
    rl1.explorationProb = 0
    totalReward1 = util.simulate(modified_mdp, rl1, numTrials=trails)

    rl2 = QLearningAlgorithm(original_mdp.actions, original_mdp.discount(), featureExtractor, 1)
    util.simulate(original_mdp, rl2, numTrials=trails)
    rl2.explorationProb = 0
    totalReward2 = util.simulate(modified_mdp, rl2, numTrials=trails)

    print("rl0(parameters from value iteration on the original problem) \
        value of start state(through experiment): " + str(sum(totalReward0) / float(trails)))
    print("rl1(parameters from q-learning trained on the modified problem) \
        value of start state(through experiment): " + str(sum(totalReward1) / float(trails)))
    print("rl2(parameters from q-learning trained on the original problem) \
        value of start state(through experiment): " + str(sum(totalReward2) / float(trails)))

    # END_YOUR_CODE

