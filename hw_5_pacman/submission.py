from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    # This implies agents of this class can only be a max agent.
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """
  def getValue(self, gameState, curDepth, agentCount):
    if gameState.isWin() or gameState.isLose():
        return gameState.getScore()
    if curDepth == 0:
        return self.evaluationFunction(gameState)

    agentIndex = curDepth % agentCount
    legalMoves = gameState.getLegalActions(agentIndex)
    if agentIndex == 0:
        # Max agent.
        values = [self.getValue(gameState.generateSuccessor(agentIndex, action), curDepth - 1, agentCount) for action in legalMoves]
        return max(values)
    else:
        # Min agent.
        values = [self.getValue(gameState.generateSuccessor(agentIndex, action), curDepth - 1, agentCount) for action in legalMoves]
        return min(values)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    legalMoves = gameState.getLegalActions(0)
    modifiedDepth = gameState.getNumAgents() * self.depth
    values = [(self.getValue(gameState.generateSuccessor(0, action), modifiedDepth - 1, gameState.getNumAgents()), action)
            for action in legalMoves]
    bestScore = max(values)
    bestIndices = [index for index in range(len(values)) if values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]

    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """
  def getValueAndAction(self, gameState, curDepth, agentCount, lb, ub):
    # Returns a dummy action if current state is an end state.
    if gameState.isWin() or gameState.isLose():
        return (gameState.getScore(), Directions.STOP)
    if curDepth == 0:
        return (self.evaluationFunction(gameState), Directions.STOP)

    agentIndex = curDepth % agentCount
    legalMoves = gameState.getLegalActions(agentIndex)
    cmpFunc = max if agentIndex == 0 else min

    valueAndActions = []
    if agentIndex == 0:
        # Max agent.
        for action in legalMoves:
            valueAndAction = (self.getValueAndAction(gameState.generateSuccessor(agentIndex, action),
                                                     curDepth - 1,
                                                     agentCount,
                                                     lb, ub)[0],
                              action)
            lb = cmpFunc(lb, valueAndAction[0])
            valueAndActions.append(valueAndAction)
            if lb >= ub:
                return (lb, action)
    else:
        # Min agent.
        for action in legalMoves:
            valueAndAction = (self.getValueAndAction(gameState.generateSuccessor(agentIndex, action),
                                                     curDepth - 1,
                                                     agentCount,
                                                     lb, ub)[0],
                              action)
            ub = cmpFunc(ub, valueAndAction[0])
            valueAndActions.append(valueAndAction)
            if lb >= ub:
                return (ub, action)

    bestValueAndAction = cmpFunc(valueAndActions)
    bestIndices = [index for index in range(len(valueAndActions)) if valueAndActions[index][0] == bestValueAndAction[0]]
    chosenIndex = random.choice(bestIndices)
    return valueAndActions[chosenIndex]

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    modifiedDepth = gameState.getNumAgents() * self.depth
    rtn = self.getValueAndAction(gameState, modifiedDepth, gameState.getNumAgents(), - sys.maxint - 1, sys.maxint)
    return rtn[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """
  def getValueAndActionEM(self, gameState, curDepth, agentCount):
    # Returns a dummy action if current state is an end state.
    if gameState.isWin() or gameState.isLose():
        return (gameState.getScore(), Directions.STOP)
    if curDepth == 0:
        return (self.evaluationFunction(gameState), Directions.STOP)

    agentIndex = curDepth % agentCount
    legalMoves = gameState.getLegalActions(agentIndex)
    
    if agentIndex == 0:
        # Max agent.
        valueAndActions = []
        for action in legalMoves:
            valueAndAction = (self.getValueAndActionEM(gameState.generateSuccessor(agentIndex, action),
                                                       curDepth - 1,
                                                       agentCount)[0],
                              action)
            valueAndActions.append(valueAndAction)

        bestValueAndAction = max(valueAndActions)
        bestIndices = [index for index in range(len(valueAndActions)) if valueAndActions[index][0] == bestValueAndAction[0]]
        chosenIndex = random.choice(bestIndices)
        return valueAndActions[chosenIndex]

    else:
        # Min agent.
        values = [self.getValueAndActionEM(gameState.generateSuccessor(agentIndex, action),
                                           curDepth - 1,
                                           agentCount)[0] for action in legalMoves]
        return (sum(values) / float(len(values)), Directions.STOP)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    modifiedDepth = gameState.getNumAgents() * self.depth
    rtn = self.getValueAndActionEM(gameState, modifiedDepth, gameState.getNumAgents())
    return rtn[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def getMinManhattanDist(capsules, pacManPos):
    if len(capsules) == 0:
        return 1000000
    dists = [manhattanDistance(pos, pacManPos) for pos in capsules]
    return min(dists)

def betterEvaluationFunction(gs):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>
    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    if gs.isLose():
        return -100000

    if gs.isWin():
        return 100000

    ghostStates = gs.getGhostStates()

    rtn = float(0)

    rtn += gs.getScore()

    for ghost in ghostStates:
        if ghost.scaredTimer > 1:
            rtn = rtn \
                  + 300 / float(manhattanDistance(ghost.getPosition(), gs.getPacmanPosition()) + 1)
        else:
            rtn = rtn \
                  - 300 / float(manhattanDistance(ghost.getPosition(), gs.getPacmanPosition()) + 1)

    # currentFood = gs.getFood()
    # minManhattanDist = 10000

    # for x in range(0, gs.data.layout.width):
    #     for y in range(0, gs.data.layout.height):
    #         if currentFood[x][y] == True:
    #             minManhattanDist = min(minManhattanDist, manhattanDistance(gs.getPacmanPosition(), (x, y)))
        
    # rtn += 300 / float(minManhattanDist + 1)
    return rtn
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
