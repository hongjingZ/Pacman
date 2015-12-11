# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] #may occur same index
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsuleplaces = currentGameState.getCapsules()
        curfood = currentGameState.getFood().asList()
        "*** YOUR CODE HERE ***"
        foodlist = newFood.asList()
        dist = []
        g_dist = []
        if len(foodlist) > 0:
            for food in foodlist:
                dist.append(abs(newPos[0]-food[0])+abs(newPos[1]-food[1]))
            min_dist = min(dist)
            max_dist = max(dist)
            for ghost in newGhostStates:
               g_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
                ##print "g:",ghost.getPosition()[0]
            min_g_dist = min(g_dist)
            stop =0
            if action == Directions.STOP:
                stop = 1
            w1 = -1.5     #weight of minimal distance to food
            w2 = -1     #weight of stop punishment
            w3 = -10     #weight of minimal distance to ghost
            w4 = 50     #weisght of eating a food
            evaluation = w1*min_dist + w2*stop + w3*(1/(min_g_dist+0.1)) + w4*(len(curfood)-len(foodlist))
            return evaluation
        else:
            return 0
        #return successorGameState.getScore()

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
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.actual_length = {}
    def is_finish(self,state,depth):
        max_depth = self.depth*self.agentCount
        ##print "agentCount:",self.agentCount,"depth:",depth
        if depth == max_depth or state.isWin() or state.isLose(): #max recur depth or win or loss
            return True
        else:
            return False

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maxval(self,state,agent,depth):   #pacman's turn get the max val
        val = -999999
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            val = max(val,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))
        return val

    def minval(self,state,agent,depth):
        val = 999999
        for action in state.getLegalActions(agent):
            val = min(val,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))
        return val

    def minimax(self,gameState,agent,depth):
        score = 0
        if agent == self.agentCount: ##it's pacman's turn
            agent = self.index
        if self.is_finish(gameState,depth):
            score = self.evaluationFunction(gameState)
        elif agent == self.index:   ## judge again, is pacman's turn?
            score = self.maxval(gameState,agent,depth)
        else:         ##ghosts's turn
            score = self.minval(gameState,agent,depth)
        return score

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        depth = 0; agentIndex = self.index
        Dict = {}
        self.agentCount = gameState.getNumAgents()
        actions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            eval_f = self.minimax(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth+1)
            Dict[eval_f] = action
        choice = Dict[max(Dict)]
        ##print "my choice:", choice
        return choice

class AlphaBetaAgent(MultiAgentSearchAgent):

  def alphabeta(self,state,agent,depth,action,alpha,beta):
    score = ()
    if agent == self.agentCount:
        agent = self.index
    if self.is_finish(state,depth):
        score = (self.evaluationFunction(state),action)
    elif agent == self.index:
        score = self.maxval(state,agent,depth,alpha,beta)

    else:
        score = self.minval(state,agent,depth,alpha,beta)


    return score

  def maxval(self,state,agent,depth,alpha,beta):
    v = (-999999,Directions.STOP)
    actions = state.getLegalActions(agent)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    for action in actions:
        tempv = (self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta)[0],action)
        v = max(v,tempv)
        if v[0] > beta:
            return v
        alpha = max (alpha,v[0])
    return v

  def minval(self,state,agent,depth,alpha,beta):
    v = (999999,Directions.STOP)
    for action in state.getLegalActions(agent):
        tempv = (self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta)[0],action)
        v = min(v,tempv)
        if v[0] < alpha:
            return v
        beta = min(beta,v[0])
    return v

  def getAction(self, gameState):

    "*** YOUR CODE HERE ***"
    state = gameState
    self.agentCount = state.getNumAgents()
    alpha = -999999
    beta = 999999
    depth = 0
    agent = self.index
    action = Directions.STOP
    v = ()
    v = self.alphabeta(state,agent,depth,action,alpha,beta)
    return v[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxval(self,state,agent,depth):   #pacman's turn get the max val
        val = -999999
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            val = max(val,self.value(state.generateSuccessor(agent,action),agent+1,depth+1))
        ##print "max:",val,"agent:",agent
        return val
    def expval(self,state,agent,depth):
        val = []
        weight = []
        exp = 0
        total = len(state.getLegalActions(agent))
        for action in state.getLegalActions(agent):
            weight.append(1/float(len(state.getLegalActions(agent))))
            val.append(self.value(state.generateSuccessor(agent,action),agent+1,depth+1))
        for i in range(total):
            exp = exp + val[i]*weight[i]
        ##print "expection:",exp,"agent:",agent,"weight",weight,"value",val
        return exp
    def value(self,state,agent,depth):
        score=0
        if agent == self.agentCount:
            agent = self.index
        if self.is_finish(state,depth):
            score = (self.evaluationFunction(state))
        elif agent == self.index:
            score = self.maxval(state,agent,depth)
        else:
            score = self.expval(state,agent,depth)

        ##print "expval returns:",retval," agent:",agent
        return score
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0; agentIndex = self.index
        Dict = {}
        self.agentCount = gameState.getNumAgents()
        actions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            eval_f = self.value(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth+1)
            Dict[eval_f] = action
        choice = Dict[max(Dict)]
        ##print "my choice:", choice
        return choice

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsuleplaces = currentGameState.getCapsules()
    curfood = currentGameState.getFood().asList()
    dist = []
    g_dist = []
    if len(curfood) > 0:
        for food in curfood:
            dist.append(abs(newPos[0]-food[0])+abs(newPos[1]-food[1]))
        min_dist = min(dist)
        max_dist = max(dist)
        is_scared = 0
        is_dead = 0
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                g_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
                if abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]) == 0:
                    is_dead = 1
            else:
                g_dist.append(ghost.scaredTimer*2+abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
                is_scared = 1
        min_g_dist = min(g_dist)
        w1 = -1.5     #weight of minimal distance to food
        w2 = -1   #weight of max_dis
        w3 = -15     #weight of minimal distance to ghost 15 is a 1400 option
        w4 = -50     #weisght of eating a food
        w5 = -500    #weight of eating a capsule

        ##if pacman is in the left tunal and eat all the food, it must get out soon.
        w6 = -1       #weight of legal actions
        evaluation = is_dead * (-1000)+w1*min_dist + w2*max_dist + w3*(1/(min_g_dist+0.1)) + w4*(len(curfood)) + w5*(len(capsuleplaces)) +  currentGameState.getScore()/float(5.0)
        return evaluation
    else:
        return 100000

# Abbreviation
better = betterEvaluationFunction
"""
def contestEvaluationFunction(currentGameState):

  "*** YOUR CODE HERE ***"
  newPos = currentGameState.getPacmanPosition()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  capsuleplaces = currentGameState.getCapsules()
  curfood = currentGameState.getFood().asList()
  "*** YOUR CODE HERE ***"
  dist = []
  g_dist = []
  g_bonus = 0
  g_punish = 0
  if len(curfood) > 0:
      for food in curfood:
          dist.append(abs(newPos[0]-food[0])+abs(newPos[1]-food[1]))
      min_dist = min(dist)
      max_dist = max(dist)
      is_scared = 0
      for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            g_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]) == 0:
                g_punish = 1
        else:
            g_dist.append(ghost.scaredTimer*2+abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]) == 0:
                g_bonus = g_bonus + 1
      min_g_dist = min(g_dist)
      bad = 0
      w1 = -1.5     #weight of minimal distance to food
      w2 = 0   #weight of max_dis
      w3 = -15     #weight of minimal distance to ghost 15 is a 1400 option
      w4 = -50     #weisght of eating a food
      w5 = -500    #weight of eating a capsule
      w6 = 400       #weight of legal actions
      w7 = -10000   #punish for death
      evaluation =  w7*g_punish+ w1*min_dist + w2*max_dist + w3*(1/(min_g_dist+0.1)) + w4*(len(curfood)) + w5*(len(capsuleplaces)) + w6*g_bonus
      +currentGameState.getScore()/float(10.0)
      return evaluation
  else:
      return 1000000+currentGameState.getScore()
"""
def contestEvaluationFunction(currentGameState):

  "*** YOUR CODE HERE ***"
  newPos = currentGameState.getPacmanPosition()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  capsuleplaces = currentGameState.getCapsules()
  curfood = currentGameState.getFood().asList()
  "*** YOUR CODE HERE ***"
  dist = []
  g_dist = []
  scared_dist = []
  cap_dist = []
  good = 0
  good1 = 0
  if len(capsuleplaces) > 0:
      for cap in capsuleplaces:
          cap_dist.append(abs(newPos[0]-cap[0])+abs(newPos[1]-cap[1]))
  if len(curfood) > 0:
      for food in curfood:
          dist.append(abs(newPos[0]-food[0])+abs(newPos[1]-food[1]))
      min_dist = min(dist)
      is_scared = 0
      is_dead = 0
      eat_ghost = 0
      for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            g_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]) == 0:
                is_dead = 1
        else:
            g_dist.append(ghost.scaredTimer*2+abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            scared_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])<= 0.5 and abs(newPos[1]-ghost.getPosition()[1]) <= 0.5:
                eat_ghost = 1
            is_scared = 1
      min_g_dist = min(g_dist)
      w1 = -1.5     #weight of minimal distance to food
      w2 = 0   #weight of max_dis
      w3 = -15     #weight of minimal distance to ghost 15 is a 1400 option
      w4 = -50     #weisght of eating a food
      w5 = -500    #weight of eating a capsule
      w6 = 0
      kk = 0
      min_scared = 0
      if is_scared == 1:
          w5 = 500
          if len(scared_dist) >= 2:
           ## w6 = -10       #weight of scared_dist achieved 2500
            w6 = -20
            w1 = 0
            kk = min(scared_dist)
      else:
          w2 = -2
      w7 = 400
      ##if pacman is in the left tunal and eat all the food, it must get out soon.
      if len(cap_dist) > 0:
        t = min(cap_dist)
      else:
        t=0
      if len(curfood) == 1 and len(capsuleplaces) == 0:
          w6 = 0
      evaluation = is_dead * (-1000000)+w1*min_dist + w2*t+ w3*(1/(min_g_dist+0.1)) + w4*(len(curfood)) + w5*(len(capsuleplaces)) + currentGameState.getScore()/float(2.0)+\
                   w6*kk + w7*eat_ghost
      return evaluation
  else:

      is_dead = 0
      for ghost in newGhostStates:
        if ghost.scaredTimer == 0:
            g_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]) == 0:
                is_dead = 1
        else:
            g_dist.append(ghost.scaredTimer*2+abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            scared_dist.append(abs(newPos[0]-ghost.getPosition()[0])+abs(newPos[1]-ghost.getPosition()[1]))
            if abs(newPos[0]-ghost.getPosition()[0])<= 0.5 and abs(newPos[1]-ghost.getPosition()[1]) <= 0.5:
                eat_ghost = 1
            is_scared = 1
      if len(capsuleplaces) > 0:
          count = 0
          judge = 0
          for cap in capsuleplaces:
              count = count + 1
              if cap[0] == 1 and cap[1] == 1:
                  judge = 1
          if judge == 1 and count <= 2 and newPos[0] == 1:
              return 100000
          else:
              return -10000+currentGameState.getScore()/float(5.0)+(-10000)*is_dead+len(capsuleplaces)*(-500)
      else:

        return 1000000+currentGameState.getScore()*200
def inf(action):
    if action == Directions.EAST:
        return Directions.WEST
    if action == Directions.WEST:
        return Directions.EAST
    if action == Directions.NORTH:
        return Directions.SOUTH
    if action == Directions.SOUTH:
        return Directions.NORTH
    if action == Directions.STOP:
        return Directions.STOP

class ContestAgent(MultiAgentSearchAgent):

  def alphabeta(self,state,agent,depth,action,alpha,beta,a1,a2,a3):
    score = ()
    if agent == self.agentCount:
        agent = self.index
    if self.is_finish(state,depth):
        score = (contestEvaluationFunction(state),action)
    elif agent == self.index:
        score = self.maxval(state,agent,depth,alpha,beta,a1,a2,a3)

    else:
        score = self.minval(state,agent,depth,alpha,beta,a1,a2,a3)

    return score

  def maxval(self,state,agent,depth,alpha,beta,a1,a2,a3):
    v = (-999999,Directions.STOP)
    actions = state.getLegalActions(agent)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    for action in actions:
        tempv = (self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta,a1,a2,a3)[0],action)
        v = max(v,tempv)
        if v[0] > beta:
            return v
        alpha = max (alpha,v[0])
    return v

  def minval(self,state,agent,depth,alpha,beta,a1,a2,a3):
    v = (999999,Directions.STOP)

    actions = state.getLegalActions(agent)
    if depth > 4:
        if depth % 4 == 1:
            if a1 in actions and inf(a1) in actions:
                actions.remove(inf(a1))
        if depth % 4 == 2:
            if a2 in actions and inf(a2) in actions:
                actions.remove(inf(a2))
        if depth % 4 == 3:
            if a3 in actions and inf(a3) in actions:
                actions.remove(inf(a3))
    for action in actions:
        if depth % 4 == 1:
            a1 = action
        elif depth % 4 ==2:
            a2 = action
        elif depth % 4 ==3:
            a3 = action
        tempv = (self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta,a1,a2,a3)[0],action)
        v = min(v,tempv)
        if v[0] < alpha:
            return v
        beta = min(beta,v[0])
    return v

  def getAction(self, gameState):

    "*** YOUR CODE HERE ***"
    state = gameState
    self.agentCount = state.getNumAgents()
    self.depth = 5
    if(len(gameState.getFood().asList()) <= 30):   ##1346.59 59/100
       self.depth = 5
    ##all depth 4 : 1352 50/100
    alpha = -999999
    beta = 999999
    depth = 0
    agent = self.index
    action = Directions.STOP
    a1 = Directions.STOP
    a2 = Directions.STOP
    a3 = Directions.STOP
    v = ()
    v = self.alphabeta(state,agent,depth,action,alpha,beta,a1,a2,a3)
    return v[1]
"""
  if len(curfood) == 1:
      ## current score >1800 && no power point
      ## return is_dead *w + dist_to_food + win*w
      ## else:
      ## evaluation = distance to power_point and eat the ghost.
"""