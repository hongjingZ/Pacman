# inference.py

import util
import random
import game
import sys
import capture

class InferenceModule:
  def __init__(self):
    numParticles=300
    self.setNumParticles(numParticles)
    self.Captured=False
    self.moveList=[(0,1),(0,-1),(1,0),(-1,0),(0,0)]

  def setNumParticles(self, numParticles):
    self.numParticles = numParticles

  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)

  def initializeUniformly(self, gameState):
    "Initializes a list of particles. Use self.numParticles for the number of particles"
    self.Particles=[]
    for i in range(self.numParticles):
        pos1=random.choice(self.legalPositions)
        pos2=random.choice(self.legalPositions)
        self.Particles.append((pos1,pos2))

  def observe(self, noisyDistance, gameState,agentID):
    """
    Update beliefs based on the given distance observation.
    What if a ghost was eaten by agent?
    The former assumption will be reinitialized, which is apparently unnecssary.
    We need to find the method which can determine whether a certain agent is eaten, then like "go to jail", we just put them in the inital pos.gameState.getInitialAgentPosition(agentID)
    """
    AgentPosition = gameState.getAgentPosition(agentID)
    weights=[1 for i in range(self.numParticles)]
    for index in range(self.numParticles):
        for i in range(2):
            trueDistance=util.manhattanDistance(self.Particles[index][i],AgentPosition)
            weights[index]*=gameState.getDistanceProb(trueDistance,noisyDistance[i])

    if sum(weights)==0:
        self.initializeUniformly(gameState)
        return
    else:
        newParticals=util.nSample(weights,self.Particles,self.numParticles)
        self.Particles=newParticals

  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.
    """

    newParticles = []
    for oldParticle in self.Particles:
        newParticle = list(oldParticle) # A list of ghost positions
        for i in range(2):#for each ghost. In fact, only one enemy moves between two elasptime. Try to figure out whih enemy move will improve the efficiency
            #Hint, the index of the enmey which just moved is less 1 than this agen.
            counter=0
            pos=newParticle[i]
            newPosList=[]
            newPosDistribution=util.Counter()
            for move in self.moveList:
                newPos=(pos[0]+move[0],pos[1]+move[1])
                if not gameState.hasWall(newPos[0],newPos[1]):
                    newPosList.append(newPos)
                    counter+=1 # assume all the action for this enemy is equally possible
            for newPos in newPosList:
                newPosDistribution[newPos]=1.0/counter# in this form, we can use sample method
            newParticle[i]=(util.sample(newPosDistribution))
        newParticles.append(tuple(newParticle))

    self.particles = newParticles

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    belief=util.Counter()
    for p in self.Particles:
        for pos in p:
            belief[pos]+=1;

    belief.divideAll(self.numParticles)

    return belief

