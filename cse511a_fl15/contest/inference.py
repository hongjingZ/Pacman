# inference.py

import util
import random
import game
import sys
import capture

class InferenceModule:
  def __init__(self):
    numParticles=10000
    self.setNumParticles(numParticles)
    self.Captured=False
    self.moveList=[(0,1),(0,-1),(1,0),(-1,0)]
    self.enemies=[]

  def setNumParticles(self, numParticles):
    self.numParticles = numParticles

  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)

  def initializeUniformly(self, gameState):
    "Initializes a list of particles. Use self.numParticles for the number of particles"
    print "initialzed particles"
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
            weights[index]*=gameState.getDistanceProb(trueDistance,noisyDistance[self.enemies[i]])

    if sum(weights)==0:
        self.initializeUniformly(gameState)
        return
    else:
        newParticals=util.nSample(weights,self.Particles,self.numParticles)
        self.Particles=newParticals

  def elapseTime(self, gameState,agentID):
    """
    Update beliefs for a time step elapsing.
    """
    enemyID=((agentID+3)%4)/2 #(agentID+4-1)%4/2 calculating which agent just moved
    #print "agentID="+str(agentID)+"enemyID="+str(enemyID)
    newParticles = []
    for oldParticle in self.Particles:
        newParticle = list(oldParticle) # A list of ghost positions
        pos=newParticle[enemyID] # certain enemy's position
        newPosDistribution=util.Counter()
        for move in self.moveList: # get every move
            newPos=(pos[0]+move[0],pos[1]+move[1]) # get the new position
            if newPos in self.legalPositions: # if the posistion is illeagle, ingore it.
                newPosDistribution[newPos]=1
        newPosDistribution.normalize()
        newParticle[enemyID]=(util.sample(newPosDistribution))
        newParticles.append(tuple(newParticle))

    self.particles = newParticles

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    belief has two couter, store the enemies' position repectively
    """
    belief=[util.Counter(),util.Counter()]
    for p in self.Particles:
            belief[0][p[0]]+=1
            belief[1][p[1]]+=1

    belief[0].divideAll(self.numParticles*1.0) #this 1.0 may be unnecessarily
    belief[1].divideAll(self.numParticles*1.0)

    return belief

