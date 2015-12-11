# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        print "-----------------------------------------------------"
        "*** MY CODE BEGINS ***"
        k = 0
        while k < iterations:
            val = self.values.copy()  #before each iteration, copy one.
            for s in mdp.getStates():
                if mdp.isTerminal(s) == False:
                    max = -999999
                    for action in mdp.getPossibleActions(s):
                        v = 0
                        for pos_pro in mdp.getTransitionStatesAndProbs(s,action):
                            v = v + pos_pro[1]*(mdp.getReward(s,action,pos_pro[0])+discount*self.values[pos_pro[0]])
                        if v > max:
                            max = v
                    val[s] = max
                else:
                    for action in mdp.getPossibleActions(s):
                        v = 0
                        for pos_pro in mdp.getTransitionStatesAndProbs(s,action):
                            v = v + pos_pro[1]*(mdp.getReward(s,action,pos_pro[0])+discount*self.values[pos_pro[0]])
                        val[s] = v
            k = k+1
            for s in mdp.getStates():
                self.values[s] = val[s]
            #after each iteration, using val to update values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        if self.mdp.isTerminal(state) == False:
            for pos_pro in self.mdp.getTransitionStatesAndProbs(state,action):
                q = q + pos_pro[1]*(self.mdp.getReward(state,action,pos_pro[0])+self.discount*self.values[pos_pro[0]])
            return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        dic = {}
        if self.mdp.isTerminal(state) == False:
            for action in self.mdp.getPossibleActions(state):
                v = self.getQValue(state,action)
                d = {action:v}
                dic.update(d)
            maxx = max(dic.values())             #finds the max value
            keys = [x for x,y in dic.items() if y ==maxx]  #list of all
            return keys[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
