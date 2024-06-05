# qlearningAgents.py͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# ------------------͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# Licensing Information:  You are free to use or extend these projects for͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# educational purposes provided that (1) you do not distribute or publish͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# solutions, (2) you retain this notice, and (3) you provide clear͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# The core projects and autograders were primarily created by John DeNero͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# Student side autograding was added by Brad Miller, Nick Hay, and͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
# Pieter Abbeel (pabbeel@cs.berkeley.edu).͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)] #creating a map with 2 keys


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Follows exact logic as computeValueFromQValues as value iteration
        max_q = []
        for action in self.getLegalActions(state):
            q = self.getQValue(state, action)
            max_q.append(q)
        if not self.getLegalActions(state):
            return 0.0
        return max(max_q)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Follows exact same logic as in value iteration but this one uses all the legal actions
        actions = []
        best = 0
        for action in self.getLegalActions(state):
            v = self.getQValue(state, action)
            actions.append(v)
            if v == max(actions):
                best = action
        return best

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        prob = self.epsilon #getting prob to put into random.choice
        actions = []

        if not legalActions: #if there are no legal actions
            return None
        for a in legalActions: #go through all the legal actions
            actions.append(a)
        if util.flipCoin(prob): #if it will take a random action
            return random.choice(actions)
        else: #normal best action
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        d = (1 - self.alpha) #following part of the bellman equation
        curr_q = self.getQValue(state, action)

        #setting up getting the next reward
        rewards = []
        for next_action in self.getLegalActions(nextState):
            next_q = self.getQValue(nextState, next_action)
            rewards.append(next_q)
        if rewards: #if there are no rewards, then it will be 0
            next_reward = max(rewards)
        else:
            next_reward = 0 

        dis_next_reward = self.discount * next_reward
        self.q_values[(state, action)] = d * curr_q + self.alpha * (reward + dis_next_reward) #full approximate bellman update equation
        return curr_q
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
        PacmanQAgent.final(self, state)

        # did we finish training?͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging͏󠄂͏️͏󠄌͏󠄎͏󠄎͏︇͏󠄌
            "*** YOUR CODE HERE ***"
            pass
