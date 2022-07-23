
import numpy as np
import random
BOARD_ROWS = 4
BOARD_COLS = 4	
DETERMINISTIC = True
WIN_STATE = (3,3)
LOSE_STATE = [(1,1), (1,3), (2,3), (3,0)]
START = (0, 0)

class Env:

	def __init__(self, state, probs, seed = 123, DETERMINISTIC=False):
		self.state = state
		self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
		self.board[WIN_STATE] = 2
		self.probabilities = probs
		for i in LOSE_STATE:
			self.board[i] = -1
		self.isEnd = False
		self.actions = {0:"up", 1:"down", 2:"left", 3:"right"}
		self.deterministic = DETERMINISTIC
		self.MDP = self.P()
		self.Totalreward = 0
		self.episodes = 0
		self.seed = seed

	def giveReward(self, state, past_state):
		if state == WIN_STATE and past_state != WIN_STATE:
			return 1
		elif state in LOSE_STATE:
			return 0#-1
		else:
			return 0


	def nxtPosition(self, action, state, prob="GoOn"):
		"""
		action: up, down, left, right
		-------------
		0 | 1 | 2| 3|
		1 |
		2 |
		return next position
		"""
		if self.deterministic:
			if state == WIN_STATE or state in LOSE_STATE:
				return state
			else:
				if action == "up":
					nxtState = (state[0] - 1, state[1])
				elif action == "down":
					nxtState = (state[0] + 1, state[1])
				elif action == "left":
					nxtState = (state[0], state[1] - 1)
				else:
					nxtState = (state[0], state[1] + 1)
				# if next state legal
				if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
					if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):
						#if nxtState != (1, 1):
						return nxtState
				return state
		else: #If the environment is not deterministic
			if state == WIN_STATE or state in LOSE_STATE:
				return state
			else:
				if prob == "GoOn":#Go to the intended state
					if action == "up":
						nxtState = (state[0] - 1, state[1])
					elif action == "down":
						nxtState = (state[0] + 1, state[1])
					elif action == "left":
						nxtState = (state[0], state[1] - 1)
					else:
						nxtState = (state[0], state[1] + 1)
					# if next state legal
					if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
						if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):
							#if nxtState != (1, 1):
							return nxtState
					return state
				elif prob == "Reverse": #Go reverse of the intended state
					if action == "down":
						nxtState = (state[0] - 1, state[1])
					elif action == "up":
						nxtState = (state[0] + 1, state[1])
					elif action == "right":
						nxtState = (state[0], state[1] - 1)
					else:
						nxtState = (state[0], state[1] + 1)
					# if next state legal
					if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
						if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):
							#if nxtState != (1, 1):
							return nxtState
					return state
				else: #Stay in the same place
					return state

	def isEndFunc(self, state, past_state):
		if (state == WIN_STATE) or (state in LOSE_STATE) or (past_state == WIN_STATE) or (past_state in LOSE_STATE):
			return True
		else:
			return False

	@property
	def IdxtoLocation(self):
		IdxToLoc = {}
		idx = 0
		for i in range(BOARD_ROWS):
			for j in range(BOARD_COLS):
				IdxToLoc[idx] = (i,j)
				idx += 1
		self.DicIdxtoLocation = IdxToLoc
		return IdxToLoc

	def showBoard(self):
		self.board[self.state] = 1
		for i in range(0, BOARD_ROWS):
			print('-----------------')
			out = '| '
			for j in range(0, BOARD_COLS):
				if self.board[i, j] == 1:
					token = '*'
				if self.board[i, j] == 2:
					token = 'V'
				if self.board[i, j] == -1:
					token = 'X'
				if self.board[i, j] == 0:
					token = '0'
				out += token + ' | '
			print(out)
		print('-----------------')

	def P(self):
		MDP = {}
		Dic_location = self.IdxtoLocation #TO FIX, change Dic_location for self.DicIdxtoLocation, it should be part of the environment
		for state in range(BOARD_ROWS*BOARD_COLS):
			MDP[state] = {}
			for action in range(len(self.actions)):
				MDP[state][action] = []

				if self.deterministic:
					Nposition = self.nxtPosition(self.actions[action], Dic_location[state])
					#print(self.actions[action], Dic_location[state], Nposition)
					IdxNposition = list(Dic_location.keys())[list(Dic_location.values()).index(Nposition)]
					#try:
					MDP[state][action] = [(1.0, IdxNposition, self.giveReward(Nposition, Dic_location[state]), self.isEndFunc(Nposition, Dic_location[state]))]
					#except:
					#	MDP[state] = {action : [(1.0, IdxNposition, self.giveReward(Nposition, Dic_location[state]), self.isEndFunc(Nposition, Dic_location[state]))]}
				else:
					for prob in self.probabilities.keys():
						Nposition = self.nxtPosition(self.actions[action], Dic_location[state], self.probabilities[prob])
						#print(self.actions[action], Dic_location[state], Nposition)
						IdxNposition = list(Dic_location.keys())[list(Dic_location.values()).index(Nposition)]
						#try:
						MDP[state][action] += [(prob, IdxNposition, self.giveReward(Nposition, Dic_location[state]), self.isEndFunc(Nposition, Dic_location[state]))]
						#except:
						#	MDP[state] = {action : [(prob, IdxNposition, self.giveReward(Nposition, Dic_location[state]), self.isEndFunc(Nposition, Dic_location[state]))]}
						

		return MDP

	def reset(self):
		self.state = START
		self.isEnd = False
		return self.state

	def step(self, action: int):
		#Make a step in the environment. For deterministic game, simply make the desired action
		#For a stochastic game, generate a random number and check what action corresponds to that number according to the games probs
		#action (int): action to be made
		if self.deterministic:
			nxt_state = nxtPosition(action = action, state = self.state)
		else:
			n = np.random.random()
			sum_ = 0
			for p in self.probabilities:
				if sum_ >= n or sum_ <= sum_+p:
					prob_action = self.probabilities[p]
					break
				else:
					sum_ += p
		#Transform from coordinates of the game (.,.) to idx
		try:
			#If IdxtoLocation has already been built
			idx_state = list(self.DicIdxtoLocation.keys())[list(self.DicIdxtoLocation.values()).index(self.state)]
		except:
			#If not, build it
			dic = self.IdxtoLocation
			idx_state = list(self.DicIdxtoLocation.keys())[list(self.DicIdxtoLocation.values()).index(self.state)]

		action_idx = self.Actiontoidx(action, prob_action) #Transform probs of actions to definitive actions
		
		resulted_env_stats = self.MDP[idx_state][action_idx]
		prob, next_state, reward, done = resulted_env_stats[0]
		past_state = action_idx
		self.state = self.IdxtoLocation[next_state]
		self.isEnd = done
		self.Totalreward += reward
		self.episodes += 1
		return past_state, action, reward, next_state #Return experience tuple
		#return next_state, self.isEndFunc(next_state), 


	def Actiontoidx(self, action: int, prob_action: str):
		#Function to return the action depending of the probs of the environment
		if self.deterministic:
			return action
		else:
			if len(self.actions) == 4:
				if prob_action == "GoOn":
					return action
				elif prob_action == "Reverse":
					if action == 1:
						return 0
					elif action == 0:
						return 1
					elif action == 2:
						return 3
					else:
						return 2
				else:
					return None
			else:
				if len(self.actions) == 2:
					if prob_action == "GoOn":
						return action
					elif prob_action == "Reverse":
						if action == 1:
							return 0
						else:
							return 1
					else:
						return None

	def set_seed(self, seed):
		self.seed = seed
		np.random.seed(seed)


class Agent:

	def __init__(self, actions, state, policy):
		#Initialize the agent. For now, the agent has a defined policy (there is no learning here)
		self.Accumulated_reward = 0
		self.Nepisodes = 0
		self.action_space = actions
		self.state = state
		self.experience = []
		self.policy = policy

	def update_state(self, state):
		#Update the state of the agent
		self.state = state

	def update_AccumulatedReward(self, reward):
		#Update the accumulated reward of the agent
		self.Accumulated_reward += reward

	def update_lifetime(self):
		#Update the episodes the agent has been active
		self.Nepisodes += 1

	def update_experience(self, state, action, reward, next_state):
		#Update the entire experience tuple
		self.experience.append((state, action, reward, next_state))
		self.state = next_state
		self.Accumulated_reward += reward
		self.Nepisodes += 1

	def Make_action(self, policy):
		return policy(self.state)




def value_iteration(P, gamma=1.0, theta=1e-10):
	V = np.zeros((len(P)), dtype="float64")
	while True:
		Q = np.zeros((len(P), len(P[0])), dtype="float64")

		for s in range(len(P)):
			for a in range(len(P[s])):
				for prob, next_state, reward, done in P[s][a]:
					Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
		if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
			break
		#print(Q)
		#print(V)
		V = np.max(Q, axis=1)
	pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
	return V, pi


def print_policy(pi, P, action_symbols=('^', 'v', '<', '>'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
	random.seed(123); np.random.seed(123) #; env.seed(123)
	results = []
	for _ in range(n_episodes):
		state, done, steps = env.reset(), False, 0
		state = list(env.DicIdxtoLocation.keys())[list(env.DicIdxtoLocation.values()).index(state)]
		while not done and steps < max_steps:
			past_state, _, _, state = env.step(pi(state))
			done = env.isEndFunc(env.DicIdxtoLocation[state], env.DicIdxtoLocation[past_state])
			steps += 1
		results.append(state == list(env.DicIdxtoLocation.values()).index(goal_state))
	return np.sum(results)/len(results)


env = Env(START, probs = {0.5:"GoOn", 0.33:"Stay", 0.166:"Reverse"}, DETERMINISTIC=False) #TO FIX: the keys of the dict in probs should be the actions and the numbers should be the cases. The problem is all of them have the same prob, you are overwriting the dict
#env.showBoard()
MDP = env.P()
print(env.state)
#print(env.IdxtoLocation)
#env.step(1)
#print(env.state)

V, pi = value_iteration(P=MDP)
#print(V)
#print_policy(pi, MDP, n_cols=BOARD_COLS)
#agent = Agent(env.actions, env.state)
#print(policy)

