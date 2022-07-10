
import numpy as np

BOARD_ROWS = 4
BOARD_COLS = 4	
DETERMINISTIC = True
WIN_STATE = (3,3)
LOSE_STATE = [(1,1), (1,3), (2,3), (3,0)]
START = (0, 0)

class Env:

	def __init__(self, state, probs, DETERMINISTIC=False):
		self.state = state
		self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
		self.board[WIN_STATE] = 2
		self.probabilities = probs
		for i in LOSE_STATE:
			self.board[i] = -1
		self.isEnd = False
		self.actions = {0:"up", 1:"down", 2:"left", 3:"right"}
		self.deterministic = DETERMINISTIC

	def giveReward(self, state, past_state):
		if state == WIN_STATE and past_state != WIN_STATE:
			return 1
		elif state in LOSE_STATE:
			return 0#-1
		else:
			return 0


	def nxtPosition(self, action, state, prob=1):
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
		Dic_location = self.IdxtoLocation
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

"""class Agent:

	def __init__(self, actions, state):
		self.reward = 0
		self.actions = actions
		self.state = state
		self.memory = []

	def update_state(self, state):
		self.state = state

	def Make_action(self, policy):"""



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


env = Env(START, probs = {0.5:"GoOn", 0.33:"Stay", 0.166:"Reverse"}, DETERMINISTIC=False)
env.showBoard()
MDP = env.P()
V, pi = value_iteration(P=MDP)
print(V)
print_policy(pi, MDP, n_cols=BOARD_COLS)
#agent = Agent(env.actions, env.state)
#print(policy)

