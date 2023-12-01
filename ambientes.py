import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

class ABC():
    
    def __init__(self):
        self.nA = 2
        self.action_space = [0,1]
        self.nS = 3
        self.A = 0
        self.B = 1
        self.C = 2
        self.LEFT = 0
        self.RIGHT = 1
        P = {}
        P[self.A] = {a:[] for a in range(self.nA)}
        P[self.A][self.LEFT] = [(1, self.A, -1, False)]
        P[self.A][self.RIGHT] = [(0.1, self.A, -1, False), (0.9, self.B, -1, False)]
        P[self.B] = {a:[] for a in range(self.nA)}
        P[self.B][self.LEFT] = [(1, self.A, -1, False)]
        P[self.B][self.RIGHT] = [(0.1, self.B, -1, False), (0.9, self.C, 10, True)]
        P[self.C] = {a:[] for a in range(self.nA)}
        self.P = P
        self.dict_acciones = {self.LEFT:'LEFT', self.RIGHT:'RIGHT'}
        self.dict_states = {self.A:'A', self.B:'B', self.C:'C'}
        self.p_right = 0.9
        self.state = self.A
        
    def reset(self):
        self.state = self.A
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        str(f'Estado: {self.state}')

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {self.dict_states[s]}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{self.dict_states[x[1]]}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string

class GridworldEnv():
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Termial states are top left and
    the bottom right corner.
    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reaches a terminal state.
    """

    def __init__(self, shape=(4,4)):
        assert(shape[0] == shape[1])
        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))
        self.state = np.random.randint(1, self.nS - 2)
        self.NORTH = 0
        self.WEST = 1
        self.SOUTH = 2
        self.EAST = 3
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            position = self._State2Car(s)
            P[s][self.NORTH] = self._transition_prob(position, [0, 1])
            P[s][self.WEST] = self._transition_prob(position, [-1, 0])
            P[s][self.SOUTH] = self._transition_prob(position, [0, -1])
            P[s][self.EAST] = self._transition_prob(position, [1, 0])
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"⬆", 1:"⬅", 2:"⬇", 3:"➡"}
        self.proportion = 6

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = np.clip(coord[0], 0, self.shape[0] - 1)
        coord[1] = np.clip(coord[1], 0, self.shape[1] - 1)
        return coord

    def _Car2State(self, casilla:tuple) -> int:
        X, Y = casilla
        return np.ravel_multi_index((Y, X), self.shape)

    def _State2Car(self, index:int) -> tuple:
        Y, X = np.unravel_index(index, self.shape)
        return (X, Y)

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (x, y)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        # if stuck in terminal state
        current_state = self._Car2State(current)
        if current_state == self._Car2State((self.shape[0] - 1, 0)) or current_state == self._Car2State((0, self.shape[1] - 1)):
            return [(1.0, current_state, 0, True)]
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = self._Car2State(new_position)
        is_done = new_state == self._Car2State((self.shape[0] - 1, 0)) or new_state == self._Car2State((0, self.shape[1] - 1))
        return [(1.0, new_state, -1, is_done)]

    def reset(self):
        self.state = np.random.randint(1, self.nS - 2)
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def _find_figsize(self):
        x, y = self.shape
        if x == y:
            return (self.proportion,self.proportion)
        elif x > y:
            return (int(self.proportion*(x/y)),self.proportion)
        else:
            return (self.proportion,int(self.proportion*(y/x)))
        
    def _find_offset(self):
        return 1/(self.shape[0]*2), 1/(self.shape[1]*2)

    def render(self):
        # Dibuja el laberinto
        fig, axes = plt.subplots(figsize=self._find_figsize())
        # Dibujo el tablero
        step_x = 1./self.shape[0]
        step_y = 1./self.shape[1]
        tangulos = []
        # Borde del tablero
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                            facecolor='xkcd:sky blue',\
                                            edgecolor='black',\
                                            linewidth=1))
        offsetX, offsetY = self._find_offset()
        #Poniendo las salidas
        for casilla in [(0,self.shape[1]-1), (self.shape[0]-1,0)]:
            X, Y = casilla
            arr_img = plt.imread("./imagenes/salida.png", format='png')
            image_salida = OffsetImage(arr_img, zoom=0.05)
            image_salida.image.axes = axes
            ab = AnnotationBbox(
                image_salida,
                [(X*step_x) + offsetX, (Y*step_y) + offsetY],
                frameon=False)
            axes.add_artist(ab)
		# Creo las líneas del tablero
        for j in range(self.shape[1]):
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, j * step_y), 1, 0.008],\
            facecolor='black'))
        for j in range(self.shape[0]):
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(j * step_x, 0), 0.008, 1],\
            facecolor='black'))
        for t in tangulos:
            axes.add_patch(t)
        #Poniendo agente
        Y, X = np.unravel_index(self.state, self.shape)
        imagen_robot = "./imagenes/robot.png"
        arr_img = plt.imread(imagen_robot, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.125)
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(X*step_x) + offsetX, (Y*step_y) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        axes.axis('off')
        plt.show()
        return axes

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s} at {np.unravel_index(s, self.shape)}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    Y, X = np.unravel_index(x[1], self.shape)
                    string += f'new_state:{x[1]} at ({X}, {Y}), '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    

class Dado():
    
    def __init__(self):
        self.nA = 2
        self.action_space = [0,1]
        self.nS = 2
        self.p_end = 1/3
        self.state = 0
        self.STAY = 0
        self.QUIT = 1
        P = {}
        P[0] = {a:[] for a in range(self.nA)}
        P[0][self.STAY] = [(1-self.p_end, 0, 4, False), (self.p_end, 1, 4, True)]
        P[0][self.QUIT] = [(1, 1, 10, True)]
        P[1] = {a:[(1,1,0,True)] for a in range(self.nA)}
        self.P = P
        self.dict_actions = {0:'STAY', 1:'QUIT'}
        self.dict_states = {0:'IN', 1:'END'}
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        print(f'Estado: {self.estado}')

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState:{self.dict_states[s]}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_actions[a]}'
                for x in self.P[s][a]:
                    print(x)
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{x[1]}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
        

class RandomWalkEnv():
    """
    A Random Walk environment from Sutton's Reinforcement 
    Learning book chapter 6. Initial state is the center state.
    Termial states are the left-most and right-most cells.
    Actions are (RIGHT=0, LEFT=1).
    Reward of 0 at each step, except right-most cell, 
    where reward is +1.
    """

    def __init__(self, shape:int=7):
        self.shape = shape
        self.nS = self.shape
        self.nA = 2
        self.action_space = list(range(self.nA))
        self.state = int(self.shape/2)
        self.EAST = 0
        self.WEST = 1
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            P[s][self.EAST] = self._transition_prob(s, 1)
            P[s][self.WEST] = self._transition_prob(s, -1)
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"►", 1:"◄"}

    def _transition_prob(self, current_state, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        if current_state == self.nS - 1:
            return [(1.0, current_state, 1, True)]
        if current_state == 0:
            return [(1.0, current_state, 0, True)]
        new_state = current_state + delta
        new_state = self._limit_coordinates(new_state)
        if new_state == self.nS - 1:
            return [(1.0, new_state, 1, True)]
        if new_state == 0:
            return [(1.0, new_state, 0, True)]
        return [(1.0, new_state, 0, False)]

    def _limit_coordinates(self, state):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        state = min(state, self.nS - 1)
        state = max(state, 0)
        return state
    
    def reset(self):
        self.state = int(self.shape/2)
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        state = self.state
        output = ''
        for s in range(self.nS):
            if s == state:
                if state == 0 or state == self.nS - 1:
                    output += '@'
                else:
                    output += "x"
            # Print terminal state
            elif s == 0 or s == self.nS - 1:
                output += "o"
            else:
                output += "_"
        print(output)

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{x[1]}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    
    
class WindyGridworldEnv():
    """
    A 4x4 Windy Grid World environment from Sutton's Reinforcement 
    Learning book chapter 6. Termial states are top left and
    the bottom right corner.
    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reachs a terminal state.
    vertical wind will shift result state by a number of
    cells proportional to wind strength.
    """

    def __init__(self, shape=(4,4)):
        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))
        self.wind_strength = [self._wind_strength(col) for col in range(self.shape[1])]
        self.state = np.ravel_multi_index((int(self.shape[0]/2),0), self.shape)
        self.objective = np.ravel_multi_index((int(self.shape[0]/2),int(self.shape[0]*2/3) + 1), self.shape)
        self.NORTH = 0
        self.EAST = 1
        self.SOUTH = 2
        self.WEST = 3
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            position = np.unravel_index(s, self.shape)
            P[s][self.NORTH] = self._transition_prob(position, [-1, 0])
            P[s][self.EAST] = self._transition_prob(position, [0, 1])
            P[s][self.SOUTH] = self._transition_prob(position, [1, 0])
            P[s][self.WEST] = self._transition_prob(position, [0, -1])
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"▲", 1:"►", 2:"▼", 3:"◄"}

    def _wind_strength(self, x):
        numerator = 3
        denominator = 1 + (abs(self.shape[1]*2/3 - x - 1))*0.65
        return int(numerator / denominator)
    
    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        # if stuck in terminal state
        current_state = np.ravel_multi_index(tuple(current), self.shape)
        if current_state == self.objective:
            return [(1.0, current_state, 0, True)]
        new_position = np.array(current) + np.array(delta) - np.array([self.wind_strength[current[1]], 0])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = new_state == self.objective
        return [(1.0, new_state, -1, is_done)]

    def reset(self):
        self.state = np.ravel_multi_index((int(self.shape[0]/2),0), self.shape)
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        state = self.state
        output = ''
        for s in range(self.nS):
            if s == state:
                if state == self.objective:
                    output += '@'
                else:
                    output += "x"
            # Print terminal state
            elif s == self.objective:
                output += "o"
            else:
                output += "_"
            position = np.unravel_index(s, self.shape)
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'
        print(output)

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s} at {np.unravel_index(s, self.shape)}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{x[1]} at {np.unravel_index(x[1], self.shape)}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    
class CliffworldEnv():
    """
    A 4x12 Cliff World environment from Sutton's Reinforcement 
    Learning book chapter 6. Initial state is the bottom left corner.
    Termial state is the bottom right corner and all the bottom cells.
    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step, except bottom cliff, where reward is -100.
    """

    def __init__(self, shape:tuple=(4,12)):
        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))
        self.state = self.nS - self.shape[1]
        P = {}
        self.NORTH = 0
        self.EAST = 1
        self.SOUTH = 2
        self.WEST = 3
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            position = np.unravel_index(s, self.shape)
            P[s][self.NORTH] = self._transition_prob(position, [-1, 0])
            P[s][self.EAST] = self._transition_prob(position, [0, 1])
            P[s][self.SOUTH] = self._transition_prob(position, [1, 0])
            P[s][self.WEST] = self._transition_prob(position, [0, -1])
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"▲", 1:"►", 2:"▼", 3:"◄"}

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _transition_prob(self, current_position, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        # if stuck in terminal state
        current_state = np.ravel_multi_index(tuple(current_position), self.shape)
        if current_state == self.nS - 1:
            return [(1.0, current_state, 0, True)]
        if current_position[0] == self.shape[0] - 1 and 0 < current_position[1] < self.shape[1] - 1:
            return [(1.0, current_state, -100, True)]
        new_position = np.array(current_position) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if new_state == self.nS - 1:
            return [(1.0, new_state, -1, True)]
        if new_position[0] == self.shape[0] - 1  and 0 < new_position[1] < self.shape[1] - 1:
            return [(1.0, new_state, -100, True)]
        return [(1.0, new_state, -1, False)]

    def reset(self):
        self.state = self.nS - self.shape[1]
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        state = self.state
        output = ''
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if s == state:
                if position[0] == self.shape[0] - 1 and position[1] != 0:
                    output += '@'
                else:
                    output += "x"
            # Print terminal state
            elif s == self.nS - 1:
                output += "o"
            # Print cliff
            elif position[0] == self.shape[0] - 1 and position[1] != 0:
                output += '#'
            else:
                output += "_"
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'
        print(output)

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s} at {np.unravel_index(s, self.shape)}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{x[1]} at {np.unravel_index(x[1], self.shape)}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string