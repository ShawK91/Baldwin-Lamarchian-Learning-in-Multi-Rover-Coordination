
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import SReLU
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
import math
import numpy as np, time
import random



def save_model(model):
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    model.save_weights('shaw_2/model_weights.h5',overwrite=True)

class Gridworld:
    def __init__(self, dim_row = 10, dim_column = 10, num_agents = 2, num_poi = 2, agent_rand = False, poi_rand = False, angle_res = 10, angled_repr = False, obs_dist = 1, coupling = 2):
        self.observe = 1; self.dim_row = dim_row; self.dim_col = dim_column
        self.num_agents = num_agents; self.num_poi = num_poi; self.angle_res = angle_res #Angle resolution
        self.angled_repr = angled_repr #Angled state representation
        if num_agents > 1: self.coupling = coupling #coupling requirement
        else: self.coupling = 1
        self.obs_dist = obs_dist #Observation radius requirements
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()

        #Resettable stuff
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        self.poi_pos = []; self.goal_complete = []; self.poi_obs = []
        for i in range(self.num_poi):
            self.init_poi(poi_rand) #Initialize POIs
            self.goal_complete.append(False) #Check if goal is complete
            self.poi_obs.append([]) #Track the identity of agents within the coupling requirements at all applicable time steps

        self.agent_pos = []
        for i in range(self.num_agents):
            self.init_agent(agent_rand)
        #self.optimal_steps = self.get_optimal_steps()
        #self.update_soft_states()


    def init_wall(self):
        for i in range(self.observe):
            for x in range(self.state.shape[0]):
                self.state[x][i] = 3
                self.state[x][self.state.shape[1] - 1-i] = 3
            for y in range(self.state.shape[1]):
                self.state[i][y] = 3
                self.state[self.state.shape[0] - 1-i][y] = 3

    def check_spawn(self, x, y):
        for i in range(len(self.poi_pos)):
            if x == self.poi_pos[i][0] and y == self.poi_pos[i][1]:  # Check to see for other POI
                return False
        try:
            for i in range(len(self.agent_pos)):
                if x == self.agent_pos[i][0] and y == self.agent_pos[i][1]:  # Check to see for other agent
                    return False
        except:
            1 + 1
        return True

    def init_poi(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1
        rad = int(self.dim_row/math.sqrt(3)/2)
        center = int((start + end)/2)
        if rand_start:
            while True:
                rand = random.random()
                if rand < 0.25:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif rand < 0.5:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif rand < 0.75:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad - 1)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad + 1, end)
                if self.check_spawn(x, y):
                    break

        else:
            x = self.state.shape[0] - self.observe-1 ;y = self.state.shape[1] - self.observe-1 #Goals at ends
            #x = self.state.shape[0]/2 + 1 ;y = self.state.shape[1]/2 #Goals in middle
            if len(self.poi_pos) > 0:
                x = self.observe ;
                y = self.state.shape[1] - self.observe - 1
                # for i in range(len(self.poi_pos)):
                #     if x == self.poi_pos[i][0] and y == self.poi_pos[i][1]:  # Check to see for other agent
                #         y = self.state.shape[0] - self.observe - 2
                #         x = self.observe + 1
        self.state[x][y] = 2
        self.poi_pos.append([x, y])
        return [x,y]

    def init_agent(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1
        rad = int(self.dim_row/math.sqrt(3)/3)
        center = int((start + end)/2)
        if rand_start:
            while True:
                x = randint(center - rad, center + rad)
                y = randint(center - rad, center + rad)
                if self.check_spawn(x, y):
                    break
        else: #Not random
            while True:
                if len(self.agent_pos) == 0:
                    x = start; y = start
                if len(self.agent_pos) == 1:
                    x = start + (end - start) / 2; y = start
                if len(self.agent_pos) == 2:
                    x = end; y = start
                if len(self.agent_pos) > 2:
                    x = start + len(self.agent_pos); y = start
                for i in range(len(self.agent_pos)):
                    if x == self.agent_pos[i][0] and y == self.agent_pos[i][1]:  # Check to see for other POIs
                        continue
                break
        self.state[x][y] = 1 #Agent Code
        self.agent_pos.append([x,y])
        return [x,y]

    def reset(self, agent_rand, poi_rand):
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        self.poi_pos = []; self.goal_complete = []; self.poi_obs = []
        for i in range(self.num_poi):
            self.init_poi(poi_rand) #Initialize POIs
            self.goal_complete.append(False) #Check if goal is complete
            self.poi_obs.append([]) #Track the identity of agents within the coupling requirements at all applicable time steps

        self.agent_pos = []
        for i in range(self.num_agents):
            self.init_agent(agent_rand)
        #self.optimal_steps = self.get_optimal_steps()
        #self.update_soft_states()

    def move(self, all_actions):
        for agent_id in range(self.num_agents): #Move all agents (first agent has priority in case of conflict)
            action = all_actions[agent_id]
            next_pos = np.copy(self.agent_pos[agent_id])
            if action == 1: next_pos[1] += 1  # Right
            elif action == 2: next_pos[0] += 1  # Down
            elif action == 3: next_pos[1] -= 1  # Left
            elif action == 4: next_pos[0] -= 1  # Up

            # Computer reward and check illegal moves
            x = next_pos[0]; y = next_pos[1]
            if self.state[x][y] == 3: next_pos[0] = self.agent_pos[agent_id][0]; next_pos[1] = self.agent_pos[agent_id][1] # Wall
            if self.state[x][y] == 1 and action != 0: next_pos[0] = self.agent_pos[agent_id][0]; next_pos[1] = self.agent_pos[agent_id][1] # Other Agent
            if self.state[x][y] == 2 and action != 0: next_pos[0] = self.agent_pos[agent_id][0]; next_pos[1] = self.agent_pos[agent_id][1] #POI

            # Update gridworld and agent position
            if self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] != 2:
                self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] = 0
            if self.state[next_pos[0]][next_pos[1]]!= 2:
                self.state[next_pos[0]][next_pos[1]] = 1
            self.agent_pos[agent_id][0] = next_pos[0]
            self.agent_pos[agent_id][1] = next_pos[1]

    def update_poi_observations(self):
        #Check for credit assignment
        for poi_id in range(self.num_poi): # POI COUPLED
            soft_stat = []
            for ag in range(self.num_agents):
                if abs(self.poi_pos[poi_id][0] - self.agent_pos[ag][0]) <= self.obs_dist and abs(self.poi_pos[poi_id][1] - self.agent_pos[ag][1]) <= self.obs_dist: # and self.goal_complete[poi_id] == False:
                    soft_stat.append(ag)
            if len(soft_stat) >= self.coupling: #If coupling requirement is met
                self.goal_complete[poi_id] = True
                self.poi_obs[poi_id].append(soft_stat) #Store the identity of agents aiding in meeting that tight coupling requirement





        # for poi_id in range(self.num_poi): # POI COUPLED
        #     if self.goal_complete[poi_id] == False:
        #         try:
        #             self.poi_soft_status[poi_id].remove(agent_id)
        #         except: 1+1
        #         if abs(self.poi_pos[poi_id][0] - self.agent_pos[agent_id][0]) <= self.obs_dist and abs(self.poi_pos[poi_id][1] - self.agent_pos[agent_id][1]) <= self.obs_dist:
        #             self.poi_soft_status[poi_id].append(agent_id)
        #         if len(self.poi_soft_status[poi_id]) >= self.coupling:
        #             self.goal_complete[poi_id] = True

    def update_soft_states(self): #Reset soft states
        self.poi_soft_status = [] #Reset soft_status
        for i in range(self.num_poi):
            self.poi_soft_status.append([])
        for poi_id in range(self.num_poi): # POI COUPLED
            for ag in range(self.num_agents):
                if abs(self.poi_pos[poi_id][0] - self.agent_pos[ag][0]) <= self.obs_dist and abs(self.poi_pos[poi_id][1] - self.agent_pos[ag][1]) <= self.obs_dist and self.goal_complete[poi_id] == False:
                    self.poi_soft_status[poi_id].append(ag)

    def get_state(self, agent_id, sensor_avg):  # Returns a flattened array around the agent position
        if self.angled_repr: #If state representation uses angle
            st = self.angled_state(agent_id, sensor_avg)
            st = np.append(st, 0) #Add action to the state
            st = np.append(st, 0)  # Add action to the state
            st = np.append(st, 0)  # Add action to the state
            st = np.append(st, 0)  # Add action to the state
            st = np.append(st, 0)  # Add action to the state
            st = np.reshape(st, (1,len(st)))
            return st
        else: #DEPRECATED - IGNORE (BINARY ENCODING REPRESENTATION)
            x_beg = self.agent_pos[agent_id][0] - self.observe
            y_beg = self.agent_pos[agent_id][1] - self.observe
            x_end = self.agent_pos[agent_id][0] + self.observe + 1
            y_end = self.agent_pos[agent_id][1] + self.observe + 1
            st = np.copy(self.state)
            st = st[x_beg:x_end, :]
            st = st[:, y_beg:y_end]
            st = np.reshape(st, (1, pow(self.observe * 2 + 1, 2))) # Flatten array
            #return st
            k = np.reshape(np.zeros(len(st[0]) * 4), (len(st[0]), 4)) #4-bit encoding
            for i in range(len(st[0])):
                k[i][int(st[0][i])] = 1
            k = np.reshape(k, (1, len(st[0]) * 4))  # Flatten array

            return k

    def get_first_state(self, agent_id, use_rnn, sensor_avg):  # Get first state, action input to the q_net
        if not use_rnn: #Normal NN
            st = self.get_state(agent_id, sensor_avg)
            return st

        rnn_state = []
        st = self.get_state(agent_id)
        for time in range(3):
            rnn_state.append(st)
        rnn_state = np.array(rnn_state)
        rnn_state = np.reshape(rnn_state, (1, rnn_state.shape[0], rnn_state.shape[2]))
        return rnn_state

    def referesh_state(self, current_state, agent_id, use_rnn):
        st = self.get_state(agent_id)
        if use_rnn:
            new_state = np.roll(current_state, -1, axis=1)
            new_state[0][2] = st
            return new_state
        else:
            return st

    def check_goal_complete(self):
        goal_complete = True #Check if all agents found POI's
        for poi_id in range(self.num_poi):
            goal_complete *= self.goal_complete[poi_id]
        return  goal_complete

    def get_optimal_steps(self):
        #TODO MAKE MATCHING GENERALIZABLE
        return self.dim_col + self.dim_row
        steps = []
        for i in range(self.num_poi):
            ig = []
            for agent_id in range(self.num_agents):
                ig.append(abs(self.poi_pos[i][0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[i][1] - self.agent_pos[agent_id][1]))
            steps.append(ig)
        opt_steps = 10000000000
        for i in range(self.num_poi):
            for j in range(self.num_poi):
                step = steps[j]

                # if abs(self.poi_pos[i][0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[i][1] - self.agent_pos[agent_id][1])  > opt_steps:
                #     opt_steps = abs(self.poi_pos[0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[1] - self.agent_pos[agent_id][1])
        return opt_steps

    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two agents relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  # dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        #angle = math.degrees(angle)
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    def angled_state(self, agent_id, sensor_avg):
        state = np.zeros(self.num_agents *2 + self.num_poi *2)
        if sensor_avg: #Average distance
            dist_poi_list = [[] for x in xrange(360/self.angle_res)]
            dist_agent_list = [[] for x in xrange(360 / self.angle_res)]

        for id in range(self.num_poi):
            if True: #self.goal_complete[id] == False: #For all POI's that are still active
                x1 = self.poi_pos[id][0] - self.agent_pos[agent_id][0]; x2 = 1
                y1 = self.poi_pos[id][1] - self.agent_pos[agent_id][1]; y2 = 0
                angle, dist = self.get_angle_dist(x1,y1,x2,y2)
                state[2*id] = angle
                state[2*id+1] = dist/(2.0*self.dim_col)

                #bracket = int(angle / self.angle_res)
                #state[bracket][0] += 1.0/self.num_poi #Add POIs
                #if sensor_avg: dist_poi_list[bracket].append(dist/(2.0*self.dim_col))
                #else: #Min distance
                    #if state[bracket][1] > dist/(2.0*self.dim_col) or state[bracket][1] == 0:  # Update min distance from POI
                        #state[bracket][1] = dist/(2.0*self.dim_col)


        for id in range(self.num_agents):
            if id != agent_id: #FOR ALL AGENTS MINUS MYSELF
                x1 = self.agent_pos[id][0] - self.agent_pos[agent_id][0]; x2 = 1
                y1 = self.agent_pos[id][1] - self.agent_pos[agent_id][1]; y2 = 0
                angle, dist = self.get_angle_dist(x1,y1,x2,y2)
                state[2*self.num_poi+2 * id] = angle
                state[2*self.num_poi + 2 * id + 1] = dist / (2.0 * self.dim_col)
                # bracket = int(angle / self.angle_res)
                # state[bracket][2] += 1.0/(self.num_agents-1) #Add agent
                # if sensor_avg: dist_agent_list[bracket].append(dist/(2.0*self.dim_col))
                # else: #Min distance
                #     if state[bracket][3] > dist/(2.0*self.dim_col) or state[bracket][3] == 0: #Update min distance from other agent
                #         state[bracket][3] = dist/(2.0*self.dim_col)

        if sensor_avg:
            for bracket in range(len(dist_agent_list)):

                try: state[bracket][1] = sum(dist_poi_list[bracket])/len(dist_poi_list[bracket]) #Encode average POI distance
                except: None
                try: state[bracket][3] = sum(dist_agent_list[bracket]) / len(dist_agent_list[bracket])  # Encode average POI distance
                except: None

        state = np.reshape(state, (1, self.num_agents *2 + self.num_poi *2)) #Flatten array
        return state

class statistics(): #Tracker
    def __init__(self):
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.avg_mpc = 0; self.tr_avg_mpc = []; self.mpc_std = []; self.tr_mpc_std = []


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 200:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            self.save_csv(generation)

    def add_mpc(self, all_pop):
        all_mpc = np.zeros(len(all_pop))
        for i, sub_pop in enumerate(all_pop): all_mpc[i] = sub_pop.delta_mpc
        self.avg_mpc = np.average(all_mpc) #Average mpc
        self.mpc_std = np.std(all_mpc)


    def save_csv(self, generation):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt('avg_fitness.csv', np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

def init_rnn(gridworld, hidden_nodes, angled_repr, angle_res, sim_all, hist_len = 3, design = 1):
    model = Sequential()
    if angled_repr:
        sa_sp = (360/angle_res) * 4
    else:
        sa_sp = (pow(gridworld.observe * 2 + 1,2)*4) #BIT ENCODING
    if design == 1:
        model.add(LSTM(hidden_nodes, init= 'he_uniform', return_sequences=False, input_shape=(hist_len, sa_sp), inner_init='orthogonal', forget_bias_init='one', inner_activation='sigmoid'))#, activation='sigmoid', inner_activation='hard_sigmoid'))
    elif design == 2:
        model.add(SimpleRNN(hidden_nodes, init='he_uniform', input_shape=(hist_len, sa_sp), inner_init='orthogonal'))
    elif design == 3:
        model.add(GRU(hidden_nodes, init='he_uniform', consume_less= 'cpu',  input_shape=(hist_len, sa_sp),inner_init='orthogonal'))
    #model.add(Dropout(0.1))
    #model.add(LeakyReLU(alpha=.2))
    model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.add(Activation('sigmoid'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dense(1, init= 'he_uniform'))
    model.compile(loss='mse', optimizer='Nadam')
    return model

def init_nn(hidden_nodes, angle_res, sim_all, gridworld, pretrain=False, train_x = 0, valid_x = 0, middle_layer = False, weights = 0):
    model = Sequential()
    # if sim_all: sa_sp = (360/angle_res) * 4 + 5
    # else: sa_sp = (360/angle_res) * 2 + 5
    sa_sp = gridworld.num_agents * 2 + gridworld.num_poi * 2 + 5
    if middle_layer:
        model.add(Dense(hidden_nodes, input_dim=sa_sp, weights=weights, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    else:
        model.add(Dense(hidden_nodes, input_dim=sa_sp, init='he_uniform', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(LeakyReLU(alpha=.2))
    #model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.1))
    #model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if not middle_layer:
        model.add(Dense(sa_sp, init= 'he_uniform'))
    model.compile(loss='mse', optimizer=sgd)

    if pretrain: #Autoencoder pretraining
        model.fit(train_x, train_x, nb_epoch=50, batch_size=32, shuffle=True, validation_data=(valid_x, valid_x),
                        verbose=1)

    return model

def dev_EvaluateGenomeList_Parallel(genome_list, evaluator, cores=4, display=True, ipython_client=None):
    #''' If ipython_client is None, will use concurrent.futures.
    #Pass an instance of Client() in order to use an IPython cluster '''
    fitnesses = []
    curtime = time.time()

    #if ipython_client is None:# or not ipython_installed:
    with ProcessPoolExecutor(max_workers=cores) as executor:
        for i, fitness in enumerate(executor.map(evaluator, genome_list)):
            fitnesses += [fitness]

            if display:
                #if ipython_installed: clear_output(wait=True)
                print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), fitness))
    #else:

        # if type(ipython_client) == Client:
        #     lbview = ipython_client.load_balanced_view()
        #     amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
        #     for i, fitness in enumerate(amr):
        #         if display:
        #             #if ipython_installed: clear_output(wait=True)
        #             print('Individual:', i, 'Fitness:', fitness)
        #         fitnesses.append(fitness)
        # else:
        #     raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return fitnesses

def q_values(hist_input, q_model):
    # action_ind = hist_input.shape[2] - 1
    # for i in range(5):
    #     k[i][2][action_ind] = i
    return q_model.predict(hist_input)

def test_hist(q_model):
    gridworld = Gridworld()
    agent2 = Agent(gridworld, True)
    hist_input = get_first_hist(gridworld, agent2)
    return q_model.predict(hist_input)

def dispGrid(gridworld, state = None, full=True, agent_id = None):


    if state == None: #Given agentq
        if full:
            st = np.copy(gridworld.state)
        else:
            x_beg = gridworld.agent_pos[agent_id][0] - gridworld.observe
            y_beg = gridworld.agent_pos[agent_id][1] - gridworld.observe
            x_end = gridworld.agent_pos[agent_id][0] + gridworld.observe + 1
            y_end = gridworld.agent_pos[agent_id][1] + gridworld.observe + 1
            st = np.copy(gridworld.state)
            st = st[x_beg:x_end,:]
            st = st[:,y_beg:y_end]
    else:
        st = []
        print len(state)
        row_leng = int(math.sqrt(len(state)))
        for i in range(row_leng):
            ig = []
            for j in range(row_leng):
                ig.append(state[i*row_leng + j])
            st.append(ig)

    grid = [["-" for i in range(len(st))] for i in range(len(st))]
    grid[0][0] = "o"
    for i in range(len(st)):
        for j in range(len(st)):
            if st[i][j] == 2:
                grid[i][j] = '$'
            if st[i][j] == 1:
                grid[i][j] = '*'
            if st[i][j] == 3:
                grid[i][j] = '#'
    for row in grid:
        for e in row:
            print e,
        print '\t'


def bck_dispGrid(gridworld, state = None, full=True, agent_id = None):

    if state == None: #Given agentq
        if full:
            st = np.copy(gridworld.state)
        else:
            x_beg = gridworld.agent_pos[agent_id][0] - gridworld.observe
            y_beg = gridworld.agent_pos[agent_id][1] - gridworld.observe
            x_end = gridworld.agent_pos[agent_id][0] + gridworld.observe + 1
            y_end = gridworld.agent_pos[agent_id][1] + gridworld.observe + 1
            st = np.copy(gridworld.state)
            st = st[x_beg:x_end,:]
            st = st[:,y_beg:y_end]
    else:
        st = []
        print len(state)
        row_leng = int(math.sqrt(len(state)))
        for i in range(row_leng):
            ig = []
            for j in range(row_leng):
                ig.append(state[i*row_leng + j])
            st.append(ig)

    grid = [["-" for i in range(len(st))] for i in range(len(st))]
    grid[0][0] = "o"
    for i in range(len(st)):
        for j in range(len(st)):
            if st[i][j] == 2:
                grid[i][j] = '$'
            if st[i][j] == 1:
                grid[i][j] = '*'
            if st[i][j] == 3:
                grid[i][j] = '#'
    for row in grid:
        for e in row:
            print e,
        print

def test_q_table(q_table, gridworld, agent1):
    steps = 0
    tot_reward = 0
    while True:  # Till goal is not reached
        steps += 1
        table_pos = [agent1.position[0] - gridworld.observe,
                     agent1.position[1] - gridworld.observe]  # Transform to work q-table
        action = np.argmax(q_table[table_pos[0]][[table_pos[1]]])
        # Get Reward and move
        reward = move_and_get_reward(gridworld, agent1, action)
        tot_reward += reward

        if gridworld.poi_pos[0] == agent1.position[0] and gridworld.poi_pos[1] == agent1.position[1]:  # IF POI is met
            break
        if steps > 100:
            break
    return tot_reward, steps



def load_model_architecture(seed='Models/architecture.json'):  # Get model architecture
    import yaml
    from keras.models import model_from_yaml
    with open('Models/architecture.yaml', 'r') as f:
        yaml_string = yaml.load(f)
    model_arch = model_from_yaml(yaml_string)
    return model_arch

def save_model_architecture(qmodel, foldername = '/Models/'):
    import os, yaml
    import json
    from keras.models import model_from_json
    from keras.models import model_from_yaml
    #Create folder to store all networks if not present
    filename = os.getcwd() + foldername
    if not os.path.exists(os.path.dirname(filename)):
        try: os.makedirs(os.path.dirname(filename))
        except: 1+1
    yaml_string = qmodel.to_yaml()
    output_stream = open("Models/architecture.yaml", "w")
    yaml.dump(yaml_string, output_stream)#, default_flow_style=False)




def load_model(foldername = 'Models/'):
    import copy
    q_model = []
    model_arch = load_model_architecture()
    for i in range(5):
        ig = copy.deepcopy(model_arch)
        ig.load_weights(foldername + 'model_weights_' + str(i) + '.h5')
        q_model.append(ig)
        #q_model[i].compile(loss='mse', optimizer='rmsprop')
    return q_model

def save_qmodel(q_model, foldername = '/Models/'):
    import os
    #Create folder to store all networks if not present
    filename = os.getcwd() + foldername
    if not os.path.exists(os.path.dirname(filename)):
        try: os.makedirs(os.path.dirname(filename))
        except: 1+1
    #Save weights
    for i in range(len(q_model)):
        q_model[i].save_weights('Models/model_weights_' + str(i) + '.h5', overwrite=True)

def roulette_wheel(scores):
    scores = scores / np.sum(scores)  # Normalize
    rand = random.random()
    counter = 0
    for i in range(len(scores)):
        counter += scores[i]
        if rand < counter:
            return i

#BACKUPS
def bck_move_and_get_reward(self, agent_id, action):
    next_pos = np.copy(self.agent_pos[agent_id])
    if action == 1:
        next_pos[1] += 1  # Right
    elif action == 2:
        next_pos[0] += 1  # Down
    elif action == 3:
        next_pos[1] -= 1  # Left
    elif action == 4:
        next_pos[0] -= 1  # Up

    # Computer reward and check illegal moves
    reward = 0  # If nothing else
    x = next_pos[0]
    y = next_pos[1]
    if self.state[x][y] == 3:  # Wall
        next_pos[0] = self.agent_pos[agent_id][0];
        next_pos[1] = self.agent_pos[agent_id][1]
        # reward = -0.0001
    if self.state[x][y] == 1 and action != 0:  # Other Agent
        # reward = -0.05
        next_pos[0] = self.agent_pos[agent_id][0];
        next_pos[1] = self.agent_pos[agent_id][1]
        # if self.state[x][y] == 0 or (self.state[x][y] == 1 and action == 0):  # Free Space
        # reward = -0.0001
    if self.state[x][y] == 2 and action != 0:  # POI
        next_pos[0] = self.agent_pos[agent_id][0];
        next_pos[1] = self.agent_pos[agent_id][1]

    # Update gridworld and agent position
    if self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] != 2:
        self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] = 0
    if self.state[next_pos[0]][next_pos[1]] != 2:
        self.state[next_pos[0]][next_pos[1]] = 1
    self.agent_pos[agent_id][0] = next_pos[0]
    self.agent_pos[agent_id][1] = next_pos[1]

    # Check for credit assignment
    for poi_id in range(self.num_poi):  # POI COUPLED
        if self.goal_complete[poi_id] == False:
            try:
                self.poi_soft_status[poi_id].remove(agent_id)
            except:
                1 + 1
            if abs(self.poi_pos[poi_id][0] - self.agent_pos[agent_id][0]) <= self.obs_dist and abs(
                            self.poi_pos[poi_id][1] - self.agent_pos[agent_id][1]) <= self.obs_dist:
                self.poi_soft_status[poi_id].append(agent_id)
                reward += 1.0 - 1.0 / (len(self.poi_soft_status[poi_id]) + 1)
            if len(self.poi_soft_status[poi_id]) >= self.coupling:
                self.goal_complete[poi_id] = True
                reward = 1

    return reward


def test_nets():
    from fann2 import libfann
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import LSTM, GRU, SimpleRNN
    from keras.layers.advanced_activations import LeakyReLU, PReLU, SReLU
    from keras.models import model_from_json
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD, Nadam
    test_x = np.arange(50)
    model = Sequential()
    model.add(Dense(50, input_dim=50, init='he_uniform'))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(1, init= 'he_uniform'))
    model.compile(loss='mse', optimizer='Nadam')
    ann = libfann.neural_net()
    ann.create_standard_array([3, 50, 50, 1])
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

    curtime = time.time()
    for i in range (1000000):
        ann.run(test_x)
    elapsed = time.time() - curtime
    print elapsed

    test_x = np.reshape(test_x, (1,50))
    curtime = time.time()
    for i in range (1000000):
        model.predict(test_x)
    elapsed = time.time() - curtime
    print elapsed


def bck_angled_state(self, agent_id, sensor_avg):
    state = np.zeros(((360 / self.angle_res), 4))
    if sensor_avg:  # Average distance
        dist_poi_list = [[] for x in xrange(360 / self.angle_res)]
        dist_agent_list = [[] for x in xrange(360 / self.angle_res)]

    for id in range(self.num_poi):
        if self.goal_complete[id] == False:  # For all POI's that are still active
            x1 = self.poi_pos[id][0] - self.agent_pos[agent_id][0];
            x2 = 1
            y1 = self.poi_pos[id][1] - self.agent_pos[agent_id][1];
            y2 = 0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            bracket = int(angle / self.angle_res)
            state[bracket][0] += 1.0 / self.num_poi  # Add POIs
            if sensor_avg:
                dist_poi_list[bracket].append(dist / (2.0 * self.dim_col))
            else:  # Min distance
                if state[bracket][1] > dist / (2.0 * self.dim_col) or state[bracket][
                    1] == 0:  # Update min distance from POI
                    state[bracket][1] = dist / (2.0 * self.dim_col)

    for id in range(self.num_agents):
        if id != agent_id:  # FOR ALL AGENTS MINUS MYSELF
            x1 = self.agent_pos[id][0] - self.agent_pos[agent_id][0];
            x2 = 1
            y1 = self.agent_pos[id][1] - self.agent_pos[agent_id][1];
            y2 = 0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            bracket = int(angle / self.angle_res)
            state[bracket][2] += 1.0 / (self.num_agents - 1)  # Add agent
            if sensor_avg:
                dist_agent_list[bracket].append(dist / (2.0 * self.dim_col))
            else:  # Min distance
                if state[bracket][3] > dist / (2.0 * self.dim_col) or state[bracket][
                    3] == 0:  # Update min distance from other agent
                    state[bracket][3] = dist / (2.0 * self.dim_col)

    if sensor_avg:
        for bracket in range(len(dist_agent_list)):

            try:
                state[bracket][1] = sum(dist_poi_list[bracket]) / len(
                    dist_poi_list[bracket])  # Encode average POI distance
            except:
                None
            try:
                state[bracket][3] = sum(dist_agent_list[bracket]) / len(
                    dist_agent_list[bracket])  # Encode average POI distance
            except:
                None

    state = np.reshape(state, (1, 360 / self.angle_res * 4))  # Flatten array
    return state