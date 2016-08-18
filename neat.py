import numpy as np, math, copy, time
import MultiNEAT as NEAT
import mod_rqe as mod
from random import randint

hidden_nodes = 20  # Simulator hidden nodes (also the input to the Evo-net
augmented_input = 1
baldwin = 1
online_learning = 1
update_sim = 1
wheel_action = 1
vizualize = False

if True:
    params = NEAT.Parameters()
    params.PopulationSize = 100
    fs_neat = False
    evo_hidden = 0
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.EliteFraction = 0.05
    params.RecurrentProb = 0.2
    params.RecurrentLoopProb = 0.2

    params.MaxWeight = 5
    params.MutateAddNeuronProb = 0.005
    params.MutateAddLinkProb = 0.005
    params.MutateRemLinkProb = 0.01
    params.MutateAddLinkProb = 0.03
    params.MutateRemSimpleNeuronProb = 0.0005
    params.MutateNeuronActivationTypeProb = 0.005

    params.ActivationFunction_SignedSigmoid_Prob = 0.01
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.5
    params.ActivationFunction_Tanh_Prob = 0.1
    params.ActivationFunction_SignedStep_Prob = 0.1


#ROVER DOMAIN MACROS
if True: #Macros
    grid_row = 15
    grid_col = 15
    obs_dist = 1 #Observe distance (Radius of POI that agents have to be in for successful observation)
    coupling = 1 #Number of agents required to simultaneously observe a POI
    total_steps = 20 #Total roaming steps without goal before termination
    num_agents = 3
    num_poi = 10
    angle_res = 45

    agent_rand = 1
    poi_rand = 1
    total_generations = 5000
    #ABLATION VARS
    if True:
        use_rnn = 0 # Use recurrent instead of normal network
        success_replay  = False
        neat_growth = 0
        angled_repr = True


#Create simulator lists for each individual in the population
class baldwin_util:
    def __init__(self):
        self.simulator = []
        for i in range(params.PopulationSize + 5):
            self.simulator.append(mod.init_nn(hidden_nodes, angle_res))  # Create simulator for each agent
        self.interim_model = mod.init_nn(hidden_nodes, angle_res, True, self.simulator[0].layers[0].get_weights())
        self.traj_x = []; self.traj_y = []  # trajectory for batch learning
        self.best_sim_index = 0

    #Updates the weight to the interim model
    def update_interim_model(self, index):
        if index < len(self.simulator):
            weights = self.simulator[index].layers[0].get_weights()
            self.interim_model.layers[0].set_weights(weights)

    # Get the inputs to the Evo-net (extract hidden nodes from the sim-net)
    def get_evo_input(self, input):  # Extract the hidden layer representatiuon that will be the input to the EvoNet
        evo_inp = self.interim_model.predict(input)
        evo_inp = np.reshape(evo_inp, (len(evo_inp[0])))
        return evo_inp

    #Train simulator offline at the end
    def offline_train(self):
        x = np.array(self.traj_x);
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[3]))
        y = np.array(self.traj_y);
        y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[3]))
        self.simulator[self.best_sim_index].fit(x, y, verbose=0, nb_epoch=1)
        self.update_interim_model()

    #Copy the best eprforming candidate's simulator to next generation
    def port_best_sim(self):
        # for i in range(len(self.simulator)):
        #     self.simulator[i] = copy.deepcopy(self.simulator[self.best_sim_index])
        w = self.simulator[self.best_sim_index].get_weights()
        for i in range(len(self.simulator)):
            self.simulator[i].set_weights(w)

    # LEARNING PART
    def learning(self, state, pred, index):
        if online_learning and baldwin and update_sim:
            x = np.array(state);
            x = np.reshape(x, (len(x), len(x[0][0])))
            y = np.array(pred);
            y = np.reshape(y, (len(y), len(y[0][0])))
            self.simulator[index].fit(x, y, verbose=0, nb_epoch=1)
            self.update_interim_model(index)
        elif baldwin and update_sim:  # Just append the trajectory
            self.traj_x.append(state)
            self.traj_y.append(pred)

tracker = mod.statistics()
gridworld = mod.Gridworld (grid_row, grid_col, num_agents, num_poi, agent_rand, poi_rand, angled_repr=angled_repr, angle_res=angle_res, obs_dist=obs_dist, coupling=coupling)  # Create gridworld
if baldwin: bald = baldwin_util()
####################################### END OF MACROS ####################################

#Reset gridworld
def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range(num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn))
    return first_input, 0, 0

#Get action choice from Evo-net
def run_evo_net(net, state):
    net.Flush()
    net.Input(state)  # can input numpy arrays, too for some reason only np.float64 is supported
    net.Activate()
    scores = []
    for i in range(5):
        if not math.isnan(1 * net.Output()[i]):
            scores.append(1 * net.Output()[i])
        else:
            scores.append(0)
    if wheel_action and sum(scores) != 0:
        action = mod.roulette_wheel(scores)
    else:
        action = np.argmax(scores)
    if action == None: action = randint(0,4)
    return action
    # if 'action' in locals():
    #     return action
    # else:
    #     return randint(0,4)

#Simulate rovers exploration for a full episode
def rover_sim(net, index):
    nn_state, steps, tot_reward = reset_board()  # Reset board
    #print interim_model.predict(nn_state[0])[0][0]
    #mod.dispGrid(gridworld),
    if online_learning and baldwin: bald.update_interim_model(index)
    for steps in range(total_steps):  # One training episode till goal is not reached
        x = []
        y = []
        for agent_id in range(num_agents):  # 1 turn per agent
            state_inp = nn_state[agent_id][:]
            state_inp = state_inp[0][0:-5] #Delete last 5 action elements
            state_inp = np.reshape(state_inp, (1, len(state_inp)))
            if baldwin:
                evo_input = bald.get_evo_input(nn_state[agent_id]) #Hidden nodes from simulator
                # if augmented_input: evo_input = np.append(evo_input, nn_state[agent_id].flatten()) #Augment input with state info
                if augmented_input: evo_input = np.append(evo_input, state_inp.flatten()) #Augment input with state info
            else:
                evo_input = np.reshape(nn_state[agent_id], (nn_state[agent_id].shape[1])) #State input only (non -baldwin)

            action = run_evo_net(net, evo_input) #Get action from the Evo-net
            #action = randint(0,5)
            #print action

            # Get Reward and move
            #if vizualize:
                #print action,
            gridworld.move_and_get_reward(agent_id, action) #Move gridworld

            x.append(nn_state[agent_id][:]); x[agent_id][0][len(x[0])-4+action] = 1 #Code action taken into the x learning target

        #Get new nnstates after all an episode of moves have completed
        for agent_id in range(num_agents):
            nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn)
            y.append(nn_state[agent_id])
        if baldwin: bald.learning(x, y, index) #Learning
        if gridworld.check_goal_complete():
            break
    tot_reward = 1.0 * sum(gridworld.goal_complete)/num_poi
    return tot_reward

#Fitness function evaluation for a genome
def evaluate(genome, index):
    net = NEAT.NeuralNetwork(); genome.BuildPhenotype(net); net.Flush() #Build net from genome
    fitness = rover_sim(net, index)
    return fitness


if __name__ == "__main__":
    mod.dispGrid(gridworld)

    # all_pop = []
    # for i in range(num_agents):
    #     if baldwin:
    #         g = NEAT.Genome(i, hidden_nodes, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
    #                         NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    #     else:
    #         g = NEAT.Genome(i, 360 * 4 / angle_res, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
    #                         NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    #     all_pop.append(NEAT.Population(g, params, True, 1.0, 0))
    #     all_pop[i].RNG.Seed(0)
    seed = 0 if (evo_hidden == 0) else 1 #Controls sees based on genome initialization
    evo_input_size = hidden_nodes + (360 / angle_res) * 4 if baldwin else 360 * 4 / angle_res #Controls input size to Evo_input
    print evo_input_size
    g = NEAT.Genome(0, evo_input_size, evo_hidden, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, seed, params) #Constructs genome
    pop = NEAT.Population(g, params, True, 1.0, 0) #Constructs population of genome
    pop.RNG.Seed(0)

    for gen in range (total_generations):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list, id_list, best_index = mod.dev_EvaluateGenomeList_Serial(genome_list, evaluate, display=0)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()

        if not online_learning and baldwin and update_sim: bald.offline_train()
        if baldwin and update_sim: bald.port_best_sim()  #Deep copy best simulator
        best_fit = max(fitness_list); tracker.add_fitness(best_fit, gen)
        print 'Gen:', gen, ' Baldwin' if baldwin else ' Darwinian', 'Online' if online_learning else 'Offline',   ' Max fit percent:', int(best_fit * 100), ' Avg:', int(100*tracker.avg_fitness), 'Delta MPC', int(pop.GetCurrentMPC() - pop.GetBaseMPC())









