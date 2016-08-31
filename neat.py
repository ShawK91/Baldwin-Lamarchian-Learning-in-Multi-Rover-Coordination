import numpy as np, math, copy, time
import MultiNEAT as NEAT
import mod_rqe as mod, sys
from random import randint, choice


hidden_nodes = 20  # Simulator hidden nodes (also the input to the Evo-net
augmented_input = 1
baldwin = 0
online_learning = 1
update_sim = 1
pre_train = 0
D_reward = 1 #D reward scheme

vizualize = False
share_sim_subpop = 1 #Share simulator within a sub-population
sensor_avg = True #Average distance as state input vs (min distance by default)

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
    coupling = 3 #Number of agents required to simultaneously observe a POI
    total_steps = 30 #Total roaming steps without goal before termination
    num_agents = 4
    num_poi = 2
    angle_res = 90

    agent_rand = 1
    poi_rand = 1
    total_generations = 5000
    wheel_action = 1
    #ABLATION VARS
    if True:
        use_rnn = 0 # Use recurrent instead of normal network
        success_replay  = False
        neat_growth = 0
        angled_repr = True


tracker = mod.statistics()
gridworld = mod.Gridworld (grid_row, grid_col, num_agents, num_poi, agent_rand, poi_rand, angled_repr=angled_repr, angle_res=angle_res, obs_dist=obs_dist, coupling=coupling)  # Create gridworld

def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range(num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn, sensor_avg))
    return first_input, 0, 0


    # if 'action' in locals():
    #     return action
    # else:
    #     return randint(0,4)

# Get rover data for the simulartor
def get_sim_ae_data(num_rounds):  # Get simulation data
    nn_state, steps, tot_reward = reset_board()  # Reset board
    x = []
    for ii in range(num_rounds):
        for steps in range(50):  # One training episode till goal is not reached

            for agent_id in range(num_agents):  # 1 turn per agent
                action = randint(0, 5)  # Get action from the Evo-net
                gridworld.move_and_get_reward(agent_id, action)  # Move gridworld
                x.append(nn_state[agent_id][:])

            # Get new nnstates after all an episode of moves have completed
            for agent_id in range(num_agents):
                nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn, sensor_avg)
            if gridworld.check_goal_complete():
                break
    x = np.array(x)
    x = np.reshape(x, (x.shape[0], x.shape[2]))
    return x

num_rounds = 10000
if pre_train:
    train_x = get_sim_ae_data(num_rounds)
    valid_x = get_sim_ae_data(num_rounds/10)
else:
    train_x = 0; valid_x = 0

#Create simulator lists for each individual in the population
class baldwin_util:
    def __init__(self):
        if share_sim_subpop:
            self.simulator = mod.init_nn(hidden_nodes, angle_res, pre_train, train_x, valid_x)
            self.interim_model = mod.init_nn(hidden_nodes, angle_res, middle_layer=True, weights=self.simulator.layers[0].get_weights())
        else:
            self.simulator = []
            for i in range(params.PopulationSize + 5): self.simulator.append(mod.init_nn(hidden_nodes, angle_res, pre_train, train_x, valid_x))  # Create simulator for each agent
            self.interim_model = mod.init_nn(hidden_nodes, angle_res, middle_layer=True, weights=self.simulator[0].layers[0].get_weights())
        self.traj_x = []; self.traj_y = []  # trajectory for batch learning
        self.best_sim_index = 0

    #Updates the weight to the interim model
    def update_interim_model(self, index):
        #if index < len(self.simulator):
        if share_sim_subpop: weights = self.simulator.layers[0].get_weights()
        else: weights = self.simulator[index].layers[0].get_weights()
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
        w = self.simulator[self.best_sim_index].get_weights()
        for i in range(len(self.simulator)):
            self.simulator[i].set_weights(w)

    # LEARNING PART
    def learning(self, state, pred, index):
        if online_learning and baldwin and update_sim:
            x = np.array(state);
            x = np.reshape(x, (len(x), len(x[0])))
            y = np.array(pred);
            y = np.reshape(y, (len(y), len(y[0])))

            if share_sim_subpop: self.simulator.fit(x, y, verbose=0, nb_epoch=1)
            else: self.simulator[index].fit(x, y, verbose=0, nb_epoch=1)
            self.update_interim_model(index)
        elif baldwin and update_sim:  # Just append the trajectory
            self.traj_x.append(state)
            self.traj_y.append(pred)

class evo_net():
    def __init__(self, evo_input_size, seed):

        if baldwin: self.bald = baldwin_util()
        g = NEAT.Genome(0, evo_input_size, evo_hidden, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, seed, params)  # Constructs genome
        self.pop = NEAT.Population(g, params, True, 1.0, 0)  # Constructs population of genome
        self.pop.RNG.Seed(0)
        self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
        self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations through an iteration
        self.net_list = [[] for x in xrange(len(self.genome_list))] #Stores the networks for the genomes
        self.base_mpc = self.pop.GetBaseMPC()
        self.current_mpc = self.pop.GetCurrentMPC()
        self.delta_mpc = self.current_mpc - self.base_mpc



    def referesh_genome_list(self):
        self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
        self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations throug an iteration
        self.net_list = [[] for x in xrange(len(self.genome_list))]  # Stores the networks for the genomes

    def build_net(self, index):
        if not self.net_list[index]: #if not already built
            self.net_list[index] = NEAT.NeuralNetwork();
            self.genome_list[index].BuildPhenotype(self.net_list[index]);
            self.net_list[index].Flush()  # Build net from genome




    # Get action choice from Evo-net
    def run_evo_net(self, index, state):
        self.net_list[index].Flush()
        self.net_list[index].Input(state)  # can input numpy arrays, too for some reason only np.float64 is supported
        self.net_list[index].Activate()
        scores = []
        for i in range(5):
            if not math.isnan(1 * self.net_list[index].Output()[i]):
                scores.append(1 * self.net_list[index].Output()[i])
            else:
                scores.append(0)
        #print scores
        if wheel_action and sum(scores) != 0: action = mod.roulette_wheel(scores)
        elif sum(scores) != 0: action = np.argmax(scores)
        else: action = randint(0,4)
        #if action == None: action = randint(0, 4)
        return action

    def update_fitness(self): #Update the fitnesses of the genome and also encode the best one for the generation
        best = 0; best_sim_index = 0
        for i, g in enumerate(self.genome_list):
            if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                avg_fitness = sum(self.fitness_evals[i])/len(self.fitness_evals[i])
                if avg_fitness > best:
                    best = avg_fitness;
                    best_sim_index = i
                g.SetFitness(avg_fitness) #Update fitness
                g.SetEvaluated() #Set as evaluated
        if baldwin: self.bald.best_sim_index = best_sim_index #Assign the new top simulator #TODO Generalize this to best performing index and ignore if not evaluated
        self.current_mpc = self.pop.GetCurrentMPC(); self.delta_mpc = self.current_mpc - self.base_mpc #Update MPC's as well

num_evals = 5
def evolve(all_pop):
    best_global = 0
    #NOTE: ALL keyword means all sub-populations
    for i in range(len(all_pop)): all_pop[i].referesh_genome_list() #get new genome list and fitness evaluations trackers
    teams = np.zeros(num_agents).astype(int) #Team definitions by index
    selection_pool = [] #Selection pool listing the individuals with multiples for to match number of evaluations
    for i in range(num_agents): #Filling the selection pool
        selection_pool.append(np.arange(len(all_pop[i].genome_list)))
        for j in range(num_evals - 1): selection_pool[i] = np.append(selection_pool[i], np.arange(len(all_pop[i].genome_list)))

    for evals in range(params.PopulationSize * num_evals): #For all evaluation cycles
        for i in range(len(teams)): #Pick teams
            rand_index = randint(0, len(selection_pool[i])-1) #Get a random index
            teams[i] = selection_pool[i][rand_index] #Pick team member from that index
            selection_pool[i] = np.delete(selection_pool[i], rand_index) #Delete that index
        for i in range(len(teams)): all_pop[i].build_net(teams[i])  # build network for the genomes within the team
        rewards, global_reward = run_simulation(all_pop, teams) #Returns rewards for each memr of the team
        if global_reward > best_global: best_global = global_reward; #Store the best global performance
        for i, sub_pops in enumerate(all_pop): sub_pops.fitness_evals[teams[i]].append(rewards[i]) #Assign those rewards to the members of the team across the sub-populations

    for sub_pop in all_pop:
        sub_pop.update_fitness()# Assign fitness to genomes
        sub_pop.pop.Epoch() #Epoch update method inside NEAT
        if baldwin and update_sim and not share_sim_subpop: sub_pop.bald.port_best_sim()  # Deep copy best simulator
    return best_global

if baldwin: bald = baldwin_util()
####################################### END OF MACROS ####################################

#Reset gridworld

def diff_reward(gridworld):
    rewards = np.zeros(gridworld.num_agents)
    for i in range(gridworld.num_poi):
        if gridworld.goal_complete[i]:
            for ag in gridworld.poi_soft_status[i]:
                rewards[ag] += 1
    return rewards

def run_simulation(all_pop, teams): #Run simulation given a team and return fitness for each individuals in that team
    nn_state, steps, tot_reward = reset_board()  # Reset board
    if online_learning and baldwin: #Update interim model belonging to the teams[i] indexed individual in the ith sub-population
        for i in range(len(all_pop)):
            all_pop[i].bald.update_interim_model(teams[i])
    #rewards = np.zeros(len(teams))

    for steps in range(total_steps):  # One training episode till goal is not reached
        x = []
        for agent_id in range(num_agents):  # 1 turn per agent
            #TODO OPTIMIZE THESE VARIABLES
            state_inp = nn_state[agent_id][:]
            state_inp = state_inp[0][0:-5] #Delete last 5 action elements
            state_inp = np.reshape(state_inp, (1, len(state_inp)))
            if baldwin:
                evo_input = all_pop[agent_id].bald.get_evo_input(nn_state[agent_id]) #Hidden nodes from simulator
                # if augmented_input: evo_input = np.append(evo_input, nn_state[agent_id].flatten()) #Augment input with state info
                if augmented_input: evo_input = np.append(evo_input, state_inp.flatten()) #Augment input with state info
            else:
                evo_input = np.reshape(nn_state[agent_id], (nn_state[agent_id].shape[1])) #State input only (non -baldwin)
            #print 'STate: ',state_inp
            #print 'Evo: ',agent_id, 'NUM', evo_input
            #print
            action = all_pop[agent_id].run_evo_net(teams[agent_id], evo_input) #Get action from the Evo-net
            #action = randint(0,4)
            gridworld.move_and_get_reward(agent_id, action) #Move gridworld
            x.append(nn_state[agent_id][:]); x[agent_id][0][len(x[0])-4+action] = 1 #Code action taken into the x learning target

        #Get new nnstates after all an episode of moves have completed
        for agent_id in range(num_agents):
            nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn, sensor_avg)
        y = (nn_state[agent_id])
        if baldwin:
            for i in range(len(all_pop)):
                all_pop[i].bald.learning(x[i], y, teams[i]) #Learning
        if gridworld.check_goal_complete():
            break
    #TODO Credit assignment with D and/or D++


    #print sum(gridworld.goal_complete)
    g_reward = 1.0 * sum(gridworld.goal_complete)/num_poi #Global Reward
    if D_reward: rewards = diff_reward(gridworld) #D_reward
    else: rewards = np.zeros(gridworld.num_agents) + g_reward #global reward
    return rewards, g_reward






    fitness = rover_sim(net, index)
    return fitness


if __name__ == "__main__":
    mod.dispGrid(gridworld)
    seed = 0 if (evo_hidden == 0) else 1 #Controls sees based on genome initialization
    evo_input_size = hidden_nodes + (360 / angle_res) * 4 if baldwin else 360 * 4 / angle_res #Controls input size to Evo_input
    all_pop = [] #Contains all the evo-net populations for all the teams
    for i in range(num_agents): #Spawn populations of team of agents
        all_pop.append(evo_net(evo_input_size, seed))



    for gen in range (total_generations): #Main Loop
        best_global = evolve(all_pop) #CCEA
        tracker.add_fitness(best_global, gen) #Add best global performance to tracker
        tracker.add_mpc(all_pop) #Update mpc statistics
        print 'Gen:', gen, ' Baldwin' if baldwin else ' Darwinian', 'Online' if online_learning else 'Offline', ' Best global reward', int(best_global * 100), ' Avg:', int(100 * tracker.avg_fitness), ' Delta MPC:', int(tracker.avg_mpc), ' MPC_spread:', int(tracker.mpc_std)

        continue





        #TODO Offline learning
        if not online_learning and baldwin and update_sim: bald.offline_train()










