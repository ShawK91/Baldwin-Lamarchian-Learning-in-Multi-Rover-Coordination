#!/usr/bin/python3
import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial, EvaluateGenomeList_Parallel
from MultiNEAT import GetGenomeList, ZipFitness
from concurrent.futures import ProcessPoolExecutor, as_completed
import mod_rqe as mod


def dev_EvaluateGenomeList_Serial(genome_list, evaluator, display=True):
    fitnesses = []
    id_list = []
    count = 0
    curtime = time.time()
    for g in genome_list:
        id_list.append(g.GetID())
        #print g.GetID()

        f = evaluator(g)
        fitnesses.append(f)

        if display:
            #if ipython_installed: clear_output(wait=True)
            print('Individuals: (%s/%s) Fitness: %3.4f' % (count, len(genome_list), f))
        count += 1

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %s' % elapsed)

    return fitnesses, id_list

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


#NEAT parameters
params = NEAT.Parameters()
params.PopulationSize = 100
if False:
    params = NEAT.Parameters()
    params.PopulationSize = 100
    params.DynamicCompatibility = True
    params.WeightDiffCoeff = 4.0
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = True
    params.EliteFraction = 0.05
    params.RecurrentProb = 0.5
    params.OverallMutationRate = 0.8

    params.MutateWeightsProb = 0.90

    params.WeightMutationMaxPower = 2.5
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.25

    params.MaxWeight = 8

    params.MutateAddNeuronProb = 0.05
    params.MutateAddLinkProb = 0.05
    params.MutateRemLinkProb = 0.0

    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9

    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 0.0

    params.CrossoverRate = 0.75  # mutate only 0.25
    params.MultipointCrossoverRate = 0.4
    params.SurvivalRate = 0.2

#ROVER DOMAIN MACROS
if True: #Macros
    grid_row = 15
    grid_col = 15
    obs_dist = 1 #Observe distance (Radius of POI that agents have to be in for successful observation)
    coupling = 2 #Number of agents required to simultaneously observe a POI
    observability = 1 #DEPRECATED
    hidden_nodes = 4
    total_steps = 50 #Total roaming steps without goal before termination
    num_agents = 3
    num_poi = 5
    angle_res = 45

    agent_rand = 1
    poi_rand = 1

    #ABLATION VARS
    use_rnn = 0 # Use recurrent instead of normal network
    success_replay  = False
    neat_growth = 0
    use_prune = True #Prune duplicates
    angled_repr = True
    temperature = 0.1
    illustrate = False
    total_generations = 5000

    if not use_rnn:
        hidden_nodes *= 5  # Normalize RNN and NN flexibility

baldwin = 0
online_learning = 0
update_sim = True

simulator = mod.init_nn(hidden_nodes, angle_res) #Create simulator
gridworld = mod.Gridworld (grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr=angled_repr, angle_res=angle_res, obs_dist=obs_dist, coupling=coupling)  # Create gridworld
sim_x = []; sim_y = [] #trajectory for batch learning
interim_model = mod.init_nn(hidden_nodes, angle_res, True, simulator.layers[0].get_weights())

####################################### END OF MACROS ####################################

class statistics():
    def __init__(self):
        self.train_goal_met = 0
        self.reward_matrix = np.zeros(10) + 10000
        self.coverage_matrix = np.zeros(10) + 10000
        self.tr_reward_mat = []
        self.tr_coverage_mat = []
        #self.coverage_matrix = np.zeros(100)
        self.tr_reward = []
        self.tr_coverage = []
        self.tr_reward_mat = []

    def save_csv(self, reward, coverage, train_epoch):
        self.tr_reward.append(np.array([train_epoch, reward]))
        self.tr_coverage.append(np.array([train_epoch, coverage]))

        np.savetxt('reward.csv', np.array(self.tr_reward), fmt='%.3f', delimiter=',')
        np.savetxt('coverage.csv', np.array(self.tr_coverage), fmt='%.3f', delimiter=',')

        self.reward_matrix = np.roll(self.reward_matrix, -1)
        self.reward_matrix[-1] = reward
        self.coverage_matrix = np.roll(self.coverage_matrix, -1)
        self.coverage_matrix[-1] = coverage
        if self.reward_matrix[0] != 10000:
            self.tr_reward_mat.append(np.array([train_epoch, np.average(self.reward_matrix)]))
            np.savetxt('reward_matrix.csv', np.array(self.tr_reward_mat), fmt='%.3f', delimiter=',')
            self.tr_coverage_mat.append(np.array([train_epoch, np.average(self.coverage_matrix)]))
            np.savetxt('coverage_matrix.csv', np.array(self.tr_coverage_mat), fmt='%.3f', delimiter=',')

def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range(num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn))
    return first_input, 0, 0

def run_evo_net(net, state):
    net.Flush()
    net.Input(state)  # can input numpy arrays, too for some reason only np.float64 is supported
    net.Activate()
    score = -1000000
    for i in range(5):
        if score < 1 * net.Output()[i]:
            action = i
            score = net.Output()[i]
    return action

def get_evo_input(input): #Extract the hidden layer representatiuon that will be the input to the EvoNet
    weights = simulator.layers[0].get_weights()
    #interim_model = mod.init_nn(hidden_nodes, angle_res, True, weights)
    interim_model.layers[0].set_weights(weights)
    evo_inp = interim_model.predict(input)
    evo_inp = np.reshape(evo_inp, (len(evo_inp[0])))
    return evo_inp

def rover_sim(net):

    prev_action = 0.0
    nn_state, steps, tot_reward = reset_board()  # Reset board
    #mod.dispGrid(gridworld),
    for steps in range(total_steps):  # One training episode till goal is not reached
        for agent_id in range(num_agents):  # 1 turn per agent\

            prev_nn_state = nn_state[:]  # Backup current state input
            if baldwin:
                evo_input = get_evo_input(nn_state[agent_id])
            else:
                evo_input = np.reshape(nn_state[agent_id], (360 * 4 / angle_res + 1))


            #Get action from the Evo-net
            action = run_evo_net(net, evo_input)
            #print action
            prev_action = action

            # Get Reward and move
            _ = gridworld.move_and_get_reward(agent_id, action)

            # Update current state
            nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn, prev_action)

            #LEARNING PART
            if online_learning and baldwin and update_sim:
                x = prev_nn_state[agent_id]
                y = np.delete(nn_state[agent_id], -1, 1)
                simulator.fit(x, y, verbose=0, nb_epoch=1)
            elif baldwin and update_sim: #Just append the trajectory
                sim_x.append(prev_nn_state[agent_id])
                sim_y.append(np.delete(nn_state[agent_id], -1, 1))

        tot_reward = 1.0 * sum(gridworld.goal_complete)/num_poi
        if gridworld.check_goal_complete():
            # print 'GOAL MET'
            #tracker.train_goal_met += 1
            break
            # END OF ONE ROUND OF SIMULATION
    #mod.dispGrid(gridworld)
    #print
    return tot_reward

def evaluate(genome):
    net = NEAT.NeuralNetwork(); genome.BuildPhenotype(net); net.Flush() #Build net from genome
    fitness = rover_sim(net)
    return fitness


if __name__ == "__main__":
    if baldwin:
        g = NEAT.Genome(0, hidden_nodes, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    else:
        g = NEAT.Genome(0, 360 * 4 / angle_res, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    pop.RNG.Seed(0)


    for gen in range (total_generations):
        if gen == 500:
            update_sim = False
        genome_list = NEAT.GetGenomeList(pop)
        sim_x = []; sim_y = []
        fitness_list, id_list = dev_EvaluateGenomeList_Serial(genome_list, evaluate, display=0)
        #fitness_list= dev_EvaluateGenomeList_Parallel(genome_list, evaluate, display=1)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        print 'Gen: ', gen, '  Baldwin' if baldwin else '  Darwinian', 'Online' if online_learning else 'Offline',   '    Max fitness: ', max(fitness_list)#, id_list[fitness_list.index(max(fitness_list))]
        if not online_learning and baldwin and update_sim:
            x= np.array(sim_x); x = np.reshape(x, (len(x), len(x[0][0])) )
            y= np.array(sim_y); y = np.reshape(y, (len(y), len(y[0][0])) )
            simulator.fit(x, y, verbose=0, nb_epoch=1)








