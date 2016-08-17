import numpy as np, math
import MultiNEAT as NEAT
import mod_rqe as mod

#NEAT parameters
params = NEAT.Parameters()
params.PopulationSize = 100
fs_neat = False
evo_hidden = 20
params.MutateAddNeuronProb = 0.0
params.MutateAddLinkProb = 0.0
params.MutateRemLinkProb = 0.01
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

    params.MutateAddNeuronProb = 0.0
    params.MutateAddLinkProb = 0.0
    params.MutateRemLinkProb = 0.1

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
    coupling = 3 #Number of agents required to simultaneously observe a POI
    hidden_nodes = 10 #Simulator hidden nodes (also the input to the Evo-net
    total_steps = 15 #Total roaming steps without goal before termination
    num_agents = 6
    num_poi = 2
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


#GLOBAL
augmented_input = 1
tracker = mod.statistics()
baldwin = 1
online_learning = 1
update_sim = 1
wheel_action = 1


simulator = mod.init_nn(hidden_nodes, angle_res) #Create simulator
gridworld = mod.Gridworld (grid_row, grid_col, num_agents, num_poi, agent_rand, poi_rand, angled_repr=angled_repr, angle_res=angle_res, obs_dist=obs_dist, coupling=coupling)  # Create gridworld
sim_x = []; sim_y = [] #trajectory for batch learning
interim_model = mod.init_nn(hidden_nodes, angle_res, True, simulator.layers[0].get_weights())
vizualize = False
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
    return action
    # if 'action' in locals():
    #     return action
    # else:
    #     return randint(0,4)

#Updates the weight to the interim model
def update_interim_model():
    weights = simulator.layers[0].get_weights()
    interim_model.layers[0].set_weights(weights)

#Get the inputs to the Evo-net (extract hidden nodes from the sim-net)
def get_evo_input(input): #Extract the hidden layer representatiuon that will be the input to the EvoNet
    evo_inp = interim_model.predict(input)
    evo_inp = np.reshape(evo_inp, (len(evo_inp[0])))
    return evo_inp

#Simulate rovers exploration for a full episode
def rover_sim(net):

    nn_state, steps, tot_reward = reset_board()  # Reset board
    #print interim_model.predict(nn_state[0])[0][0]
    #mod.dispGrid(gridworld),
    for steps in range(total_steps):  # One training episode till goal is not reached
        x = []
        y = []
        for agent_id in range(num_agents):  # 1 turn per agent
            if baldwin:
                evo_input = get_evo_input(nn_state[agent_id])
                if augmented_input:

                    #if test_event:
                    #evo_input *= 0


                    evo_input = np.append(evo_input, nn_state[agent_id].flatten())

            else:
                evo_input = np.reshape(nn_state[agent_id], (nn_state[agent_id].shape[1]))

            #Get action from the Evo-net
            action = run_evo_net(net, evo_input)
            #action = randint(0,5)
            #print action

            # Get Reward and move
            #if vizualize:
                #print action,
            gridworld.move_and_get_reward(agent_id, action)

            x.append(nn_state[agent_id][:])
            x[agent_id][0][len(x[0])-4+action] = 1 #Code action taken into the x learning target

        #if vizualize: mod.dispGrid(gridworld);
        #Get new nnstates after all an episode of moves have completed
        for agent_id in range(num_agents):
            nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn)
            y.append(nn_state[agent_id])

        # LEARNING PART
        if online_learning and baldwin and update_sim:
            x = np.array(x);
            x = np.reshape(x, (len(x), len(x[0][0])))
            y = np.array(y);
            y = np.reshape(y, (len(y), len(y[0][0])))
            simulator.fit(x, y, verbose=0, nb_epoch=1)
            update_interim_model()
        elif baldwin and update_sim:  # Just append the trajectory
            sim_x.append(x)
            sim_y.append(y)

        if gridworld.check_goal_complete():
            #print 'GOAL MET'
            break
    tot_reward = 1.0 * sum(gridworld.goal_complete)/num_poi

        # END OF ONE ROUND OF SIMULATION
    #mod.dispGrid(gridworld)
    return tot_reward

#Fitness function evaluation for a genome
def evaluate(genome):
    net = NEAT.NeuralNetwork(); genome.BuildPhenotype(net); net.Flush() #Build net from genome
    fitness = rover_sim(net)
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


    if baldwin:
        g = NEAT.Genome(0, hidden_nodes + (360/angle_res) * 4 + 5, evo_hidden, 5, fs_neat, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 1, params)
    else:
        g = NEAT.Genome(0, 360 * 4 / angle_res, evo_hidden, 5, fs_neat, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 1, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    pop.RNG.Seed(0)


    for gen in range (total_generations):
        #if gen == 500:update_sim = False
        genome_list = NEAT.GetGenomeList(pop)
        sim_x = []; sim_y = []
        fitness_list, id_list, best_genome = mod.dev_EvaluateGenomeList_Serial(genome_list, evaluate, display=0)
        try:
            best_genome.Save('best')
            #from MultiNEAT import viz
            #viz.render_nn(best_genome)
        except:
            1 + 1



        #fitness_list= mod.dev_EvaluateGenomeList_Parallel(genome_list, evaluate, display=1)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        best_fit = max(fitness_list); tracker.add_fitness(best_fit, gen)
        print 'Gen:', gen, ' Baldwin' if baldwin else ' Darwinian', 'Online' if online_learning else 'Offline',   ' Max fit percent:', int(best_fit * 100), ' Avg:', int(100*tracker.avg_fitness)#, id_list[fitness_list.index(max(fitness_list))]
        if not online_learning and baldwin and update_sim:
            x= np.array(sim_x); x = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[3]) )
            y= np.array(sim_y); y = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[3]) )
            simulator.fit(x, y, verbose=0, nb_epoch=1)
            update_interim_model()

        if gen % 100 == 0:
            vizualize = True






