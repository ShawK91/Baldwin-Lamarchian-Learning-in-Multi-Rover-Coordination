import numpy as np, math, copy, time
import MultiNEAT as NEAT
import mod_rqe as mod, sys
from random import randint, choice

class Parameters:
    def __init__(self):
        self.population_size = 3
        self.predictor_hnodes = 20  # Prediction module hidden nodes (also the input to the Evo-net)
        self.augmented_input = 1
        self.baldwin = 0
        self.online_learning = 1
        self.update_sim = 1
        self.pre_train = 0
        self.D_reward = 0  # D reward scheme
        self.sim_all = 0  # Simulator learns to predict the enmtrie state including the POI's
        self.share_sim_subpop = 1  # Share simulator within a sub-population
        self.sensor_avg = True  # Average distance as state input vs (min distance by default)
        self.split_learner = 1
        self.state_representation = 2  # 1 --> Angled brackets, 2--> List of agent/POIs
        if self.split_learner: self.state_representation = 1 #ENSURE

        self.grid_row = 10
        self.grid_col = 10
        self.obs_dist = 1  # Observe distance (Radius of POI that agents have to be in for successful observation)
        self.coupling = 1  # Number of agents required to simultaneously observe a POI
        self.total_steps = 20 # Total roaming steps without goal before termination
        self.num_agents = 2
        self.num_poi = 4
        self.angle_res = 90
        self.agent_random = 0
        self.poi_random = 0
        self.total_generations = 10000
        self.wheel_action = 0

        self.use_rnn = 0  # Use recurrent instead of normal network
        self.success_replay = False
        self.vizualize = False
        self.is_halloffame = 1
        self.hof_weight = 0.5
        self.leniency = 1

        # Determine Evo-input size

        self.evo_input_size = 0
        if self.state_representation == 2:
            if self.baldwin and self.augmented_input:
                self.evo_input_size = self.predictor_hnodes + self.num_agents * 2 + self.num_poi * 2
            elif self.baldwin and not self.augmented_input:
                self.evo_input_size = self.predictor_hnodes
            else:
                self.evo_input_size = self.num_agents * 2 + self.num_poi * 2
        elif self.state_representation == 1:
            if self.baldwin and self.augmented_input:
                self.evo_input_size = self.predictor_hnodes + (360 * 4 / self.angle_res)
            elif self.baldwin and not self.augmented_input:
                self.evo_input_size = self.predictor_hnodes
            else:
                self.evo_input_size = (360 * 4 / self.angle_res)
        elif self.state_representation == 3:
            k = 1
            #TODO Complete this
        if self.split_learner and not self.baldwin: #Strict Darwin with no Baldwin for split learner
            self.evo_input_size = self.num_agents * 2 + self.num_poi * 2 + (360 * 4 / self.angle_res)
        print self.evo_input_size

        #EVoNET
        self.use_neat = 1  # Use NEAT VS. Keras based Evolution module
        self.keras_evonet_hnodes = 25  # Keras based Evo-net's hidden nodes
        self.keras_evonet_leniency = 1 #Fitness calculation based on leniency vs averaging


        #NEAT parameters
        self.params = NEAT.Parameters()
        self.params.PopulationSize = self.population_size
        self.params.fs_neat = 0
        self.params.evo_hidden = 0
        self.params.MinSpecies = 5
        self.params.MaxSpecies = 10
        self.params.EliteFraction = 0.05
        self.params.RecurrentProb = 0.2
        self.params.RecurrentLoopProb = 0.2

        self.params.MaxWeight = 8
        self.params.MutateAddNeuronProb = 0.01
        self.params.MutateAddLinkProb = 0.05
        self.params.MutateRemLinkProb = 0.01
        self.params.MutateRemSimpleNeuronProb = 0.005
        self.params.MutateNeuronActivationTypeProb = 0.005

        self.params.ActivationFunction_SignedSigmoid_Prob = 0.01
        self.params.ActivationFunction_UnsignedSigmoid_Prob = 0.5
        self.params.ActivationFunction_Tanh_Prob = 0.1
        self.params.ActivationFunction_SignedStep_Prob = 0.1

parameters = Parameters() #Create the Parameters class
tracker = mod.statistics() #Initiate tracker
gridworld = mod.Gridworld (parameters)  # Create gridworld

# # Get rover data for the simulartor
# def get_sim_ae_data(num_rounds):  # Get simulation data
#     nn_state, steps, tot_reward = reset_board()  # Reset board
#     x = []
#     for ii in range(num_rounds):
#         for steps in range(50):  # One training episode till goal is not reached
#
#             for agent_id in range(num_agents):  # 1 turn per agent
#                 action = randint(0, 5)  # Get action from the Evo-net
#                 gridworld.move_and_get_reward(agent_id, action)  # Move gridworld
#                 x.append(nn_state[agent_id][:])
#
#             # Get new nnstates after all an episode of moves have completed
#             for agent_id in range(num_agents):
#                 nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn, sensor_avg)
#             if gridworld.check_goal_complete():
#                 break
#     x = np.array(x)
#     x = np.reshape(x, (x.shape[0], x.shape[2]))
#     return x
#
# num_rounds = 10000
# if pre_train:
#     train_x = get_sim_ae_data(num_rounds)
#     valid_x = get_sim_ae_data(num_rounds/10)
# else:
#     train_x = 0; valid_x = 0

num_evals = 5
def evolve(gridworld, parameters, hof_score):
    best_team = None
    best_global = 0
    #NOTE: ALL keyword means all sub-populations
    for i in range(parameters.num_agents): gridworld.agent_list[i].evo_net.referesh_genome_list() #get new genome list and fitness evaluations trackers
    teams = np.zeros(parameters.num_agents).astype(int) #Team definitions by index
    selection_pool = [] #Selection pool listing the individuals with multiples for to match number of evaluations
    for i in range(parameters.num_agents): #Filling the selection pool
        if parameters.use_neat: ig_num_individuals = len(gridworld.agent_list[i].evo_net.genome_list) #NEAT's number of individuals can change
        else: ig_num_individuals = parameters.population_size #For keras_evo-net the number of individuals stays constant at population size
        selection_pool.append(np.arange(ig_num_individuals))
        for j in range(num_evals - 1): selection_pool[i] = np.append(selection_pool[i], np.arange(ig_num_individuals))

    for evals in range(parameters.population_size * num_evals): #For all evaluation cycles
        for i in range(len(teams)): #Pick teams
            rand_index = randint(0, len(selection_pool[i])-1) #Get a random index
            teams[i] = selection_pool[i][rand_index] #Pick team member from that index
            selection_pool[i] = np.delete(selection_pool[i], rand_index) #Delete that index
        for i in range(len(teams)): gridworld.agent_list[i].evo_net.build_net(teams[i])  # build network for the genomes within the team
        rewards, global_reward = run_simulation(parameters, gridworld, teams) #Returns rewards for each member of the team
        if global_reward > best_global:
            best_global = global_reward; #Store the best global performance
            best_team = np.copy(teams) #Store the best team

        for id, agent in enumerate(gridworld.agent_list): agent.evo_net.fitness_evals[teams[id]].append(rewards[id]) #Assign those rewards to the members of the team across the sub-populations

    if parameters.is_halloffame: #HAll of Fame
        if gridworld.agent_list[0].evo_net.hof != None: #Quit first time (special case for first run)
            for sub_pop, agent in enumerate(gridworld.agent_list):  # For each agent population
                for index in range(len(gridworld.agent_list[sub_pop].evo_net.genome_list)): #Each individual in an agent sub-population
                    rewards, global_reward = run_simulation(parameters, gridworld, teams=None, hof_run=True, hof_sub_pop_id=sub_pop, hof_ag_index=index)
                    agent.evo_net.hof_fitness_evals[index].append(rewards[sub_pop]) #Assign hof scores
                    if global_reward > best_global:
                        best_global = global_reward;  # Store the best global performance
                        if best_global > hof_score:
                            agent.evo_net.hof = agent.evo_net.net_list[index] #Update the HOF team
                            print 'HOF CHANGE', global_reward








    for agent in gridworld.agent_list:
        agent.evo_net.update_fitness()# Assign fitness to genomes
        #agent.evo_net.pop.Epoch() #Epoch update method inside NEAT
        if parameters.baldwin and parameters.update_sim and not parameters.share_sim_subpop: agent.evo_net.bald.port_best_sim()  # Deep copy best simulator

    if parameters.is_halloffame and best_global > hof_score: #Hall of fame-new team found
        for i in range(len(best_team)):
            gridworld.agent_list[i].evo_net.hof = gridworld.agent_list[i].evo_net.net_list[best_team[i]]
            print 'HOF changed', hof_score, gridworld.agent_list[i].evo_net.hof



    return best_global


####################################### END OF MACROS ####################################

#Reset gridworld



def run_simulation(parameters, gridworld, teams, hof_run = False, hof_sub_pop_id = None, hof_ag_index = None): #Run simulation given a team and return fitness for each individuals in that team
    gridworld.reset(teams, hof_run, hof_sub_pop_id, hof_ag_index)  # Reset board

    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        for id, agent in enumerate(gridworld.agent_list):  #get all the action choices from the agents
            if steps == 0: agent.perceived_state = gridworld.get_state(agent) #Update all agent's perceived state
            if steps == 0 and parameters.split_learner: agent.split_learner_state = gridworld.get_state(agent, 2) #If split learner
            if hof_run and id != hof_sub_pop_id: #Hall of Fame run minus the one population being tested
                agent.take_action(None, hof_run)
            elif hof_run: agent.take_action(hof_ag_index, False) #For hof_run for the evaluated individual
            else: agent.take_action(teams[id], hof_run) #Make the agent take action using the Evo-net with given id from the population

        gridworld.move() #Move gridworld
        gridworld.update_poi_observations() #Figure out the POI observations and store all credit information

        learn_activate = True
        for id, agent in enumerate(gridworld.agent_list):
            if not hof_run: agent.referesh(teams[id], gridworld) #Update state and learn if applicable
            else:
                if id == hof_sub_pop_id: agent.referesh(None, gridworld)  # Update state and learn if applicable
                else: agent.referesh(None, gridworld, learn_activate=False)




        if gridworld.check_goal_complete(): break #If all POI's observed

    rewards, global_reward = gridworld.get_reward()
    return rewards, global_reward

def random_baseline():
    total_trials = 10
    g_reward = 0.0
    for trials in range(total_trials):
        best_reward = 0
        for iii in range(population_size*5):
            nn_state, steps, tot_reward = reset_board()  # Reset board
            for steps in range(total_steps):  # One training episode till goal is not reached
                all_actions = []  # All action choices from the agents
                for agent_id in range(num_agents):  # get all the action choices from the agents
                    action = randint(0,4); all_actions.append(action)  # Store all agent's actions
                gridworld.move(all_actions)  # Move gridworld
                gridworld.update_poi_observations()  # Figure out the POI observations and store all credit information

                # Get new nnstates after all an episode of moves have completed
                for agent_id in range(num_agents):
                    nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn, sensor_avg, state_representation)
                if gridworld.check_goal_complete(): break

            reward = 100 * sum(gridworld.goal_complete) / num_poi  # Global Reward
            if reward > best_reward: best_reward = reward
            #End of one full population worth of trial
        g_reward += best_reward

    print 'Random Baseline: ', g_reward/total_trials



if __name__ == "__main__":
    #random_baseline()
    mod.dispGrid(gridworld)
    hof_score = 0




    for gen in range (parameters.total_generations): #Main Loop


        curtime = time.time()

        best_global = evolve(gridworld, parameters, hof_score) #CCEA
        if best_global > hof_score: hof_score = best_global
        tracker.add_fitness(best_global, gen) #Add best global performance to tracker
        _, hof_reward = tracker.run_hof_simulation(parameters, gridworld)
        print hof_reward


        if parameters.use_neat: tracker.add_mpc(gridworld, parameters) #Update mpc statistics
        elapsed = time.time() - curtime
        continue
        if parameters.use_neat:
            print 'Gen:', gen, ' Baldwin' if parameters.baldwin else ' Darwinian', 'Online' if parameters.online_learning else 'Offline', ' Best g_reward', int(best_global * 100), ' Avg:', int(100 * tracker.avg_fitness), ' Delta MPC:', int(tracker.avg_mpc), '+-', int(tracker.mpc_std), 'Elapsed Time: ', elapsed, 'Best HOF: ', hof_score #' Delta generations Survival: '      #for i in range(num_agents): print all_pop[i].delta_age / params.PopulationSize,
            #print
        else:
            print 'Gen:', gen, ' Baldwin' if parameters.baldwin else ' Darwinian', 'Online' if parameters.online_learning else 'Offline', ' Best g_reward', int(best_global * 100), ' Avg:', int(100 * tracker.avg_fitness), ' Delta generations Survival: '
            #for i in range(num_agents): print all_pop[i].pop.longest_survivor,
            #print
        continue

        #TODO Offline learning
        if not online_learning and baldwin and update_sim: bald.offline_train()










