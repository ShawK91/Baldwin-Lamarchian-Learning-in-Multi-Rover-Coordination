import numpy as np
from random import randint
import random, sys
import numpy as np
import mod_rqe as mod
from keras.models import model_from_json
from pympler import summary
import MultiNEAT as NEAT
from MultiNEAT import EvaluateGenomeList_Serial


#MACROS
#NEAT parameters
if True:
    params = NEAT.Parameters()
    params.PopulationSize = 50
    params.DynamicCompatibility = True
    params.WeightDiffCoeff = 4.0
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = True
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
if True: #Macros
    grid_row = 10
    grid_col = 10
    obs_dist = 4 #Observe distance (Radius of POI that agents have to be in for successful observation)
    coupling = 2 #Number of agents required to simultaneously observe a POI
    observability = 1
    hidden_nodes = 1
    epsilon = 0.5  # Exploration Policy
    alpha = 0.25  # Learning rate
    gamma = 0.7 # Discount rate
    total_steps = 50 #Total roaming steps without goal before termination
    num_agents = 3
    num_poi = 2
    total_train_epoch = 150000
    angle_res = 90
    online_learning = False
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

    if not use_rnn:
        hidden_nodes *= 5  # Normalize RNN and NN flexibility

def test_random(gridworld,  illustrate = False):
    rand_suc = 0
    for i in range(1000):
        nn_state, steps, tot_reward = reset_board()
        hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                action = randint(0,4)
                hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if 0: #illustrate:
                    mod.dispGrid(gridworld)
                    #raw_input('Press Enter')
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        #print hist
        if steps < gridworld.optimal_steps * 2:
            rand_suc += 1
    return rand_suc/1000

def test_dqn(q_model, gridworld, illustrate = False, total_samples = 10):
    cumul_rew = 0; cumul_coverage = 0
    for sample in range(total_samples):
        nn_state, steps, tot_reward = reset_board()
        #hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                q_vals = get_qvalues(nn_state[agent_id], q_model)  # for first step, calculate q_vals here
                action = np.argmax(q_vals)
                if np.amax(q_vals) - np.amin(q_vals) == 0:  # Random if all choices are same
                    action = randint(0, len(q_model) - 1)
                #hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if illustrate:
                    mod.dispGrid(gridworld)
                    print agent_id, action, reward
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        cumul_rew += tot_reward/(steps+1); cumul_coverage += sum(gridworld.goal_complete) * 1.0/gridworld.num_poi
    return cumul_rew/total_samples, cumul_coverage/total_samples

def display_q_values(q_model):
    gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board
    for x in range(grid_row):
        for i in range(x):
            _ = mod.move_and_get_reward(gridworld, agent, action=2)
        for y in range(grid_col):
            #mod.dispGrid(gridworld, agent)
            print(get_qvalues(nn_state, q_model))
            _ = mod.move_and_get_reward(gridworld, agent, action=1)
            nn_state = mod.referesh_hist(gridworld, agent, nn_state, use_rnn)
        gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board

def test_nntrain(net, x, y):
    error = []
    for i in range(len(x)):
        input = np.reshape(x[i], (1, len(x[i])))
        error.append((net.predict(input) - y[i])[0][0])
    return error

def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range (num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn))
    return first_input, 0, 0

def get_qvalues(nn_state, q_model):
    values = np.zeros(len(q_model))
    for i in range(len(q_model)):
        values[i] = q_model[i].predict(nn_state)
    return values

def decay(epsilon, alpha):
    if epsilon > 0.1:
        epsilon -= 0.00005
    if alpha > 0.1:
        alpha -= 0.00005
    return epsilon, alpha

def reset_trajectories():
    trajectory_states = []
    trajectory_action = []
    trajectory_reward = []
    trajectory_max_q = []
    trajectory_qval = []
    trajectory_board_pos = []
    return trajectory_states, trajectory_action, trajectory_reward, trajectory_max_q, trajectory_qval, trajectory_board_pos

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
        self.tr_reward.append(reward)
        self.tr_coverage.append(coverage)
        # np.savetxt('reward.csv', np.array(self.tr_reward), fmt='%.3f')
        # np.savetxt('coverage.csv', np.array(self.tr_coverage), fmt='%.3f')
        self.reward_matrix = np.roll(self.reward_matrix, -1)
        self.reward_matrix[-1] = reward
        self.coverage_matrix = np.roll(self.coverage_matrix, -1)
        self.coverage_matrix[-1] = coverage
        if self.reward_matrix[0] != 10000:
            self.tr_reward_mat.append(np.array([train_epoch, np.average(self.reward_matrix)]))
            np.savetxt('reward_matrix.csv', np.array(self.tr_reward_mat), fmt='%.3f')
            self.tr_coverage_mat.append(np.array([train_epoch, np.average(self.coverage_matrix)]))
            np.savetxt('coverage_matrix.csv', np.array(self.tr_coverage_mat), fmt='%.3f')


def import_models(in_filename, save_filename, model_arch):
    model_arch.load_weights(in_filename)
    model_arch.save_weights(save_filename, overwrite=True)
    model_arch.compile(loss='mse', optimizer='adam')




def putracking():
    from pympler import tracker
    tr = tracker.SummaryTracker()
    #tr.print_diff()

    #tracker = statistics()
    gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr = angled_repr, angle_res = angle_res) #Create gridworld
    #mod.dispGrid(gridworld)
    #tr = tracker.SummaryTracker()
    tr.print_diff()








if __name__ == "__main__":

    gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand,
                              angled_repr=angled_repr, angle_res=angle_res, obs_dist=obs_dist,
                              coupling=coupling)  # Create gridworld
    g = NEAT.Genome(0, 360 * 4 / angle_res, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    h = NEAT.Genome(1, 360 * 4 / angle_res, 0, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)



















