
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import SReLU
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
import math
import MultiNEAT as NEAT
import numpy as np, time
import random
import pickle
import neat as py_neat
from neat import nn


class PyNeat_Config_object(object):
    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']

    def __init__(self, parameters):
        from neat.reproduction import DefaultReproduction
        from neat.stagnation import DefaultStagnation
        from neat.genes import NodeGene, ConnectionGene
        from neat.genome import Genome, FFGenome
        from neat import activation_functions

        self.registry = {'DefaultStagnation': DefaultStagnation,
                         'DefaultReproduction': DefaultReproduction}
        self.type_config = {}

        # Phenotype configuration
        self.input_nodes = parameters.evo_input_size
        self.output_nodes = 5
        self.hidden_nodes = parameters.py_neat_config.hidden_nodes
        self.initial_connection = parameters.py_neat_config.initial_connection
        self.connection_fraction = None
        self.max_weight = parameters.py_neat_config.max_weight
        self.min_weight = parameters.py_neat_config.min_weight
        self.feedforward = parameters.py_neat_config.feedforward
        self.weight_stdev = parameters.py_neat_config.weight_stdev
        self.activation_functions = parameters.py_neat_config.activation_functions.strip().split()

        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify that specified activation functions are valid.
        for fn in self.activation_functions:
            if not activation_functions.is_valid(fn):
                raise Exception("Invalid activation function name: {0!r}".format(fn))

        # Select a genotype class.
        if self.feedforward:
            self.genotype = FFGenome
        else:
            self.genotype = Genome

        # Genetic algorithm configuration
        self.pop_size = parameters.population_size
        self.max_fitness_threshold = parameters.py_neat_config.max_fitness_threshold
        self.prob_add_conn = parameters.py_neat_config.prob_add_conn
        self.prob_add_node = parameters.py_neat_config.prob_add_node
        self.prob_delete_conn = parameters.py_neat_config.prob_delete_conn
        self.prob_delete_node = parameters.py_neat_config.prob_delete_node
        self.prob_mutate_bias = parameters.py_neat_config.prob_mutate_bias
        self.bias_mutation_power = parameters.py_neat_config.bias_mutation_power
        self.prob_mutate_response = parameters.py_neat_config.prob_mutate_response
        self.response_mutation_power = parameters.py_neat_config.response_mutation_power
        self.prob_mutate_weight = parameters.py_neat_config.prob_mutate_weight
        self.prob_replace_weight = parameters.py_neat_config.prob_replace_weight
        self.weight_mutation_power = parameters.py_neat_config.weight_mutation_power
        self.prob_mutate_activation = parameters.py_neat_config.prob_mutate_activation
        self.prob_toggle_link = parameters.py_neat_config.prob_toggle_link
        self.reset_on_extinction = bool(parameters.py_neat_config.reset_on_extinction)

        # genotype compatibility
        self.compatibility_threshold = parameters.py_neat_config.compatibility_threshold
        self.excess_coefficient = parameters.py_neat_config.excess_coefficient
        self.disjoint_coefficient = parameters.py_neat_config.disjoint_coefficient
        self.weight_coefficient = parameters.py_neat_config.weight_coefficient

        # Gene types
        self.node_gene_type = NodeGene
        self.conn_gene_type = ConnectionGene

        # Default stagnation
        self.species_fitness_func = parameters.py_neat_config.species_fitness_func
        self.max_stagnation = parameters.py_neat_config.max_stagnation

        # Default Reporoduction
        self.elitism = parameters.py_neat_config.elitism
        self.survival_threshold = parameters.py_neat_config.survival_threshold

        stagnation_type_name = parameters.py_neat_config.stagnation_type
        reproduction_type_name = parameters.py_neat_config.reproduction_type

        if stagnation_type_name not in self.registry:
            raise Exception('Unknown stagnation type: {!r}'.format(stagnation_type_name))
        self.stagnation_type = self.registry[stagnation_type_name]

        self.type_config[stagnation_type_name] = [
            ('species_fitness_func', parameters.py_neat_config.species_fitness_func),
            ('max_stagnation ', parameters.py_neat_config.max_stagnation)]

        if reproduction_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.reproduction_type = self.registry[reproduction_type_name]
        self.type_config[reproduction_type_name] = [('elitism', parameters.py_neat_config.elitism), (
        'survival_threshold', parameters.py_neat_config.survival_threshold)]

        # Gather statistics for each generation.
        self.collect_statistics = True
        # Show stats after each generation.
        self.report = True
        # Save the best genome from each generation.
        self.save_best = True
        # Time in minutes between saving checkpoints, None for no timed checkpoints.
        self.checkpoint_time_interval = 30
        # Time in generations between saving checkpoints, None for no generational checkpoints.
        self.checkpoint_gen_interval = 100

    def register(self, typeName, typeDef):
        """
        User-defined classes mentioned in the config file must be provided to the
        configuration object before the load() method is called.
        """
        self.registry[typeName] = typeDef

    def get_type_config(self, typeInstance):
        return dict(self.type_config[typeInstance.__class__.__name__])

class PyNeat_handler():
    def __init__(self, parameters):
        self.pyNeat_config_object = PyNeat_Config_object(parameters)


    def get_genomes(self, pop):
        genomes = []
        for s in pop.species.species:
            genomes.extend(s.members)
        return genomes




    def epoch(self, pop, genomes):

        """
        The user-provided fitness_function should take one argument, a list of all genomes in the population,
        and its return value is ignored.  This function is free to maintain external state, perform evaluations
        in parallel, and probably any other thing you want.  The only requirement is that each individual's
        fitness member must be set to a floating point value after this function returns.
        It is assumed that fitness_function does not modify the list of genomes, or the genomes themselves, apart
        from updating the fitness member.
        """
        pop.generation += 1
        pop.reporters.start_generation(pop.generation)
        pop.total_evaluations += len(genomes)

        # Gather and report statistics.
        best = max(genomes)
        pop.reporters.post_evaluate(genomes, pop.species.species, best)

        # Save the best genome from the current generation if requested.
        if pop.config.save_best:
            with open('best_genome', 'wb') as f:
                pickle.dump(best, f)

        # Save if the fitness threshold is reached.
        if best.fitness >= pop.config.max_fitness_threshold:
            pop.reporters.found_solution(pop.generation, best)
            with open('solution_genome', 'wb') as f:
                pickle.dump(best, f)

        # Create the next generation from the current generation.
        new_population = pop.reproduction.reproduce(pop.species, pop.config.pop_size)

        # Check for complete extinction
        if not pop.species.species:
            pop.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if pop.config.reset_on_extinction:
                new_population = pop.reproduction.create_new(pop.config.pop_size)
            else:
                print 'Extinction'

        # Update species age.
        for s in pop.species.species:
            s.age += 1

        # Divide the new population into species.
        pop.species.speciate(new_population)

        if pop.config.checkpoint_gen_interval is not None and pop.generation % pop.config.checkpoint_gen_interval == 0:
            pop.save_checkpoint(checkpoint_type="generation")

        pop.reporters.end_generation()


class Baldwin_util:
    def __init__(self, parameters):
        self.parameters = parameters

        #Figure out the prediction modules's input size
        if parameters.state_representation == 2 or parameters.split_learner:
            if parameters.sim_all: predictor_input = predictor_input = parameters.num_agents * 2 + parameters.num_poi * 2 + 5
            else: predictor_input = predictor_input = parameters.num_agents * 2 + 5

        elif parameters.state_representation == 1 and not parameters.split_learner:
            if parameters.sim_all: predictor_input = (360 / parameters.angle_res) * 4 + 5
            else: predictor_input = (360 / parameters.angle_res) * 2 + 5

        #initiate the prediction module or the population of them
        if parameters.share_sim_subpop: #One prediction module
            self.simulator = init_nn(predictor_input, parameters.predictor_hnodes)
            self.interim_model = init_nn(predictor_input, parameters.predictor_hnodes, middle_layer=True, weights=self.simulator.layers[0].get_weights())
        else: #Population of prediction modules
            self.simulator = []
            for i in range(parameters.population_size + 5): self.simulator.append(init_nn(predictor_input, parameters.predictor_hnodes)) # Create simulator for each agent
            self.interim_model = init_nn(predictor_input, parameters.predictor_hnodes, middle_layer=True, weights=self.simulator[0].layers[0].get_weights())
        self.traj_x = []; self.traj_y = []  # trajectory for batch learning
        self.best_sim_index = 0

    #Updates the weight to the interim model
    def update_interim_model(self, index):
        #if index < len(self.simulator):
        if self.parameters.share_sim_subpop: weights = self.simulator.layers[0].get_weights()
        else: weights = self.simulator[index].layers[0].get_weights()
        self.interim_model.layers[0].set_weights(weights)

    # Get the inputs to the Evo-net (extract hidden nodes from the sim-net)
    def get_evo_input(self, input):  # Extract the hidden layer representatiuon that will be the input to the EvoNet
        input = self.sim_input_transform(input) #Agent only prediction scheme
        evo_inp = self.interim_model.predict(input)
        evo_inp = np.reshape(evo_inp, (len(evo_inp[0])))
        return evo_inp

    def sim_input_transform(self, input):  # Change state input to agent only input
        if self.parameters.sim_all: return input
        transformed_inp = []
        if self.parameters.state_representation == 1 and not self.parameters.split_learner:
            for i in range(len(input[0])):
                if i % 4 >= 2 or i >= len(input[0]) - 5:
                    transformed_inp.append(input[0][i])
        elif self.parameters.state_representation == 2 or self.parameters.split_learner:
            for i in range(self.parameters.num_poi * 2, len(input[0])):
                transformed_inp.append(input[0][i])

        transformed_inp = np.array(transformed_inp)
        transformed_inp = np.reshape(transformed_inp, (1, len(transformed_inp)))
        return transformed_inp

    #Train simulator offline at the end
    def offline_train(self, index):
        if len(self.traj_x) == 0: return
        x = np.array(self.traj_x);
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        y = np.array(self.traj_y);
        y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))
        if self.parameters.share_sim_subpop: self.simulator.fit(x, y, verbose=0, nb_epoch=1)
        else: self.simulator[index].fit(x, y, verbose=0, nb_epoch=1)
        #self.update_interim_model()
        self.traj_x = []; self.traj_y = []

    #Copy the best eprforming candidate's simulator to next generation
    def port_best_sim(self):
        w = self.simulator[self.best_sim_index].get_weights()
        for i in range(len(self.simulator)):
            self.simulator[i].set_weights(w)

    # LEARNING PART
    def learning(self, last_state, new_state, index):
        if self.parameters.online_learning and self.parameters.baldwin and self.parameters.update_sim:
            x = self.sim_input_transform(last_state)
            y = self.sim_input_transform(new_state)
            if self.parameters.share_sim_subpop: self.simulator.fit(x, y, verbose=0, nb_epoch=1)
            else: self.simulator[index].fit(x, y, verbose=0, nb_epoch=1)
            self.update_interim_model(index)
        elif self.parameters.baldwin and self.parameters.update_sim:  # Just append the trajectory
            x = self.sim_input_transform(last_state)
            y = self.sim_input_transform(new_state)
            self.traj_x.append(x)
            self.traj_y.append(y)

class Evo_net():
    def __init__(self, parameters):
        self.parameters = parameters
        if parameters.baldwin: self.bald = Baldwin_util(parameters)
        if parameters.use_neat:
            if parameters.use_py_neat: #Python implementation of NEAT
                from neat import population, nn#, statistics, visualize, config
                self.pyneat_handler = PyNeat_handler(self.parameters) #Make the pyNeat_handler object
                pyneat_config = self.pyneat_handler.pyNeat_config_object #Import the configurations
                self.pop = population.Population(pyneat_config, use_config_override=True)
                self.genome_list = self.pyneat_handler.get_genomes(self.pop)
                self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations through an iteration
                self.net_list = [[] for x in xrange(len(self.genome_list))] #Stores the networks for the genomes

            else:
                seed = 0 if (parameters.params.evo_hidden == 0) else 1  # Controls sees based on genome initialization
                g = NEAT.Genome(0, parameters.evo_input_size, parameters.params.evo_hidden, 5, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                                NEAT.ActivationFunction.UNSIGNED_SIGMOID, seed, parameters.params)  # Constructs genome
                g.Save('initial')
                self.pop = NEAT.Population(g, parameters.params, True, 1.0, 0)  # Constructs population of genome
                self.pop.RNG.Seed(0)
                self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
                self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations through an iteration
                self.net_list = [[] for x in xrange(len(self.genome_list))] #Stores the networks for the genomes
                self.base_mpc = self.pop.GetBaseMPC()
                self.current_mpc = self.pop.GetCurrentMPC()
                self.delta_mpc = self.current_mpc - self.base_mpc
                self.oldest_genome_id = 0
                self.youngest_genome_id = 0
                self.delta_age = self.oldest_genome_id - self.youngest_genome_id
        else:
            self.pop = Population(parameters.evo_input_size, parameters.keras_evonet_hnodes, 5, parameters.population_size)
            self.fitness_evals = [[] for x in xrange(parameters.population_size)] #Controls fitnesses calculations through an iteration
            self.net_list = [[] for x in xrange(parameters.population_size)] #Stores the networks for the genomes


    def epoch(self): #Method to complete epoch after fitness has been assigned to the genomes
        if self.parameters.use_py_neat: #Python based NEAT use Epoch method written outside
            self.pyneat_handler.epoch(self.pop, self.genome_list)
        else: #For C++ NEAT and Keras based Evonet, use inbuilt method
            self.pop.Epoch()  # Epoch update method inside NEAT and Keras




    def referesh_genome_list(self):
        if self.parameters.use_neat:
            if self.parameters.use_py_neat: #Python implementation of NEAT
                self.genome_list = self.pyneat_handler.get_genomes(self.pop)
            else:
                self.genome_list = NEAT.GetGenomeList(self.pop) #List of genomes in this subpopulation
            self.fitness_evals = [[] for x in xrange(len(self.genome_list))] #Controls fitnesses calculations throug an iteration
            self.net_list = [[] for x in xrange(len(self.genome_list))]  # Stores the networks for the genomes
        else: #Keras Evo-net
            self.fitness_evals = [[] for x in xrange(self.parameters.population_size)]  # Controls fitnesses calculations throug an iteration
            self.net_list = [[] for x in xrange(self.parameters.population_size)]  # Stores the networks for the genomes



    def build_net(self, index):
        if not self.net_list[index]: #if not already built
            if self.parameters.use_neat:
                if self.parameters.use_py_neat: #Python NEAT
                    self.net_list[index] = nn.create_feed_forward_phenotype(self.genome_list[index])
                else: #C++ NEAT
                    self.net_list[index] = NEAT.NeuralNetwork();
                    self.genome_list[index].BuildPhenotype(self.net_list[index]);
                    self.net_list[index].Flush()  # Build net from genome
                #self.genome_list[index].Save('test')


            else:
                self.net_list[index] = self.pop.net_pop[int(self.pop.pop_handle[index][0])]
        #self.net_list[index].Save('a')



    # Get action choice from Evo-net
    def run_evo_net(self, index, state):
        scores = [] #Probability output for five action choices
        if self.parameters.use_neat:
            if self.parameters.use_py_neat:  # Python NEAT
                scores = self.net_list[index].serial_activate(state) #TODO Test
            else:
                self.net_list[index].Flush()
                self.net_list[index].Input(state)  # can input numpy arrays, too for some reason only np.float64 is supported
                self.net_list[index].Activate()
                for i in range(5):
                    if not math.isnan(1 * self.net_list[index].Output()[i]):
                        scores.append(1 * self.net_list[index].Output()[i])
                    else:
                        scores.append(0)
        else: #Use keras Evo-net
            state = np.reshape(state, (1, len(state)))
            scores = self.net_list[index].predict(state)[0]
        if self.parameters.wheel_action and sum(scores) != 0: action = roulette_wheel(scores)
        elif sum(scores) != 0: action = np.argmax(scores)
        else: action = randint(0,4)
        #if action == None: action = randint(0, 4)
        #print action
        return action

    def update_fitness(self): #Update the fitnesses of the genome and also encode the best one for the generation
        if self.parameters.use_neat:
            if self.parameters.use_py_neat: #Python NEAT
                best = 0; best_sim_index = 0
                for i, g in enumerate(self.genome_list):
                    if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                        if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i])  # Use lenient learner
                        else: avg_fitness = sum(self.fitness_evals[i]) / len(self.fitness_evals[i])
                        if avg_fitness > best:
                            best = avg_fitness;
                            best_sim_index = i
                        g.fitness = avg_fitness #Update fitness

            else: #C++ NEAT
                youngest = 0; oldest = 10000000 #Magic intitalization numbers to find the oldest and youngest survuving genome
                best = 0; best_sim_index = 0
                for i, g in enumerate(self.genome_list):
                    if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                        if self.parameters.leniency: avg_fitness = max(self.fitness_evals[i])  # Use lenient learner
                        else: avg_fitness = sum(self.fitness_evals[i]) / len(self.fitness_evals[i])
                        if avg_fitness > best:
                            best = avg_fitness;
                            best_sim_index = i
                        g.SetFitness(avg_fitness) #Update fitness
                        g.SetEvaluated() #Set as evaluated
                        if g.GetID() > youngest: youngest = g.GetID();
                        if g.GetID() < oldest: oldest = g.GetID();
                self.oldest_genome_id = oldest
                self.youngest_genome_id = youngest
                self.delta_age = self.youngest_genome_id - self.oldest_genome_id
                self.current_mpc = self.pop.GetCurrentMPC(); self.delta_mpc = self.current_mpc - self.base_mpc  # Update MPC's as well

        else: #Using keras Evo-net
            best = 0; best_sim_index = 0
            for i in range(self.parameters.population_size):
                if len(self.fitness_evals[i]) != 0:  # if fitness evals is not empty (wasnt evaluated)
                    if self.parameters.keras_evonet_leniency: avg_fitness = max(self.fitness_evals[i]) #Use lenient learner
                    else: avg_fitness = sum(self.fitness_evals[i])/len(self.fitness_evals[i])
                    if avg_fitness > best:
                        best = avg_fitness; best_sim_index = i
                    self.pop.pop_handle[i][1] = 1-avg_fitness #Update fitness
        #print best

        if self.parameters.baldwin: self.bald.best_sim_index = best_sim_index #Assign the new top simulator #TODO Generalize this to best performing index and ignore if not evaluated


class Agent:
    def __init__(self, grid, parameters):
        self.parameters = parameters
        self.position = self.init_agent(grid)
        self.action = 0
        self.evo_net = Evo_net(parameters)
        self.perceived_state = None #State of the gridworld as perceived by the agent
        self.split_learner_state = None #Useful for split learner


    def init_agent(self, grid):
        start = grid.observe;  end = grid.state.shape[0] - grid.observe - 1
        rad = int(grid.dim_row / math.sqrt(3) / 3)
        center = int((start + end) / 2)
        if grid.agent_rand:
            while True:
                x = randint(center - rad, center + rad)
                y = randint(center - rad, center + rad)
                if grid.state[x][y] != 1: #position not already occupied
                    break
        else:  # Not random
            trial = 0
            while True:
                while True:
                    x = center - rad + (trial % (rad*2))
                    if x <= center + rad: break #If within limits
                while True:
                    y = center - rad + (trial / (rad*2))
                    if y <= center + rad: break #If within limits

                if grid.state[x][y] != 1: #position not already occupied
                    break
                trial+=1

        grid.state[x][y] = 1  # Agent Code
        return [x, y]

    def reset(self, grid):
        self.position = self.init_agent(grid)

    def take_action(self, net_id):
        #Modify state input to required input format
        if self.parameters.baldwin:
            if self.parameters.split_learner: padded_state = self.pad_state(self.split_learner_state)
            else: padded_state = self.pad_state(self.perceived_state)
            evo_input = self.evo_net.bald.get_evo_input(padded_state)  # Hidden nodes from simulator
            if self.parameters.augmented_input:
                #if self.parameters.split_learner: evo_input = np.append(evo_input, self.split_learner_state.flatten())  # Augment input with state info
                evo_input = np.append(evo_input, self.perceived_state.flatten())  # Augment input with state info
        else: #Darwin
            if self.parameters.split_learner:
                evo_input = np.append(self.perceived_state, self.split_learner_state)
                #evo_input = np.reshape(self.perceived_state, (self.perceived_state.shape[1]))
            else:
                evo_input = np.reshape(self.perceived_state, (self.perceived_state.shape[1]))  # State input only (Strictly Darwinian approach)
        self.action = self.evo_net.run_evo_net(net_id, evo_input) #Take action

    def referesh(self, net_id, grid):
        if not self.parameters.baldwin: #In case of Darwin
            self.perceived_state = grid.get_state(self)  # Update all agent's perceived state
            return
        else: #Baldwin
            if self.parameters.split_learner: #If split learning
                x = self.split_learner_state;
                self.perceived_state = grid.get_state(self)  # Update all agent's perceived state
                self.split_learner_state = grid.get_state(self, 2)  # Update all agent's perceived state
                y = self.split_learner_state;
            else: #If no split learning
                x = self.perceived_state;
                self.perceived_state = grid.get_state(self)  # Update all agent's perceived state
                y = self.perceived_state;

            if self.parameters.update_sim: #Learning part
                x = self.pad_state(x);
                y = self.pad_state(y)  # Pad state
                x[0][len(x[0]) - 5 + self.action] = 1  # Encode action taken
                self.evo_net.bald.learning(x, y, net_id)


    def ready_for_simulation(self, net_id):
        if self.parameters.online_learning and self.parameters.baldwin:  # Update interim model belonging to the teams[i] indexed individual in the ith sub-population
            self.evo_net.bald.update_interim_model(net_id)

    def pad_state(self, state):
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.append(state, 0)  # Add action to the state
        state = np.reshape(state, (1, len(state)))
        return state

class POI:
    def __init__(self, grid):
        self.position = self.init_poi(grid) #Initialize POI object
        self.is_observed = False #Check if goal is complete
        self.observation_history = [] #Track the identity of agents within the coupling requirements at all applicable time steps

    def init_poi(self, grid):
        start = grid.observe; end = grid.state.shape[0] - grid.observe - 1
        rad = int(grid.dim_row / math.sqrt(3) / 2)
        center = int((start + end) / 2)
        if grid.poi_rand:
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
                if grid.state[x][y] != 2: #Position not already occupied
                    break

        else: #Pre-defined starting positions
            trial = 0
            while True: #unoccuped
                k = len(grid.poi_list)
                region = trial % 4 #4 distinct regions of distribution
                access = trial / 4 #Access number
                if region == 0:
                    x = start + (access * 2) % (center - rad - 1 - start)
                    y = start + (access * 2) % (end - start)
                elif region == 1:
                    x = start + (access * 2) % (end - center - rad - 1)
                    y = end - (access * 2) % (end - start)
                elif region == 2:
                    x = end - (access * 2) % (2 * rad)
                    y = start + (access * 2) % (center - rad - 1 - start)
                else:
                    x = end - (access * 2) % (2 * rad)
                    y = end - (access * 2) % (end - center - rad - 1)
                if grid.state[x][y] != 2: #Position not already occupied
                    break
                trial += 1





        grid.state[x][y] = 2
        #grid.poi_pos.append([x, y])
        return [x,y]



    def reset(self, grid):
        self.position = self.init_poi(grid)
        self.is_observed = False
        self.observation_history = []

class Gridworld:
    def __init__(self, parameters):
        self.parameters = parameters
        self.observe = 1; self.dim_row = parameters.grid_row; self.dim_col = parameters.grid_col; self.poi_rand = parameters.poi_random; self.agent_rand = parameters.agent_random
        self.num_agents = parameters.num_agents; self.num_poi = parameters.num_poi; self.angle_res = parameters.angle_res #Angle resolution
        self.coupling = parameters.coupling #coupling requirement
        self.obs_dist = parameters.obs_dist #Observation radius requirements

        #Resettable stuff
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall() #initialize wall

        self.poi_list = [] #List of POI objects
        for i in range(self.num_poi):
            self.poi_list.append(POI(self))

        self.agent_list = []
        for i in range(self.num_agents): self.agent_list.append(Agent(self, parameters))


    def init_wall(self):
        for i in range(self.observe):
            for x in range(self.state.shape[0]):
                self.state[x][i] = 3
                self.state[x][self.state.shape[1] - 1-i] = 3
            for y in range(self.state.shape[1]):
                self.state[i][y] = 3
                self.state[self.state.shape[0] - 1-i][y] = 3

    def reset(self, teams):
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        for agent_id, agent in enumerate(self.agent_list):
            agent.reset(self)
            agent.ready_for_simulation(teams[agent_id])  # Get all agents ready by updating interim model if necesary
            if not self.parameters.online_learning: #Offline learning
                agent.evo_net.bald.offline_train(teams[agent_id])

        for poi in self.poi_list: poi.reset(self)

    def move(self):
        for agent in self.agent_list: #Move and agent
            action = agent.action
            next_pos = np.copy(agent.position)
            if action == 1: next_pos[1] += 1  # Right
            elif action == 2: next_pos[0] += 1  # Down
            elif action == 3: next_pos[1] -= 1  # Left
            elif action == 4: next_pos[0] -= 1  # Up

            # Computer reward and check illegal moves
            x = next_pos[0]; y = next_pos[1]
            if self.state[x][y] == 3: next_pos = np.copy(agent.position) #Reset if hit wall
            if self.state[x][y] == 1 or self.state[x][y] == 2 and action != 0: next_pos = np.copy(agent.position) #Reset if other Agent or POI and action != 0

            # Update gridworld and agent position
            self.state[agent.position[0]][agent.position[1]] = 0 #Encode newly freed position in the state template
            self.state[next_pos[0]][next_pos[1]] = 1 #Encode newly occupied position in the state template
            agent.position[0] = next_pos[0]; agent.position[1] = next_pos[1] #Update new positions for the agent object

    def get_state(self, agent, state_representation = None):  # Returns a flattened array around the agent position
        if state_representation == None: state_representation = self.parameters.state_representation #If no override use choice in parameters
        if state_representation == 1:  # Angle brackets
            state = np.zeros(((360 / self.angle_res), 4))
            if self.parameters.sensor_avg:  # Average distance
                dist_poi_list = [[] for x in xrange(360 / self.angle_res)]
                dist_agent_list = [[] for x in xrange(360 / self.angle_res)]

            for poi in self.poi_list:
                if not poi.is_observed:  # For all POI's that are still active
                    x1 = poi.position[0] - agent.position[0];
                    x2 = -1
                    y1 = poi.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    bracket = int(angle / self.angle_res);
                    state[bracket][0] += 1.0 / self.num_poi  # Add POIs
                    if self.parameters.sensor_avg:
                        dist_poi_list[bracket].append(dist / (2.0 * self.dim_col))
                    else:  # Min distance
                        if state[bracket][1] > dist / (2.0 * self.dim_col) or state[bracket][
                            1] == 0:  # Update min distance from POI
                            state[bracket][1] = dist / (2.0 * self.dim_col)

            for other_agent in self.agent_list:
                if other_agent != agent:  # FOR ALL AGENTS MINUS MYSELF
                    x1 = other_agent.position[0] - agent.position[0];
                    x2 = -1
                    y1 = other_agent.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    bracket = int(angle / self.angle_res)
                    state[bracket][2] += 1.0 / (self.num_agents - 1)  # Add agent
                    if self.parameters.sensor_avg:
                        dist_agent_list[bracket].append(dist / (2.0 * self.dim_col))
                    else:  # Min distance
                        if state[bracket][3] > dist / (2.0 * self.dim_col) or state[bracket][
                            3] == 0:  # Update min distance from other agent
                            state[bracket][3] = dist / (2.0 * self.dim_col)

            if self.parameters.sensor_avg:
                for bracket in range(len(dist_agent_list)):
                    try: state[bracket][1] = sum(dist_poi_list[bracket]) / len(dist_poi_list[bracket])  # Encode average POI distance
                    except: None
                    try: state[bracket][3] = sum(dist_agent_list[bracket]) / len(dist_agent_list[bracket])  # Encode average agent distance
                    except: None
            state = np.reshape(state, (1, 360 / self.angle_res * 4))  # Flatten array

        if state_representation == 2:  # List agent/POI representation fully obserbavle
            state = np.zeros(self.num_agents * 2 + self.num_poi * 2)
            for id, poi in enumerate(self.poi_list):
                if True: #not poi.is_observed:  # For all POI's that are still active
                    x1 = poi.position[0] - agent.position[0];
                    x2 = -1
                    y1 = poi.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    state[2 * id] = angle / 360.0
                    state[2 * id + 1] = dist / (2.0 * self.dim_col)

            for id, other_agent in enumerate(self.agent_list):
                if other_agent != agent:  # FOR ALL AGENTS MINUS MYSELF
                    x1 = other_agent.position[0] - agent.position[0];
                    x2 = -1
                    y1 = other_agent.position[1] - agent.position[1];
                    y2 = 0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    state[2 * self.num_poi + 2 * id] = angle /360.0
                    state[2 * self.num_poi + 2 * id + 1] = dist / (2.0 * self.dim_col)
            state = np.reshape(state, (1, self.num_agents * 2 + self.num_poi * 2))  # Flatten array

        if state_representation == 3:  # Binary state representation
            x_beg = self.agent_pos[agent_id][0] - self.observe
            y_beg = self.agent_pos[agent_id][1] - self.observe
            x_end = self.agent_pos[agent_id][0] + self.observe + 1
            y_end = self.agent_pos[agent_id][1] + self.observe + 1
            state = np.copy(self.state)
            state = state[x_beg:x_end, :]
            state = state[:, y_beg:y_end]
            state = np.reshape(state, (1, pow(self.observe * 2 + 1, 2)))  # Flatten array
            k = np.reshape(np.zeros(len(state[0]) * 4), (len(state[0]), 4))  # 4-bit encoding
            for i in range(len(state[0])):
                k[i][int(state[0][i])] = 1
            k = np.reshape(k, (1, len(state[0]) * 4))  # Flatten array
        return state



    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two agents relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  #dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    def update_poi_observations(self):
        # Check for credit assignment
        for poi in self.poi_list:  # POI COUPLED
            soft_stat = []
            for agent_id, agent in enumerate(self.agent_list):
                if abs(poi.position[0] - agent.position[0]) <= self.obs_dist and abs(
                            poi.position[1] - agent.position[1]) <= self.obs_dist:  # and self.goal_complete[poi_id] == False:
                    soft_stat.append(agent_id)
            if len(soft_stat) >= self.coupling:  # If coupling requirement is met
                poi.is_observed = True
                poi.observation_history.append(soft_stat)  # Store the identity of agents aiding in meeting that tight coupling requirement

    def check_goal_complete(self):
        is_complete = True
        for poi in self.poi_list:
            is_complete *= poi.is_observed
        return is_complete

    def get_reward(self):
        global_reward = 0 #Global reward obtained
        for poi in self.poi_list: global_reward += 1.0 * poi.is_observed
        global_reward /= self.parameters.num_poi #Scale between 0 and 1

        rewards = np.zeros(self.parameters.num_agents) #Rewards decomposed to the team
        if self.parameters.D_reward: #Difference reward scheme
            for poi in self.poi_list:
                if poi.is_observed:
                    no_reward = False
                    for ids in poi.observation_history:
                        if len(ids) > self.parameters.coupling:  # Only if it's observed by exactly the numbers needed
                            no_reward = True;
                            break;
                    if not no_reward:
                        for agent_id in poi.observation_history[0]:
                            rewards[agent_id] += 1.0 / self.parameters.num_poi  # Reward the first group of agents to get there
        else:
            rewards += global_reward  # Global reward scheme

        return rewards, global_reward



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

    def add_mpc(self, gridworld, parameters):
        all_mpc = np.zeros(parameters.num_agents)
        for id, agent in enumerate(gridworld.agent_list): all_mpc[id] = agent.evo_net.delta_mpc
        self.avg_mpc = np.average(all_mpc) #Average mpc
        self.mpc_std = np.std(all_mpc)


    def save_csv(self, generation):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt('avg_fitness.csv', np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class Population(): #Keras population
    def __init__(self, input_size, hidden_nodes, output, population_size, elite_fraction = 0.2):
        self.population_size = population_size
        self.elite_fraction = int(elite_fraction * population_size)
        self.net_pop = [] #List of networks
        for i in range(population_size): self.net_pop.append(self.init_net(input_size, hidden_nodes, output))
        self.pop_handle = np.zeros(population_size * 2, dtype=np.float64) #COntrols the indexing to the net population (net_pop) and fitness values
        self.pop_handle = np.reshape(self.pop_handle, (population_size, 2))
        for x in range(population_size):  self.pop_handle[x][0] = x #Initializing our net population indexing
        self.longest_survivor = 0
        self.best_net_index = 0 #Current index of the best net


    def init_net(self, input_size, hidden_nodes, output):
        model = Sequential()
        model.add(Dense(hidden_nodes, input_dim=input_size, init='he_uniform'))
        model.add(Activation('sigmoid'))
        model.add(Dense(output))
        model.add(Activation('softmax'))
        model.compile(loss='mean_absolute_error', optimizer='Nadam')
        return model

    def Epoch(self):
        self.pop_handle = self.pop_handle[self.pop_handle[:, 1].argsort()]  ##Ranked on fitness (reverse of weakness) s.t. 0 index is the best
        if int(self.pop_handle[0][0]) == self.best_net_index: self.longest_survivor += 1 #Check if the leader candidate is the same one
        else:
            self.longest_survivor = 0; self.best_net_index = int(self.pop_handle[0][0])
        self.best_net_index = self.pop_handle[0][0] #Update the leader candidate
        for x in range(self.elite_fraction, self.population_size): #Mutate to renew population
            many = randint(1,5); much = randint(1,10)
            if (randint(1,100) == 91):
                many = randint(1,10); much = randint(1,100)
            self.mutate(self.net_pop[int(self.pop_handle[x][0])], self.net_pop[int(self.pop_handle[x][0])], many, much) #Mutate same model in and out

    def mutate(self, model_in, model_out, many_strength=1, much_strength=1):
        # NOTE: Takes in_num file, mutates it and saves as out_num file, many_strength denotes how many mutation while
        # much_strength controls how strong each mutation is

        w = model_in.get_weights()
        for many in range(many_strength):  # Number of mutations
            i = randint(0, len(w) - 1)
            if len(w[i].shape) == 1:  # Bias
                j = randint(0, len(w[i]) - 1)
                w[i][j] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
                # if (randint(1, 100) == 5): #SUPER MUTATE
                #     w[i][j] += np.random.normal(-1 * much_strength, 1 * much_strength)
            else:  # Bias
                j = randint(0, len(w[i]) - 1)
                k = randint(0, len(w[i][j]) - 1)
                w[i][j][k] += np.random.normal(-0.1 * much_strength, 0.1 * much_strength)
                # if (randint(1, 100) == 5):  # SUPER MUTATE
                #     w[i][j][k] += np.random.normal(-1 * much_strength, 1 * much_strength)
        model_out.set_weights(w)  # Save weights






def init_nn(input_size, hidden_nodes, middle_layer = False, weights = 0):
    model = Sequential()


    if middle_layer:
        model.add(Dense(hidden_nodes, input_dim=input_size, weights=weights, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    else:
        model.add(Dense(hidden_nodes, input_dim=input_size, init='he_uniform', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(LeakyReLU(alpha=.2))
    #model.add(SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one'))
    model.add(Activation('sigmoid'))
    #model.add(Dropout(0.1))
    #model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if not middle_layer:
        model.add(Dense(input_size, init= 'he_uniform')) #Output of the prediction module
    model.compile(loss='mse', optimizer=sgd)

    # if pretrain: #Autoencoder pretraining
    #     model.fit(train_x, train_x, nb_epoch=50, batch_size=32, shuffle=True, validation_data=(valid_x, valid_x),
    #                     verbose=1)
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

def save_model(model):
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    model.save_weights('shaw_2/model_weights.h5',overwrite=True)

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


def novelty(weak_matrix, archive, k = 10):
    import bottleneck
    #Handle early gens with archive size less that 10
    if (len(archive) < k):
        k = len(archive)

    novel_matrix = np.zeros(len(archive))
    for i in range(len(archive)):
        novel_matrix[i] = np.sum(np.square(weak_matrix - archive[i]))

    #k-nearest neighbour algorithm
    k_neigh = bottleneck.partsort(novel_matrix, k)[:k] #Returns a subarray of k smallest novelty scores

    #Return novelty score as the average Euclidean distance (behavior space) between its k-nearest neighbours
    return np.sum(k_neigh)/k

def import_arch(seed = 'Evolutionary/seed.json'): #Get model architecture
    import json
    from keras.models import model_from_json
    with open(seed) as json_file:
        json_data = json.load(json_file)
    model_arch = model_from_json(json_data)
    return model_arch

def rec_weakness(setpoints, initial_state, model, n_prev=7, novelty = False, test = False): #Calculates weakness (anti fitness) of RECCURRENT models
    weakness = np.zeros(19)
    input = np.reshape(train_data[0:n_prev], (1, n_prev, 21))  #First training example in its entirety

    for example in range(len(train_data)-n_prev):#For all training examples
        model_out = model.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(19):
            weakness[index] += math.fabs(model_out[0][index] - train_data[example+n_prev][index])#Time variant simulation
        #Fill in new input data
        for k in range(len(model_out)): #Modify the last slot
            input[0][0][k] = model_out[0][k]
            input[0][0][k] = model_out[0][k]
        #Fill in two control variables
        input[0][0][19] = train_data[example+n_prev][19]
        input[0][0][20] = train_data[example+n_prev][20]
        input = np.roll(input, -1, axis=1)  # Track back everything one step and move last one to the last row
    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(train_data)-n_prev)
    else:
        return np.sum(np.square(weakness))

def ff_weakness(setpoints, initial_state, simulator, model, novelty = False, test = False, actuator_noise = 0, sensor_noise = 0, sensor_failure = None, actuator_failure = None): #Calculates weakness (anti fitness) of FEED-FORWARD models

    indices = [11,15]
    weakness = np.zeros(len(indices))
    # input = np.append(initial_state, setpoints[0])
    # input = np.reshape(input, (1,23))
    input = np.copy(initial_state)
    input[0][19] = setpoints[0][0]
    input[0][20] = setpoints[0][1]


    for example in range(len(setpoints)-1):#For all training examples
        #Add noise to the state input to the controller
        noise_input = np.copy(input)
        if sensor_noise != 0: #Add sensor noise
            for i in range(19):
                std = sensor_noise * abs(noise_input[0][i]) / 100.0
                if std != 0:
                    noise_input[0][i] +=  np.random.normal(0, std/2.0)

        if sensor_failure != None: #Failed sensor outputs 0 regardless
            for i in sensor_failure:
                noise_input[0][i] = 0

        # Get the controller output
        control_out = model.predict(noise_input)

        #Add actuator noise (controls)
        if actuator_noise != 0:
            for i in range(len(control_out[0])):
                std = actuator_noise * abs(control_out[0][i]) / 100.0
                if std != 0:
                    control_out[0][i] +=  np.random.normal(0, std/2.0)

        if actuator_failure != None: #Failed actuator outputs 0 regardless
            for i in actuator_failure:
                control_out[0][i] = 0

        # Fill in the controls
        input[0][19] = control_out[0][0]
        input[0][20] = control_out[0][1]

        #Use the simulator to get the next state
        model_out = simulator.predict(input) #Time domain simulation
        #Calculate error (weakness)
        for index in range(len(indices)):
            weakness[index] += math.fabs(model_out[0][indices[index]] - setpoints[example][index])#Time variant simulation

        #Fill in new input data
        for k in range(len(model_out[0])):
            input[0][k] = model_out[0][k]
            input[0][k] = model_out[0][k]

        #Fill in next setpoints
        input[0][19] = setpoints[example+1][0]
        input[0][20] = setpoints[example+1][1]

    if (novelty):
        return weakness
    elif (test == True):
        return np.sum(weakness)/(len(setpoints)-1)
    else:
        return np.sum(np.square(weakness))


def get_first_state(self, agent_id, use_rnn, sensor_avg,
                    state_representation):  # Get first state, action input to the q_net
    if not use_rnn:  # Normal NN
        st = self.get_state(agent_id, sensor_avg, state_representation)
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