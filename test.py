import numpy as np, sys, time
import mxnet

def putracking():
    from pympler import tracker
    tr = tracker.SummaryTracker()
    #tr.print_diff()

    #tracker = statistics()
    gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr = angled_repr, angle_res = angle_res) #Create gridworld
    #mod.dispGrid(gridworld)
    #tr = tracker.SummaryTracker()
    tr.print_diff()


class benchmark:
    def __init__(self):
        k = 1


    def benchmark_data(self):
        test_x = np.arange(500)
        return test_x

    def fann_net(self):
        from fann2 import libfann
        ann = libfann.neural_net()
        ann.create_standard_array([3, 500,50, 1])
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
        return ann

    def keras_net(self):
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        model = Sequential()
        model.add(Dense(50, input_dim=500, init='he_uniform'))
        model.add(Activation('sigmoid'))
        model.add(Dense(1, init= 'he_uniform'))
        model.compile(loss='mse', optimizer='Nadam')
        return model

    def mxnet(self):
        import mxnet as mx
        data = mx.symbol.Variable('data')
        fc1 = mx.symbol.FullyConnected(data=data, num_hidden=500)
        act1 = mx.symbol.Activation(data=fc1, act_type="sigmoid")
        fc3 = mx.symbol.FullyConnected(data=act1, num_hidden=1)
        mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
        model = mx.model.FeedForward(
            symbol=mlp,
            num_epoch=20,
            learning_rate=.1)

        #mx.set.seed(0)
        #model = mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax", num.round = 20, array.batch.size = 15, learning.rate = 0.07, momentum = 0.9, eval.metric = mx.metric.accuracy)
        return model


    def pybrain(self):
        from pybrain.tools.shortcuts import buildNetwork
        net = buildNetwork(500, 50, 1)
        return net


    def benchmark(self):
        test_x = self.benchmark_data()
        ann = self.fann_net()
        model = self.keras_net()
        mx_model = self.mxnet()

        curtime = time.time()
        for i in range (1000000):
            ann.run(test_x)
        elapsed = time.time() - curtime
        print elapsed
        #
        # test_x_ig = np.reshape(test_x, (1,len(test_x)))
        # curtime = time.time()
        # for i in range (1000000):
        #     model.predict(test_x_ig)
        # elapsed = time.time() - curtime
        # print elapsed


        # mx_model.fit(test_x, test_x)
        # #test_x = np.reshape(test_x, (1,len(test_x)))
        # curtime = time.time()
        # for i in range (1000):
        #     mx_model.predict(X = test_x)
        # elapsed = time.time() - curtime
        # print elapsed

        # pybrain_net = self.pybrain()
        # #test_x = np.reshape(test_x, (1,len(test_x)))
        # curtime = time.time()
        # for i in range (1000000):
        #     pybrain_net.activate(test_x)
        # elapsed = time.time() - curtime
        # print elapsed




if __name__ == "__main__":

    #bench = benchmark()
    #bench.benchmark()

    from fann2 import libfann

    ann = libfann.neural_net()
    ann.create_standard_array([3, 500, 50, 1])
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

    t = []
    #k = ann.get_connection_array()



















