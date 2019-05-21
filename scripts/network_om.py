import os
import re
import numpy as np
import tensorflow as tf
import keras
import time
from tf.keras.layers import Activation, Dense

class Actions():
    # Define 11 choices of actions to be:
    # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
    # [0.5*v_pref,  [-pi/6, 0, pi/6]]
    # [0,           [-pi/6, 0, pi/6]]
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class NetworkCore(tf.keras.Model):
    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def predict_p(self, x, audio):
        return self.sess.run(self.softmax_p, feed_dict={self.x: x})

    def simple_load(self, filename=None):
        if filename is None:
            print ("[network.py] Didn't define simple_load filename")
        self.saver.restore(self.sess, filename)

class NetworkFull(NetworkCore):
    def __init__(self, device, model_name, num_actions, with_sarl):
        super(self.__class__, self).__init__(device, model_name)
        self.num_actions = num_actions
        self.with_sarl = with_sarl
        if self.with_sarl:
            self.nn_sarl = NetworkSarl();
        #else: with_cadrl
        if self.with_om:
            self.nn_om = NetworkOm();
        #(TODO)if self.with_gp:
        #(TODO)if self.with_rnn:

    def _create_graph_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, Config.NN_INPUT_SIZE], name='X')

    def _create_graph_outputs(self):
        # FCN
        self.layer_out = tf.layers.dense(inputs=self.final_flat, units = 256, use_bias = True, activation=tf.nn.relu, name = 'layer_out')

        # Cost: p
        self.logits_p = tf.layers.dense(inputs = self.layer_out, units = self.num_actions, name = 'logits_p', activation = None)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)

    def _create_graph(self):
        self._create_graph_inputs()

        if Config.USE_REGULARIZATION:
            regularizer = keras.regularizers.l2(0.)
        else:
            regularizer = None

        if Config.NORMALIZE_INPUT:
            self.avg_vec = tf.constant(Config.NN_INPUT_AVG_VECTOR, dtype = tf.float32)
            self.std_vec = tf.constant(Config.NN_INPUT_STD_VECTOR, dtype = tf.float32)
            self.x_normalized = (self.x - self.avg_vec) / self.std_vec
        else:
            self.x_normalized = self.x
        #Other networks
        self.host_agent_vec = self.x_normalized[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
        self.sarl_output = self.nn_sarl._create_graph(self.x_normalized)
        self.om_output = self.nn_om._create_graph(self.x_om)
        self.layer_full1_input = tf.concat([self.host_agent_vec, self.sarl_output, self.om_output], 1, name='layer_full1_input')
        self.layer_full1 = tf.layers.dense(inputs=self.layer_full1_input, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer_full1')
        self.layer_full2 = tf.layers.dense(inputs=self.layer_full1, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer_full2')

        self.final_flat = tf.contrib.layers.flatten(self.layer_full2)

        self._create_graph_outputs() #Receive probabilities for actions

class NetworkSarl(NetworkCore):
    def __init__(self device, model_name):
        super(self.__class__, self).__init__(device, model_name)
        self.mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        self.mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        self.mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        self.attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        self.with_global_state = config.getboolean('sarl', 'with_global_state')
        self.self_state_dim = Config.HOST_AGENT_OBSERVATION_LENGTH
        self.human_state_dim = Config.OTHER_AGENT_OBSERVATION_LENGTH
        self.joint_state_dim = self.self_state_dim + self.human_state_dim
        self.gamma = config.getfloat('rl', 'gamma')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')
        self.input_dim = self.input_dim()

    def mlp(input_dim, mlp_dims, last_relu=False):
        model = tf.keras.Sequential()
        mlp_dims = [input_dim] + mlp_dims
        model.add(Dense(mlp_dims[1], input_dim = mlp_dims[0], activation = 'relu'))
        for i in range(len(mlp_dims) - 2):
            model.add(Dense(mlp_dims[i + 2]))
            if i != len(mlp_dims) - 3 or last_relu:
                model.add(Activation('relu'))
        return model

    def _create_graph(self):
        self.global_state_dim = self.mlp1_dims[-1]
        self.mlp1 = mlp(self.input_dim, self.mlp1_dims, last_relu=True)
        self.mlp2 = mlp(self.mlp1_dims[-1], self.mlp2_dims)
        if self.with_global_state:
            self.attention = mlp(self.mlp1_dims[-1] * 2, self.attention_dims)
        else:
            self.attention = mlp(self.mlp1_dims[-1], self.attention_dims)
        mlp3_input_dim = self.mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, self.mlp3_dims)
        self.attention_weights = None

    def call(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        self.mlp1_layer1 = tf.layers.dense(inputs=state.view((-1, size[2])), units=150, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer_full1')
        self.mlp1_layer2 = tf.layers.dense(inputs=state.view((-1, size[2])), units=100, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer_full1')
         = tf.layers.dense(inputs=self.layer_full1, units=256, activation=tf.nn.relu, kernel_regularizer=regularizer, name = 'layer_full2')
        mlp1_output = self.mlp1()
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        sarl_output = self.mlp3(joint_state)
        return sarl_output

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

class NetworkOm(NetworkCore):
    def __init__(self device, model_name):
        super(self.__class__, self).__init__(device, model_name)

    def _create_graph(self):

    def call(self, state):



class SDOADRL(): #Static and dynamic obstacle avoidance using deep reinforcement learning
    def __init__(self, model_name):
        self.name = 'sdoardl'

    def configure(self, config):
        self.set_parameters(config)
        a = Actions()
        self.num_actions = a.num_actions
        self.model = NetworkFull(self.device, self.name, self.num_actions, self.with_sarl)
        #self.multiagent_training = config.getboolean('sdoardl', 'ga3c')
        logging.info('Policy: {} {} sarl'.format(self.name, 'w/' if self.with_sarl else 'w/o'))

    def set_parameters(self, config):
        #fill with parameters
        self.with_sarl = config.getboolean('sdoardl', 'with_sarl')
        self.device = Config.DEVICE



class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    USE_REGULARIZATION  = True
    ROBOT_MODE          = True
    EVALUATE_MODE       = True

    SENSING_HORIZON     = 8.0

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
    MULTI_AGENT_ARCH = 'RNN'

    DEVICE                        = '/cpu:0' # Device

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])

    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH



if __name__ == '__main__':
    actions = Actions().actions
    num_actions = Actions().num_actions
    nn = NetworkVP_rnn(Config.DEVICE, 'network', num_actions)
    nn.simple_load()

    obs = np.zeros((Config.FULL_STATE_LENGTH))
    obs = np.expand_dims(obs, axis=0)

    num_queries = 10000
    t_start = time.time()
    for i in range(num_queries):
        obs[0,0] = 10 # num other agents
        obs[0,1] = np.random.uniform(0.5, 10.0) # dist to goal
        obs[0,2] = np.random.uniform(-np.pi, np.pi) # heading to goal
        obs[0,3] = np.random.uniform(0.2, 2.0) # pref speed
        obs[0,4] = np.random.uniform(0.2, 1.5) # radius
        predictions = nn.predict_p(obs, None)[0]
    t_end = time.time()
    print ("avg query time:", (t_end - t_start)/num_queries)
    print ("total time:", t_end - t_start)
    # action = actions[np.argmax(predictions)]
    # print "action:", action
