import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make('CartPole-v0')

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, output_dim, test_run):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.test_run = test_run


    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []

        hidden_weight = []
        out_weight = []

        input_nodes = 4

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)
            inp_b = np.random.rand((input_nodes))
            input_bias.append(inp_b)
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)

        return [input_weight, input_bias, hidden_weight, out_weight]

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):

        obs = obs/max(np.max(np.linalg.norm(obs)), 1)
        Ain = self.relu(obs@input_w + input_b.T)
        Ahid = self.relu(Ain@hidden_w)
        Zout = Ahid @ out_w
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w):
        obs = env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        for ep in range(self.test_run):
            score = self.run_environment(
                input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):

        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA_list):


        newDNAs = []
        length_of_dnalist = len(DNA_list)
        for i in range(self.population_size - length_of_dnalist):
            rand_1 = random.randrange(length_of_dnalist)
            rand_2 = random.randrange(length_of_dnalist)
            dna_1 = DNA_list[rand_1]
            dna_2 = DNA_list[rand_2]
            rand_w1 = random.randrange(16)
            rand_b1 = random.randrange(4)
            rand_w2 = random.randrange(8)
            rand_b2 = random.randrange(4)
            newDNA = []
            newDNA = np.concatenate((dna_1[:rand_w1], dna_2[rand_w1:16]))
            newDNA = np.concatenate((newDNA, np.concatenate((dna_1[16:16+rand_b1], dna_2[rand_b1+16:20]))))
            newDNA = np.concatenate((newDNA, np.concatenate((dna_1[20:20+rand_w2], dna_2[rand_w2+20:28]))))
            newDNA = np.concatenate((newDNA, np.concatenate((dna_1[28:rand_b2+28], dna_2[rand_b2+28:32]))))
            newDNAs.append(newDNA)
            # parent_w1 = DNA_list[rand_1][0:16] + DNA_list[rand_2][0:16]
            # parent_b1 = DNA_list[rand_1][16:20] + DNA_list[rand_2][16:20]
            # parent_w2 = DNA_list[rand_1][20:28] + DNA_list[rand_2][20:28]
            # parent_b2 = DNA_list[rand_1][28:32] + DNA_list[rand_2][28:32]
            # newdna = []
            # newdna += random.sample(list(parent_w1),16)
            # newdna += random.sample(list(parent_b1),4)
            # newdna += random.sample(list(parent_w2),8)
            # newdna += random.sample(list(parent_b2),4)
            # newDNAs.append(newdna)

        return newDNAs

    def mutation(self, DNA):

        #other mutation techniques have been included in comments
        max_gene = max(DNA)
        min_gene = min(DNA)
        for gene in DNA:
            if random.random() < self.mutation_rate:
                creep_change = random.uniform(-0.01,0.01)
                gene += creep_change

                # gene = random.uniform(min_gene,max_gene)

                # mutation = random.gauss(0, 0.1)
                # gene += mutation
        return DNA

    def next_generation(self):
        index_good_fitness = [] #index of parents selected for crossover.
        #fill the list.
        fitness_list = []
        for i in range(self.population_size):
            fitness_list.append([self.current_fitness[i],i])
        fitness_list.sort()
        fitness_list.reverse()
        for i in range(10):
            index_good_fitness.append(fitness_list[i][1])
        new_DNA_list = []
        new_fitness_list = []
        #print(index_good_fitness)
        DNA_list = []
        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[1], -1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)

        #parents selected for crossover moves to next generation
        new_DNA_list += DNA_list
        # print("PARENTS SELECTED FOR CROSSOVER")
        # print(DNA_list)
        new_DNA_list += self.crossover(DNA_list)
        # print("AFTER CROSSOVER")
        # print(new_DNA_list)
        #mutate the new_DNA_list
        for dna in new_DNA_list:
            dna = self.mutation(dna)
        # print("AFTER MUTATIONS")
        # print(new_DNA_list)
        #converting 1D representation of individual back to original (required for forward pass of neural network)
        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:

            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            #evaluate fitness of new individual and add to new_fitness_list.
            #check run_environment function for details.
            learner = self.learner
            new_fitness_list.append(learner.run_environment(new_in_w,new_in_b,new_hid_w,new_out_w))

        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]

        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        plt.plot(self.fitness_list)
        plt.show()
        #plot

    def evolve(self):
        #evolve
        for _ in range(self.number_of_generation):
            self.current_generation, self.current_fitness = self.next_generation()
            max_fitness_value = max(self.current_fitness)
            if max_fitness_value > self.best_fitness:
                self.best_fitness = max_fitness_value
                i = self.current_fitness.index(max_fitness_value)
                self.best_gen = [self.current_generation[0][i],self.current_generation[1][i],self.current_generation[2][i],self.current_generation[3][i]]
            self.fitness_list.append(max_fitness_value)
        #self.show_fitness_graph()
        print(self.best_fitness)
        return self.best_gen


def trainer():
    pop_size = 15
    num_of_generation = 100
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, pop_size)
    init_weight_list, init_fitness_list = learner.run_test()
    #instantiate the GA optimizer
    ga_optimizer = GA(init_weight_list,init_fitness_list,num_of_generation,pop_size,learner)
    #call evolve function to obtain optimized weights.
    print(type(ga_optimizer.current_generation[1]))
    params = ga_optimizer.evolve()
    return params
    #return optimized weights

def test_run_env(params):
    input_w, input_b, hidden_w, out_w = params
    obs = env.reset()
    score = 0
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, 15)
    for t in range(5000):
        env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, done, info = env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if done:
            break
    print(f"Final score: {score}")

def main():
    params = trainer()
    test_run_env(params)


if(__name__ == "__main__"):
    main()