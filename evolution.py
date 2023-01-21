from player import Player
import numpy as np
from config import CONFIG
import random
import copy
from operator import attrgetter 
import os 
class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def add_gaussian_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)

    def mutate(self, child):
        # TODO
        # child: an object of class `Player`
        threshold = 0.2

        self.add_gaussian_noise(child.nn.W1, threshold)
        self.add_gaussian_noise(child.nn.W2, threshold)
        self.add_gaussian_noise(child.nn.b1, threshold)
        self.add_gaussian_noise(child.nn.b2, threshold)

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        row_size, column_size = child1_array.shape
        section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:section_1, :] = parent1_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent1_array[section_2:, :]

            child2_array[:section_1, :] = parent2_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent2_array[section_2:, :]
        else:
            child1_array[:section_1, :] = parent2_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent2_array[section_2:, :]

            child2_array[:section_1, :] = parent1_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent1_array[section_2:, :]

    # def q_tournament(self, players, q):
    #     q_selected = np.random.choice(players, q)
    #     return max(q_selected, key=lambda player: player.fitness)

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover
            children = []
            parents = []
            for _ in range(num_players):
                parents.append(self.q_tournament(prev_players, q=3))

            for i in range(0, len(parents), 2):
                child1 = Player(self.mode)
                child2 = Player(self.mode)

                self.crossover(child1.nn.W1, child2.nn.W1, parents[i].nn.W1, parents[i + 1].nn.W1)
                self.crossover(child1.nn.W2, child2.nn.W2, parents[i].nn.W2, parents[i + 1].nn.W2)
                self.crossover(child1.nn.b1, child2.nn.b1, parents[i].nn.b1, parents[i + 1].nn.b1)
                self.crossover(child1.nn.b2, child2.nn.b2, parents[i].nn.b2, parents[i + 1].nn.b2)

                self.mutate(child1)
                self.mutate(child2)
                children.append(child1)
                children.append(child2)
            return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
    
    def q_tournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)

    def roulette_wheel(self , players , num_player):
        fitness_sum = sum([player.fitness for player in players])
        probabilities = [player.fitness / fitness_sum for player in players]
        nex_generation = np.random.choice(players, size=num_player, p=probabilities, replace=False)
        return nex_generation.tolist()

    def save_fitness(self, players):
        if not os.path.exists('fitness'):
            os.makedirs('fitness')

        f = open("fitness/output1.txt", "a")
        for p in players:
            f.write(str(p.fitness))
            f.write(" ")
        f.write("\n")
        f.close()
        
    def mutation(self, new_players):
        k = random.randint(0, len(new_players[0]) * len(new_players))  # change how many cells
        for j in range(len(new_players)):
            for i in range(k):
                yes_no = random.choices([0, 1], weights=(80, 20), k=1)  # yes or no
                if yes_no == 1:
                    change = random.randint(0, len(new_players[0]) * len(new_players))
                    new_players[change % len(new_players[0]), change % len(new_players)] = 0  # reset to zero
        return new_players

    def sus(self, players, num_players):
        total_fitness = np.sum([p.fitness for p in players])
        point_distance = total_fitness / num_players
        start_point = np.random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(num_players)]
   
        next_generation = []
        for point in points:
            i = 0
            f = 0
            while f < point:
                f += players[i].fitness
                i += 1
            next_generation.append(players[i - 1])
        return next_generation

    def apply_crossover(self, prev_players):
        new_players = []

        for i in range(0, len(prev_players), 2):
            i1 = prev_players[i]
            i2 = prev_players[i+1]

            new_child1 = self.clone_player(i1)
            new_child2 = self.clone_player(i2)

            for i in range(len(new_child1.nn.w)):
                shape = new_child1.nn.w[i].shape

                new_child1.nn.w[i][:, int(shape[1]/2):] = i2.nn.w[i][:, int(shape[1]/2):]
                new_child2.nn.w[i][:, int(shape[1] / 2):] = i1.nn.w[i][:, int(shape[1]/ 2):]

            for i in range(len(new_child1.nn.b)):
                shape = new_child1.nn.w[i].shape

                new_child1.nn.b[i][:, int(shape[1] / 2):] = i2.nn.b[i][:, int(shape[1] / 2):]
                new_child2.nn.b[i][:, int(shape[1] / 2):] = i1.nn.b[i][:, int(shape[1] / 2):]

            new_players.append(new_child1)
            new_players.append(new_child2)

        return new_players
    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        players = self.roulette_wheel(players, num_players)
        self.save_fitness(players)
        return players[: num_players]
