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

    def mutate(self, child):
        # TODO
        # child: an object of class `Player`
        mutation_threshold = 0.7
        center = 0
        margin = 0.3

        for i in range(len(child.nn.w)):
            if np.random.random_sample() >= mutation_threshold :
                child.nn.w[i] += np.random.normal(center, margin, size=(child.nn.w[i].shape))
        
        for i in range(len(child.nn.b)):
            if np.random.random_sample() >= mutation_threshold:
                child.nn.b[i] += np.random.normal(center, margin, size=(child.nn.b[i].shape))



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
            new_players = []
            parents = prev_players

            for i in range(num_players) : 
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                W1 = np.zeros((parent1.nn.layer2, parent1.nn.layer1))
                W2 = np.zeros((parent1.nn.layer3, parent1.nn.layer2))
                b1 = np.zeros((parent1.nn.layer2, 1))
                b2 = np.zeros((parent1.nn.layer3, 1))

                # cross over for W1
                for j in range(len(parent1.nn.W1)):
                    if j % 2 == 0 :
                        W1[j] = parent1.nn.W1[j]
                    else : 
                        W1[j] = parent2.nn.W1[j]
                
                # cross over for W2
                for j in range(len(parent1.nn.W2)):
                    if j % 2 == 0 :
                        W2[j] = parent1.nn.W2[j]
                    else : 
                        W2[j] = parent2.nn.W2[j]

                # cross over for b1
                for j in range(len(parent1.nn.b1)):
                    if j % 2 == 0 :
                        b1[j] = parent1.nn.b1[j]
                    else : 
                        b1[j] = parent2.nn.b1[j]

                # cross over for b2
                for j in range(len(parent1.nn.b2)):
                    if j % 2 == 0 :
                        b2[j] = parent1.nn.b2[j]
                    else : 
                        b2[j] = parent2.nn.b2[j]
                child = self.clone_player(parent1)

                W1 = self.mutation(W1)
                W2 = self.mutation(W2)
                b1 = self.mutation(b1)
                b2 = self.mutation(b2)
                new_players.append(child)
            return new_players
    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
    
    def q_tournament(self ,players ,num_players ,q ):
        selected = []
        for i in range(num_players) :
             q_selections = np.random.choice(players, q)
             selected.append(max(q_selections, key=attrgetter('fitness')))
        return selected

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
        players = self.q_tournament(players, num_players, 4)
        self.save_fitness(players)
        return players[: num_players]
