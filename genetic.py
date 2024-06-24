from copy import copy, deepcopy
import multiprocessing
import random
import time
from typing import Callable, List

from matplotlib import pyplot as plt
import numpy as np

def format_time_difference(start_time, end_time):
    time_difference = end_time - start_time

    # Calculate hours, minutes, and seconds
    hours, remainder = divmod(time_difference, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"hours: {int(hours)}, minutes: {int(minutes)}, seconds: {int(seconds)}"


def create_dataset(data: List, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    input, target = [], []
    for i in range(len(data) - window_size):
        input.append(data[i:i+window_size])
        target.append(data[i+window_size])
    return np.array(input), np.array(target)


def create_datasets(ticker_data, dataset_dict={}, train_test_split=0.8, window_range=(5,100)) -> dict:
    train_size = int(len(ticker_data) * train_test_split)
    train_data = ticker_data[:train_size]
    test_data = ticker_data[train_size:]
    if len(window_range) == 2:
        start, stop = window_range
        step = 1
    else:
        start, stop, step = window_range
    stop += 1

    for window_size in range(start, stop, step):
        x_test, y_test = create_dataset(test_data, window_size)
        x_train, y_train = create_dataset(train_data, window_size)
        dataset_dict[window_size] = {
            'x_test': x_test,
            'y_test': y_test,
            'x_train': x_train,
            'y_train': y_train
        }
    return dataset_dict


def create_plot(test, pred, ticker_name, plt_size_x=10, plt_size_y=6, c_test='b', c_pred='r-', linewidth_test=2, linewidth_pred=0.75, title=''):
    plt.figure(figsize=(plt_size_x, plt_size_y))
    plt.plot(np.arange(len(test)), test, c_test, label="Actual", linewidth=linewidth_test)
    plt.plot(np.arange(len(pred)), pred, c_pred, label="Prediction", linewidth=linewidth_pred)
    if title == '':
        title = f"{ticker_name} Price Prediction"
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prediction_plot.pdf', format='pdf')
    plt.show()

class Creature():
    def __init__(self, param_space: dict, mutable_params: list, objective: Callable, data_dict: dict, params={}) -> None:
        self._params = params
        self._param_space = param_space
        self._mutable_params = mutable_params
        self._objective = objective
        self._data_dict = data_dict
        self._score = None

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, params) -> None:
        self._params = params

    @property
    def score(self) -> None | float:
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = score

    @property
    def fitness(self) -> float:
        if self._score is None:
            if len(self._params) == 0:
                self.generate()
            self._score = self._objective(self._params, self._data_dict[self._params['window_size']])
        return 1/self._score

    def mutate(self, stats_dict=None, inplace=True, mutation_chance=0.1, rand_param=False) -> dict:
        mutated_params = self._params.copy() if not inplace else self._params   # create a copy of params dictionary or copy a reference
        for key in self._mutable_params:
            if random.random() < mutation_chance:
                floor, ceil, step = self._param_space[key]
                if rand_param:  # set random value from range defined in param_space
                    new_value = random.uniform(floor, ceil)
                    new_value = round((new_value - floor) / step) * step + floor
                else:   # increment or decrement current value by one step
                    new_value = mutated_params[key] + step * random.choice([-1, 1])
                    new_value = min(ceil, new_value)    # keep new value in range
                    new_value = max(floor, new_value)   # keep new value in range
                mutated_params[key] = new_value
                if stats_dict:
                    stats_dict['mutations'] = stats_dict['mutations'] + 1
        return mutated_params

    def generate(self) -> None:
        self._params = {}
        for param_name, (floor, ceil, step) in self._param_space.items():
            new_value = random.uniform(floor, ceil)
            new_value = round((new_value - floor) / step) * step + floor
            self._params[param_name] = new_value

    def __str__(self) -> str:
        return str(self._params) + '\n' + f'Mae: {self._score}'

    def reset_score(self) -> None:
        self._score = None

    def get_copy(self) -> 'Creature':
        new_creature = copy(self)
        new_creature.params = deepcopy(self._params)
        new_creature.reset_score()
        return new_creature

    def get_params_hash(self) -> float:
        return hash(self._params.values())


def get_fitness_wrapper(creature) -> float:
    return creature.fitness


def roulette_selection(creatures:List[Creature], num_creatures:int) -> List[Creature]:
    probs: np.ndarray = np.array(list(x.fitness for x in creatures))
    probs = np.power(probs, 3)  # use power to make differences in model performance more apparent
    probs = probs / np.sum(probs)   # scale probs such that the sum is 1
    return list(np.random.choice(np.array(creatures), num_creatures, replace=False, p=probs))


def tournament_selection(creatures:List[Creature], tournament_size:int) -> List[Creature]:
    return creatures[:tournament_size]


def one_point_crossover(c_1: Creature, c_2: Creature) -> tuple[Creature, Creature]:
    keys = list(c_1.params.keys())
    random.shuffle(keys)
    cross_point = len(keys) // 2
    keyset_1, keyset_2 = keys[cross_point:], keys[:cross_point]
    c_3, c_4 = c_1.get_copy(), c_2.get_copy()
    for key in keyset_1:
        c_3.params[key], c_4.params[key] = c_1.params[key], c_2.params[key]
    for key in keyset_2:
        c_3.params[key], c_4.params[key] = c_2.params[key], c_1.params[key]
    return c_3, c_4


class GeneticLearner():
    def __init__(self, *args, **kwargs) -> None:
        self._selection_funcs = {
            'tournament': tournament_selection,
            'roulette': roulette_selection
        }
        self._crossover_funcs = {
            'one_point': one_point_crossover
        }
        self._creatures: List[Creature] = []
        self._new_generation: List[Creature] = []
        self._best_creature = None

        self._defaults = {
            'num_rounds': 100,
            'num_creatures': 100,
            'preserve_num': 10,
            'mutation_chance': 0.1,
            'mutable_params': None,
            'param_space': None,
            'objective_func': None,
            'random_mutate': False

        }
        self._stats = {
            'creatures_tested': 0,
            'total_time': 0,
            'mutations': 0
        }
        self._score_dict: dict = {} # we cache creatures by score, as it is the output of objective function

        self._defaults['selection_func'] = next(iter(self._selection_funcs.items()))[1]
        self._defaults['crossover_func'] = next(iter(self._crossover_funcs.items()))[1]
        self._attrs = dict(self._defaults)

        try:
            self._attrs.update(zip(self._attrs, args))
            self._attrs.update(kwargs)
        except KeyError as e:
            print("KeyError:", e)
            exit()

        if not self._attrs['objective_func']:
            raise ValueError("Objective function is required.")

        if self._attrs['mutation_chance'] <= 0 or self._attrs['mutation_chance'] > 1:
            raise ValueError("Invalid value for mutation_chance. Must be in range 0-1.")

        if not isinstance(self._attrs['mutation_chance'], float):
            raise ValueError("Invalid type for mutation_chance. Must be of type float.")

        if not isinstance(self._attrs['num_rounds'], int):
            raise ValueError("Invalid type for num_rounds. Must be of type int.")

        if not isinstance(self._attrs['num_creatures'], int):
            raise ValueError("Invalid type for num_creatures. Must be of type int.")

        if not isinstance(self._attrs['preserve_num'], int):
            raise ValueError("Invalid type for preserve_num. Must be of type int.")

        if not isinstance(self._attrs['selection_func'], Callable):
            raise ValueError("Invalid type for selection_func. Must be a function.")

        if not isinstance(self._attrs['crossover_func'], Callable):
            raise ValueError("Invalid type for crossover_func. Must be a function.")

        if not isinstance(self._attrs['mutable_params'], list) and self._attrs['mutable_params'] is not None:
            raise ValueError("Invalid type for mutable_params. Must be a list or None.")

        if not isinstance(self._attrs['param_space'], dict) and self._attrs['mutable_params'] is not None:
            raise ValueError("Invalid type for mutable_params. Must be a dict or None.")

    def learn(self, data) -> Creature | None:
        # jump start population
        if len(self._creatures) == 0:
            self._populate(data)

        # train for set number of rounds
        for round_num in range(self._attrs['num_rounds']):
            tic = time.time()
            print(f'Training generation {round_num}')

            # train and sort by fitness
            self._population_eval()

            # add creatures to fitness dict
            for creature in self._creatures:
                self._score_dict[creature.get_params_hash()] = creature.score

            # print best at the beginning of training
            if self._best_creature is None:
                print(f"Best: {str(self._creatures[0])}\n")
                self._best_creature = self._creatures[0]
            # print update about new best only if it has changed
            elif self._best_creature.params != self._creatures[0].params:
                print(f"Found new best: {str(self._creatures[0])}")
                self._best_creature = self._creatures[0]

            # update the population
            self._create_new_generation(data)

            # print training time for this generation
            toc = time.time()
            print(f'Time taken: {format_time_difference(tic, toc)}')
            print()
        return self._best_creature

    def _population_eval(self) -> None:
        # get score for each creature and sort according to fitness (models are trained jit)
        self._creatures.sort(key=lambda x: x.fitness, reverse=True)

    def _population_eval_mt(self):  # doesnt work for now
        cpu_count = multiprocessing.cpu_count()
        print(f"Cpu count: {cpu_count}")
        if cpu_count > 16:
            cpu_count = 16
            print('Limiting cpu count to 16')
        pool = multiprocessing.Pool(processes=cpu_count)
        pool.map(get_fitness_wrapper, self._creatures)
        pool.close()
        pool.join()
        self._creatures.sort(key=lambda x: x.fitness, reverse=True)

    def _populate(self, data) -> None:
        # fill population with randomly generated creatures
        self._creatures = [
            Creature(self._attrs['param_space'], self._attrs['mutable_params'], self._attrs['objective_func'], data) for _ in range(self._attrs['num_creatures'])
        ]

    def _create_new_generation_old(self, data) -> None:
        # choose set number of best creatures according to selection function
        self._new_generation.extend(self._attrs['selection_func'](self._creatures, self._attrs['preserve_num']))
        # shuffle for random breeding
        random.shuffle(self._new_generation)
        # make new creatures
        for i in range(0, len(self._new_generation), 2):
            self._new_generation.extend(self._attrs['crossover_func'](*self._new_generation[i:i+2]))
        # fill the remaining space in the population with random creatures
        for _ in range(self._attrs['num_creatures'] - len(self._new_generation)):
            new_creature = self._new_generation[0].get_copy()
            new_creature.generate()
            self._new_generation.append(new_creature)
        # set new population as base, reset new
        self._creatures, self._new_generation = self._new_generation, []

    def _create_new_generation(self, data) -> None:
        # choose set number of best creatures according to selection function
        parents = self._attrs['selection_func'](self._creatures, self._attrs['preserve_num'])
        # shuffle for random breeding
        random.shuffle(parents)
        # preserve best creature
        self._new_generation.append(self._creatures[0])
        # create mutated copy
        mutated_best = self._creatures[0].get_copy()
        # attempt to mutate a creature until it is unique and has not been tested before
        for mutation_attempt in range(100):
            mutated_best.mutate()
            if not(mutated_best.params == self._creatures[0].params or mutated_best.get_params_hash() in self._score_dict):
                break
        self._new_generation.append(mutated_best)

        # make new creatures
        for i in range(0, len(parents), 2):
            children: List[Creature] = self._attrs['crossover_func'](*self._creatures[i:i+2])
            # mutate children
            for child in children:
                child.mutate(rand_param=self._attrs['random_mutate'])
                # if a child happens to have been tested before, reasign the score to avoid retraining
                if child.get_params_hash() in self._score_dict:
                    child.score = self._score_dict[child.get_params_hash()]
            self._new_generation.extend(children)

        # fill the remaining space in the population with random creatures
        for _ in range(self._attrs['num_creatures'] - len(self._new_generation)):
            # create a random creature by regenerating a copy of existing one
            new_creature = self._new_generation[0].get_copy()
            new_creature.generate()
            # if a creature happens to have been tested before, reasign the score to avoid retraining
            if new_creature.get_params_hash() in self._score_dict:
                new_creature.score = self._score_dict[new_creature.get_params_hash()]
            self._new_generation.append(new_creature)
        # set new population as base, reset new
        self._creatures, self._new_generation = self._new_generation, []