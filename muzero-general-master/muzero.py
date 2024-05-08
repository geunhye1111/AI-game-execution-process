import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import models
import self_play

class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result

@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary

import csv
import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search

def get_keywords_from_search_results(game_name, user_keyword):
    # 검색어 생성
    query = f"{game_name} {user_keyword}"
    print("실시간 크롤링 중 입니다. 시간이 조금 걸릴 수 있습니다.")

    # 구글 검색에서 게임에 대한 결과 페이지 URL 가져오기
    search_results = search(query, num=10, stop=5, pause=2) # 예를 들어, 상위 5개의 검색 결과만 사용
    keywords = []

    # 각 검색 결과 페이지에서 키워드 추출
    for url in search_results:
        try:
            # 웹 페이지에서 텍스트 가져오기
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            # 텍스트에서 단어 추출
            words = re.findall(r'\w+', text.lower())

            # 각 단어의 출현 빈도수 계산
            count = words.count(user_keyword.lower())
            keywords.append((game_name, count))
        except Exception as e:
            #print(f"Error occurred while processing URL: {url}")
            #print(e)
            continue

    return keywords

def recommend_game(games, user_keyword):
    game_counts = []
    for game in games:
        game_name = game.strip()
        keywords = get_keywords_from_search_results(game_name, user_keyword)
        total_count = sum(count for _, count in keywords)
        game_counts.append((game_name, total_count))

    # 빈도수가 가장 높은 게임 추천
    rgame = max(game_counts, key=lambda x: x[1])

    # 결과 출력
    print("게임별 키워드 언급 빈도수:")
    for game, count in game_counts:
        print(f"{game}: {count}")
    print(f"\n'{user_keyword}' 키워드를 가장 많이 언급한 게임: {rgame[0]} ({rgame[1]} mentions)")

    return rgame[0]

if __name__ == "__main__":
    # 게임 리스트
    games = ["커넥트4", "오목", "simple grid", "tic-tac-toe", "Blackjack"]
    #games = ["blackjack", "connect4", "gomoku", "simple_grid", "tictactoe"]

    # 사용자 입력 받기
    user_keyword = input("키워드를 입력하세요: ")

    gname = recommend_game(games, user_keyword)
    if(gname=="커넥트4") :
        game_name="connect4"
    elif(gname=="오목") :
        game_name="gomoku"
    elif(gname=="simple grid") :
        game_name="simple_grid"
    elif(gname=="tic-tac-toe") :
        game_name="tictactoe"
    elif(gname=="Blackjack") :
        game_name="blackjack"

    print("게임을 실행합니다.")

    muzero = MuZero(game_name)

    while True:
        # Configure running options
        options = [
            "Render some self play games",
            "Play against MuZero",
            "Test the game manually",
            "Exit",
        ]

        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        if choice == 0:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 1:
            muzero.test(render=True, opponent="human", muzero_player=0)
        elif choice == 2:
            env = muzero.Game()
            env.reset()
            env.render()

            done = False
            while not done:
                action = env.human_to_action()
                observation, reward, done = env.step(action)
                print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                env.render()
        else:
            break
        print("\nDone")

    ray.shutdown()