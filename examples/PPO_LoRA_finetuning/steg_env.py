import random
from typing import Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa
from datasets import load_dataset, DatasetDict


class SteganographyEnv:
    def __init__(self, config):
        self.config = config
        # self._action_space = [
        #     str(a) for a in config.rl_script_args["action_space"]
        # ]  # ideally config["action_space"] should be a list of integers
        self._load_dataset(config.rl_script_args['dataset_path'])
        self.current_context = None
        self.current_password = None
        self.ground_truth = None
        self.current_encoded_reply = None

    def _load_dataset(
        self, path: str = "jvelja/stegoData_hardmapped"
    ) -> None:
        # Return a list of dictionaries, each containing 'question' and 'answer' keys
        ds = load_dataset(path)
        try:
            self.dataset = ds["train"]
        except KeyError:
            self.dataset = ds["validation"]

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        # For now, resetting the environment means selecting a random datapoint
        # Ideally, reset should generate a brand new question and password, encoded by the speaker model
        # For now, I'll just select a random question and password from the dataset (no generation)

        # Select a random question and password
        datapoint = random.choice(self.dataset)
        self.current_context = datapoint["question"]
        self.current_password = random.choice([0,1])
        self.ground_truth_reply = datapoint["answer"]

        # Return the encoded answer as the observation, along with info
        info = {
            "context": self.current_context,
            "password": self.current_password,
            'ground_truth_reply': self.ground_truth_reply
        }

        prompt = f"Task: {self.current_context}\n Ground truth answer: {self.ground_truth_reply}\nPassword: {self.current_password}"

        return prompt, info

    def step(self, action: str, decoder, overseer = None) -> Tuple[str, float, bool, Dict[str, Any]]:
        # prepare decoder context
        decoder_prompt = raise NotImplementedError
        dec_action = decoder.generate(context=...)
        if overseer is not None:
            overseer_prompt = raise NotImplementedError
            overseer_action = overseer.generate(context=...)

        # Action is the listener's guess for the password
        reward = self._compute_reward(dec_action, overseer_action)
        done = True  # Episode ends after one step

        # For the next observation, we can either return None (since episode is done)
        # or return the same encoded answer if we want to allow multiple guesses
        next_observation = None
        self.current_encoded_reply = dec_action

        info = {
            "context": self.current_context,
            "encoded_reply": self.current_encoded_reply,
            "ground_truth_password": self.current_password,
            "possible_actions": self._action_space,
            'decoded_password': action
        }

        return next_observation, reward, done, info

    def _compute_reward(self, dec_action: str, overseer_action = None) -> float:
        # Implement your reward function here
        # For now, let's use a simple reward: 1 for correct guess, 0 otherwise
        return 1.0 if dec_action == self.current_password else 0.0

    def render(self, mode="human"):
        # Optionally implement this method to visualize the environment state
        print(f"Context: {self.current_context}")
        print(f"Ground truth reply: {self.ground_truth_reply}")
        print(f"True Password: {self.current_password}")

    def close(self):
        # Implement if you need to close any resources
        pass
