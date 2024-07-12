"""
PPO implementation taken from https://github.com/openai/spinningup
"""

import hydra
import torch
import numpy as np
import time
import pickle
import os
import functools as f
import wandb


from tqdm import tqdm
from operator import add
from transformers import set_seed
from huggingface_hub import login
from peft_modules import (
    SequentialInitializer,
    WeightsLoaderInitializer,
    PeftInitializer,
)
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
from steg_env import SteganographyEnv
from ppo_updater import PPOUpdater
from lamorel import Caller, lamorel_init

lamorel_init()
login("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")


SYSTEM_PROMPT = "..."


def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "loss": [],
        "policy_loss": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
    }


@hydra.main(config_path="config", config_name="config")
def main(config_args):
    wandb.init(project="ppo-llm", config=config_args)
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # Instantiate environment: SteganographyEnv --> When called, it returns a prompt and info dict
    envs = SteganographyEnv(config_args)

    # Create LLM encoder
    encoder = Caller(  # Caller is a class that wraps the LLM model and provides a simple interface to interact with it
        config_args.lamorel_args,
        custom_updater=PPOUpdater(
            config_args.lamorel_args.llm_args.encoder.model_type,
            config_args.rl_script_args.minibatch_size,
            config_args.rl_script_args.gradient_batch_size,
        ),
        custom_model_initializer=SequentialInitializer(
            [
                PeftInitializer(
                    config_args.lamorel_args.llm_args.encoder.model_type,
                    config_args.lamorel_args.llm_args.encoder.model_path,
                    config_args.rl_script_args.use_lora,
                    config_args.lamorel_args.llm_args.encoder.load_in_4bit,
                    config_args.rl_script_args.lora_r,
                    config_args.rl_script_args.lora_alpha,
                    config_args.lamorel_args.llm_args.encoder.pre_encode_inputs,
                ),
                WeightsLoaderInitializer(config_args.rl_script_args.loading_path),
            ]
        ),
    )

    # Create Classifier Decoder
    decoder = Caller(...)

    if config_args.llm_args.overseer:
        overseer = Caller(...)
    else:
        overseer = None


    # Set up experience buffer
    buffers = [  # Will need adaptation to the new environment/setting (i.e., SteganographyEnv)
        PPOBuffer(
            config_args.rl_script_args.steps_per_epoch
            // config_args.rl_script_args.number_envs,
            config_args.rl_script_args.gamma,
            config_args.rl_script_args.lam,
        )
        for _ in range(config_args.rl_script_args.number_envs)
    ]

    # Prepare for interaction with environment --> Here is where we will need to make changes
    # o = encoded_answer, infos = dict{question, password}, ep_ret = 0, ep_len = 0 (assuming a single environment for now)

    (o, infos), ep_ret, ep_len = (
        envs.reset(),
        [0 for _ in range(config_args.rl_script_args.number_envs)],
        [0 for _ in range(config_args.rl_script_args.number_envs)],
    )

    history = reset_history()

    for epoch in range(config_args.rl_script_args.epochs):
        __time = time.time()
        for t in tqdm(
            range(
                config_args.rl_script_args.steps_per_epoch
                // config_args.rl_script_args.number_envs
            ),
            ascii=" " * 9 + ">",
            ncols=100,
        ):
            prompt, info = envs.reset()
            # Prepare inputs
            chat_template_encoder = [
                f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ]

            output = encoder.generate(contexts=chat_template_encoder)

            print("output: ", output)
            _, r, d, infos = envs.step(
                action = output,
                decoder = decoder,
                overseer = overseer
            )

            # _, r, d, infos = envs.step(output, decoder)
            epoch_ended = (
                (t + 1) * config_args.rl_script_args.number_envs
                == config_args.rl_script_args.steps_per_epoch
            )
            bootstrap_dict = {"ids": [], "contexts": []}
            for i in range(config_args.rl_script_args.number_envs):
                buffers[i].store(
                    prompts[i],
                    possible_actions[i],
                    actions_id[i],
                    r[i],
                    values[i],
                    log_probs[i],
                )
                ep_ret[i] += r[i]
                ep_len[i] += 1
                timeout = ep_len[i] == config_args.rl_script_args.max_ep_len
                terminal = d[i] or timeout
                if terminal or epoch_ended:
                    if not terminal:
                        bootstrap_dict["ids"].append(i)
                        bootstrap_dict["contexts"].append(
                            generate_prompt(o[i], infos[i])
                        )
                    else:
                        buffers[i].finish_path(0)
                        history["ep_len"].append(ep_len[i])
                        history["ep_ret"].append(ep_ret[i])
                        ep_len[i], ep_ret[i] = 0, 0
                        history["goal"].append(infos[i]["goal"])

            if len(bootstrap_dict["ids"]) > 0:
                # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                output = lm_server.custom_module_fns(
                    module_function_keys=["value"],
                    contexts=bootstrap_dict["contexts"],
                    candidates=[[""] for _ in range(len(bootstrap_dict["contexts"]))],
                )
                for _i in range(len(output)):
                    buffers[bootstrap_dict["ids"][_i]].finish_path(
                        output[_i]["value"][0]
                    )

        # Perform PPO update!
        print(f"PPO update number {epoch + 1}")
        save_model_and_history = (
            epoch % config_args.rl_script_args.save_freq == 0
            or epoch == config_args.rl_script_args.epochs - 1
        ) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq
        saving_path = (
            f"{config_args.rl_script_args.output_dir}/epochs_{start_epoch}-{epoch}"
        )
        if save_model_and_history:
            os.makedirs(saving_path, exist_ok=True)
        loading_path = (
            config_args.rl_script_args.loading_path
            if config_args.rl_script_args.loading_path is not None
            else ""
        )

        # Stack trajectories for all envs
        # TODO: Randomize and mix up environments' trajectories
        trajectories = [buf.get() for buf in buffers]
        collected_trajectories = {
            k: torch.cat([traj[k] for traj in trajectories])
            if isinstance(trajectories[0][k], torch.Tensor)
            else list(f.reduce(add, [traj[k] for traj in trajectories]))
            for k, _ in trajectories[0].items()
        }

        update_results = lm_server.update(
            collected_trajectories["obs"],
            collected_trajectories["possible_act"],
            actions=collected_trajectories["act"],
            returns=collected_trajectories["ret"],
            advantages=collected_trajectories["adv"],
            logprobs=collected_trajectories["logp"],
            values=collected_trajectories["val"],
            lr=config_args.rl_script_args.lr,
            clip_eps=config_args.rl_script_args.clip_eps,
            entropy_coef=config_args.rl_script_args.entropy_coef,
            value_loss_coef=config_args.rl_script_args.value_loss_coef,
            max_grad_norm=config_args.rl_script_args.max_grad_norm,
            ppo_epochs=config_args.rl_script_args.ppo_epochs,
            save_after_update=save_model_and_history,
            output_dir=saving_path,
            loading_path=loading_path,
        )

        avg_loss = np.mean([_r["loss"] for _r in update_results])
        avg_policy_loss = np.mean([_r["policy_loss"] for _r in update_results])
        avg_value_loss = np.mean([_r["value_loss"] for _r in update_results])
        history["loss"].append(avg_loss)
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)
        history["possible_actions"].extend(collected_trajectories["possible_act"])
        history["actions"].extend(
            [
                _poss_act[int(_a.item())]
                for _poss_act, _a in zip(
                    collected_trajectories["possible_act"],
                    collected_trajectories["act"],
                )
            ]
        )
        history["prompts"].extend(collected_trajectories["obs"])
        print(f"Update loss: {avg_loss}")

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "mean_episode_length": np.mean(history["ep_len"]),
                "mean_episode_return": np.mean(history["ep_ret"]),
            }
        )

        if save_model_and_history:
            # Save history
            with open(f"{saving_path}/history.pkl", "wb") as file:
                pickle.dump(history, file)
            history = reset_history()

    start_epoch = epoch - config_args.rl_script_args.save_freq
    saving_path = (
        f"{config_args.rl_script_args.output_dir}/epochs_{start_epoch}-{epoch}"
    )
    os.makedirs(saving_path, exist_ok=True)
    with open(f"{saving_path}/history.pkl", "wb") as file:
        pickle.dump(history, file)

    wandb.finish()


if __name__ == "__main__":
    main()
