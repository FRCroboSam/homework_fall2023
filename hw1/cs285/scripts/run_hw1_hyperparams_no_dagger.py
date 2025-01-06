import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# RUN WITH: python cs285/scripts/run_hw1_hyperparams_no_dagger.py

class RunHW1HyperparametersNoDagger:
    def __init__(self, expert_policy_file, env_name, exp_name, expert_data, video_log_freq, n_iter=10, episode_length=1000, batch_size=5000, dagger=False):
        self.expert_policy_file = expert_policy_file
        self.env_name = env_name
        self.exp_name = exp_name
        self.expert_data = expert_data
        self.video_log_freq = video_log_freq
        self.n_iter = n_iter
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.dagger = dagger

    def run_experiment(self, num_agent_train_steps_values):
        results = []

        for num_agent_train_steps in num_agent_train_steps_values:
            print(f"Running experiment with num_agent_train_steps_per_iter={num_agent_train_steps}")
            command = [
                "python", "cs285/scripts/run_hw1.py",
                "--expert_policy_file", self.expert_policy_file,
                "--env_name", self.env_name,
                "--exp_name", f"{self.exp_name}_steps_{num_agent_train_steps}",
                "--n_iter", str(self.n_iter),
                "--expert_data", self.expert_data,
                "--video_log_freq", str(self.video_log_freq),
                "--num_agent_train_steps_per_iter", str(num_agent_train_steps)
            ]

            # Run the command and capture the output
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"Error: {process.stderr}")
                continue

            # Parse the output to extract return and standard deviation
            output = process.stdout

            average_return = self._parse_output(output, "Eval_AverageReturn")
            std_return = self._parse_output(output, "Eval_StdReturn")
            

            results.append((num_agent_train_steps, average_return, std_return))
            print("RESULTS ARE")
            print(results)
        return results

    def _parse_output(self, output, key):
        # Extract the value of the given key from the output
        for line in output.splitlines():
            if key in line:
                try:
                    # Split on "key :", allowing for spaces around the colon
                    return float(line.split(f"{key} :")[-1].strip())
                except ValueError:
                    return None
        return None

    def plot_results(self, results, save_path="hopper_hyperparameter_exp_steps.png"):
        num_agent_train_steps = [r[0] for r in results]
        avg_returns = [r[1] for r in results]
        std_devs = [r[2] for r in results]

        plt.figure(figsize=(10, 6))
        plt.errorbar(num_agent_train_steps, avg_returns, yerr=std_devs, fmt='-o', label='Average Return')
        plt.title(f"Hyperparameter Experiment for {self.env_name} (n_iter={self.n_iter})")
        plt.xlabel("Num Agent Train Steps Per Iter")
        plt.ylabel("Average Return")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    # Define parameters
    expert_policy_file = "cs285/policies/experts/Hopper.pkl"
    env_name = "Hopper-v4"
    exp_name = "bc_hopper"
    expert_data = "cs285/expert_data/expert_data_Hopper-v4.pkl"
    video_log_freq = -1
    n_iter = 1  # Keeping n_iter constant

    # Create an instance of the class
    runner = RunHW1HyperparametersNoDagger(
        expert_policy_file, env_name, exp_name, expert_data, video_log_freq, n_iter
    )

    # Run experiments with different num_agent_train_steps_per_iter values
    num_agent_train_steps_values = [500, 1000, 2000, 3000, 4000, 5000]
    results = runner.run_experiment(num_agent_train_steps_values)

    # Display fixed parameters
    print(f"Episode Length: {runner.episode_length}")
    print(f"Batch Size: {runner.batch_size}")
    print(f"Dagger: {'Yes' if runner.dagger else 'No'}")

    # Plot and save results
    runner.plot_results(results)
