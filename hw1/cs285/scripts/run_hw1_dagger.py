import os
import subprocess
import matplotlib.pyplot as plt

class RunDAggerExperiment:
    def __init__(self, expert_policy_file, env_name, exp_name, expert_data, video_log_freq, batch_size=5000):
        self.expert_policy_file = expert_policy_file
        self.env_name = env_name
        self.exp_name = exp_name
        self.expert_data = expert_data
        self.video_log_freq = video_log_freq
        self.batch_size = batch_size

    def run_dagger_experiment(self, n_iter):
        print(f"Running DAgger experiment with n_iter={n_iter}")
        command = [
            "python", "cs285/scripts/run_hw1.py",
            "--expert_policy_file", self.expert_policy_file,
            "--env_name", self.env_name,
            "--exp_name", f"{self.exp_name}_dagger_iter_{n_iter}",
            "--n_iter", str(n_iter),
            "--expert_data", self.expert_data,
            "--video_log_freq", str(self.video_log_freq),
            "--do_dagger"
        ]

        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Error: {process.stderr}")
            return []

        output = process.stdout
        return self._parse_all_returns(output, "Eval_AverageReturn")

    def _parse_all_returns(self, output, key):
        returns = []
        for line in output.splitlines():
            if key in line:
                try:
                    returns.append(float(line.split(f"{key} :")[-1].strip()))
                except ValueError:
                    continue
        return returns

    def plot_results(self, average_returns, expert_policy_return, bc_return, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(average_returns) + 1), average_returns, '-o', label='DAgger Policy')
        plt.axhline(y=expert_policy_return, color='r', linestyle='--', label='Expert Policy')
        plt.axhline(y=bc_return, color='g', linestyle='--', label='Behavioral Cloning')

        plt.title("DAgger Learning Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Eval Average Return")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    # Define parameters
    expert_policy_file = "cs285/policies/experts/Ant.pkl"
    env_name = "Ant-v4"
    exp_name = "ant_dagger"
    expert_data = "cs285/expert_data/expert_data_Ant-v4.pkl"
    video_log_freq = -1
    n_iter = 10  # Fixed to 10 iterations

    # Define expert and behavioral cloning performance constants
    expert_policy_return = 4680  # Replace with actual expert policy performance
    bc_return = 1276  # Replace with actual behavioral cloning performance

    runner = RunDAggerExperiment(expert_policy_file, env_name, exp_name, expert_data, video_log_freq)

    # Run DAgger experiment
    average_returns = runner.run_dagger_experiment(n_iter)

    # Plot results
    runner.plot_results(average_returns, expert_policy_return, bc_return, "dagger_learning_curve.png")
