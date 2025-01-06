import os
import tensorflow as tf
import subprocess
import shutil

def find_experiment_folders(data_path, substrings):
    """
    Searches for subfolders in the data directory containing specified substrings.
    """
    matching_folders = []
    for root, dirs, files in os.walk(data_path):
        for folder in dirs:
            if any(substring in folder for substring in substrings):
                matching_folders.append(os.path.join(root, folder))
    return matching_folders

def delete_adjusted_folders(data_path):
    """
    Deletes all folders in the data directory that contain the substring 'adjusted'.
    """
    for root, dirs, _ in os.walk(data_path):
        for folder in dirs:
            if 'adjusted' in folder:
                folder_path = os.path.join(root, folder)
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)  # Remove the folder and all of its contents

def adjust_tensorboard_logs(folder):
    """
    Adjust TensorBoard logs to merge 'Train_EnvstepsSoFar' as the step and keep 'Eval_AverageReturn' on y-axis.
    """
    adjusted_log_files = []  # Keep track of adjusted log files
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.startswith("events.out.tfevents"):
                file_path = os.path.join(subdir, file)
                print(f"Processing log file: {file_path}")

                # Create a new directory for adjusted logs
                new_log_dir = f"{subdir}_adjusted"
                os.makedirs(new_log_dir, exist_ok=True)

                # Use TensorFlow's event file reader to load and adjust events
                writer = tf.summary.create_file_writer(new_log_dir)  # Initialize the writer

                # Initialize lists to hold steps and values
                steps = []  # Store Train_EnvstepsSoFar values
                eval_values = []  # Store Eval_AverageReturn values
                baseline_loss = []

                # Pass 1: Collect both Train_EnvstepsSoFar and Eval_AverageReturn for all steps
                for event in tf.compat.v1.train.summary_iterator(file_path):
                    for value in event.summary.value:
                        if value.tag == "Train_EnvstepsSoFar":
                            step = int(value.simple_value)
                            steps.append(step)  # Store the step
                        elif value.tag == "Eval_AverageReturn":
                            eval_value = value.simple_value
                            eval_values.append(eval_value)  # Store the Eval_AverageReturn value
                        elif value.tag == "Baseline_Loss":
                            eval_value = value.simple_value
                            baseline_loss.append(eval_value)  # Store the Eval_AverageReturn value

                # Pass 2: Log Eval_AverageReturn against Train_EnvstepsSoFar
                with writer.as_default():
                    for step, eval_value in zip(steps, eval_values):
                        tf.summary.experimental.set_step(step)
                        tf.summary.scalar("Eval_AverageReturn", eval_value)
                        print(f"Logging Eval_AverageReturn at step {step} with value {eval_value}")
                    for step, eval_value in zip(steps, baseline_loss):
                        tf.summary.experimental.set_step(step)
                        tf.summary.scalar("Baseline_Loss", eval_value)
                        print(f"Logging Baseline_Loss at step {step} with value {eval_value}")

                print(f"Adjusted logs saved to: {new_log_dir}")
                adjusted_log_files.append(new_log_dir)

    return adjusted_log_files  # Return a list of adjusted log folder paths

def main():
    
    data_path = "./data"  # Path to the data folder
    substrings = ["pendulum"]#["_cartpole_", "_cartpole_rtg_", "_cartpole_na_", "_cartpole_rtg_na_"]

    # Step 1: Delete any 'adjusted' folders before proceeding
    print("Cleaning up 'adjusted' folders...")
    delete_adjusted_folders(data_path)

    # Step 2: Find matching experiment folders
    experiment_folders = find_experiment_folders(data_path, substrings)

    if not experiment_folders:
        print("No matching experiment folders found.")
        return

    print("Found the following experiment folders:")
    for folder in experiment_folders:
        print(folder)

    # Step 3: Adjust TensorBoard logs for all matching folders
    adjusted_folders = []
    for folder in experiment_folders:
        adjusted_folder = adjust_tensorboard_logs(folder)
        adjusted_folders.extend(adjusted_folder)

    # Step 4: Start TensorBoard with the adjusted folders
    logdirs = ",".join([f"{os.path.basename(folder)}:{folder}" for folder in adjusted_folders])
    print("\nLaunching TensorBoard...")
    subprocess.run(["tensorboard", "--logdir_spec", logdirs])


if __name__ == "__main__":
    main()
