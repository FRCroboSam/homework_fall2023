import os
import subprocess
import shutil

from tensorboard.backend.event_processing import event_accumulator

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

                # Load the event file with the EventAccumulator
                accumulator = event_accumulator.EventAccumulator(file_path)
                accumulator.Reload()  # Load the data

                # Extract the 'Train_EnvstepsSoFar' and 'Eval_AverageReturn' events
                steps = accumulator.Scalars("Train_EnvstepsSoFar")
                eval_values = accumulator.Scalars("Eval_AverageReturn")
                baseline_loss = accumulator.Scalars("Baseline_Loss")

                # Write adjusted logs (in this case, just the 'Eval_AverageReturn' and 'Baseline_Loss')
                with open(os.path.join(new_log_dir, "adjusted_logs.txt"), 'w') as log_file:
                    for step, eval_value in zip(steps, eval_values):
                        log_file.write(f"Step: {step[0]}, Eval_AverageReturn: {eval_value[1]}\n")
                        print(f"Logging Eval_AverageReturn at step {step[0]} with value {eval_value[1]}")
                    for step, loss_value in zip(steps, baseline_loss):
                        log_file.write(f"Step: {step[0]}, Baseline_Loss: {loss_value[1]}\n")
                        print(f"Logging Baseline_Loss at step {step[0]} with value {loss_value[1]}")

                print(f"Adjusted logs saved to: {new_log_dir}")
                adjusted_log_files.append(new_log_dir)

    return adjusted_log_files  # Return a list of adjusted log folder paths

def main():
    data_path = "./data"  # Path to the data folder
    substrings = ["Ant", "Lunar", "Humanoid", "opper"]  # Adjust substrings as needed

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

    # # Step 3: Adjust TensorBoard logs for all matching folders
    # adjusted_folders = []
    # for folder in experiment_folders:
    #     adjusted_folder = adjust_tensorboard_logs(folder)
    #     adjusted_folders.extend(adjusted_folder)

    # Step 4: Start TensorBoard with the adjusted folders
    logdirs = ",".join([f"{os.path.basename(folder)}:{folder}" for folder in experiment_folders])
    print("\nLaunching TensorBoard...")
    subprocess.run(["tensorboard", "--logdir_spec", logdirs])


if __name__ == "__main__":
    main()
