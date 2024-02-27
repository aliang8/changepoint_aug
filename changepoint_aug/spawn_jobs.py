import os
import subprocess
from itertools import product

if __name__ == "__main__":
    # Define the hyperparameters and their possible values
    # hyperparameters = {"seed": [0, 1], "num_demos": [25, 50, 75, 100]}
    hyperparameters = {"seed": [0, 1, 2], "num_demos": [50]}

    # List to store the command line calls
    commands = []

    # Generate all combinations of hyperparameter values
    combinations = list(product(*hyperparameters.values()))

    # For each combination, create a command to call the Python script
    for combo in combinations:
        command = [
            "python3",
            "bc.py",
            "--env_name",
            "metaworld-assembly-v2",
            "--n_eval_episodes",
            "25",
            "--num_bc_epochs",
            "800",
            "--dataset_file",
            "datasets/expert_dataset/image_True/assembly-v2_100_noise_0.2",
            "--image_based",
            "True",
            "--log_interval",
            "500",
            "--noise_std",
            "0.2",
        ]  # Replace 'your_script.py' with the actual script name

        # Add hyperparameter arguments to the command
        for param, value in zip(hyperparameters.keys(), combo):
            command.extend(["--" + param, str(value)])

        print(command)
        # Append the command to the list
        commands.append(command)

    processes = []
    # Perform hyperparameter search by executing the commands
    for idx, command in enumerate(commands, 1):
        print(f"Running experiment {idx}/{len(commands)}:")
        print(" ".join(command))

        # Uncomment the line below to execute the command
        # subprocess.run(command, check=True)
        # Start the subprocess in the background using subprocess.Popen
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = "5"
        process = subprocess.Popen(command, env=my_env)
        processes.append(process)
        print("Experiment completed.\n")

    for process in processes:
        process.wait()
    print("Hyperparameter search completed.")
