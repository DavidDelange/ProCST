import os

def generate_data_list(root_path, dataset, set_name):
    # Define the dataset directory
    dataset_dir = f"./dataset/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)  # Ensure the full directory is created

    # Define the output file
    output_file = f"{dataset_dir}/{set_name}.txt"

    # Open the output file and write file names
    with open(output_file, "w") as file:
        for root, dirs, files in os.walk(f"{root_path}/{dataset}/{set_name}/images"):
            for filename in files:
                file.write(filename + "\n")    

    print(f"Data list for subset '{set_name}' generated successfully.")


generate_data_list('/home/ddel/workspace/data/seescans', 'real', "train")
generate_data_list('/home/ddel/workspace/data/seescans', 'synth', "train")