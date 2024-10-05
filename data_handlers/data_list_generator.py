import os

def generate_data_list(root_path, dataset, set_name):
    # Define the dataset directory
    dataset_dir = f"/user_volume_david_delange/ProCST/dataset/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)  # Ensure the full directory is created

    # Define the output file
    output_file = f"{dataset_dir}/{set_name}.txt"

    # Open the output file and write file names
    with open(output_file, "w") as file:
        for root, dirs, files in os.walk(f"{root_path}/{dataset}/{set_name}/samples"):
            for filename in files:
                file.write(filename + "\n")    

    print(f"Data list for subset '{set_name}' generated successfully.")


#Crea una funcion que lea el archivo .txt dada una ruta y cree a partir de ella dos nuevas: una con el 80% de los datos y otra con el 20% restante que se llame train_semseg_net.txt y val_semseg_net.txt respectivamente
def split_data_file(path, set_name):
    # Leer el archivo original
    with open(f"{path}/{set_name}", 'r') as file:
        lines = file.readlines()
    
    # Calcular el punto de corte para el 80% de los datos
    split_point = int(len(lines) * 0.8)
    
    # Escribir el 80% de los datos en train_semseg_net.txt
    with open(f"{path}/train_semseg_net.txt", 'w') as train_file:
        train_file.writelines(lines[:split_point])
    
    # Escribir el 20% restante en val_semseg_net.txt
    with open(f"{path}/val_semseg_net.txt", 'w') as val_file:
        val_file.writelines(lines[split_point:])

if __name__ == "__main__":
    #generate_data_list('/user_volume_david_delange/data/procst', 'real', "val")
    #generate_data_list('/user_volume_david_delange/data/procst', 'synth', "train")
    split_data_file("/user_volume_david_delange/ProCST/dataset/synth", 'train.txt')