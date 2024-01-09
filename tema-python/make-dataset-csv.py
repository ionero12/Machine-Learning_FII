import os
import re
import pandas as pd


def read_partitioned_dataset(root_folder):
    data = {'subject': [], 'message': [], 'label': [], 'folder': []}

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    for file_name in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file_name)
                        with open(file_path, 'r', encoding='latin1') as file:
                            content = file.read()
                            match = re.search(r"Subject: (.+?)\n\n(.+)", content, re.DOTALL)
                            if match:
                                subject, message = match.group(1), match.group(2)
                                label = 1 if 'spm' in file_name else 0
                                folder_label = f"{folder_name}-{subfolder_name}"
                                data['subject'].append(subject)
                                data['message'].append(message)
                                data['label'].append(label)
                                data['folder'].append(folder_label)

    return data


def create_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    root_folder = '../lingspam_public'
    output_file = 'lingspam_dataset.csv'

    dataset = read_partitioned_dataset(root_folder)
    create_csv(dataset, output_file)
