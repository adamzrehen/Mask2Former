import subprocess
import pandas as pd
import os
import tqdm
from pathlib import Path
from datetime import datetime


def main(csv_file, output_path, config_file, base_dir):

    df = pd.read_csv(csv_file)
    grouped_videos = df.groupby(['Video', 'Clip ID'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_path, timestamp)
    os.mkdir(output_dir)

    for group in tqdm.tqdm(grouped_videos):
        file_name = group[0][0] + '_clip_' + str(group[0][1])
        output_path = os.path.join(output_dir, file_name)
        input_path = os.path.join(base_dir, Path(group[1]['Frame Path'].iloc[0]).parent,
                                  '*' + Path(group[1]['Frame Path'].iloc[0]).suffix)

        # Command to execute demo.py
        command = [
            "python", "demo.py",
            f"--config-file={config_file}",
            f"--input={input_path}",
            f"--output={output_path}"
        ]

        try:
            # Run the command
            subprocess.run(command, check=True)
            print("demo.py executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing demo.py: {e}")

if __name__ == "__main__":
    csv_file = '/home/adam/Documents/Experiments/Mask2Former/Test on different clip, same video January23_2025/train_split.csv'
    output_path = "/home/adam/PycharmProjects/Mask2Former/results"
    config_file = "/home/adam/PycharmProjects/Mask2Former/configs/ichilov/video_maskformer2_R50_bs16_8ep.yaml"
    base_dir = "/home/adam/mnt/qnap/annotation_data/data/sam2"

    main(csv_file, output_path, config_file, base_dir)
