import subprocess
import pandas as pd
import os
import tqdm
from pathlib import Path
from datetime import datetime
from demo import Evaluation


class Namespace:
    def __init__(self, **kwargs):
        # Initialize the class with any keyword arguments
        self.__dict__.update(kwargs)



def main(csv_file, output_path, config_file, base_dir, inference_output, save_visualizations,
         load_predictions):

    df = pd.read_csv(csv_file)
    grouped_videos = df.groupby(['video_name', 'clip_id'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_path, timestamp)
    os.mkdir(output_dir)
    evaluation_obj = None

    for group in tqdm.tqdm(grouped_videos):
        file_name = group[0][0] + '_clip_' + str(group[0][1])
        output_path = os.path.join(output_dir, file_name)
        input_path = [os.path.join(Path(group[1]['relative_image_path'].iloc[0]).parent, _)
                      for _ in os.listdir(Path(group[1]['relative_image_path'].iloc[0]).parent)]
        config = {
            'config_file': config_file,
            'input': input_path,
            'output': output_path if save_visualizations else '',
            'video_filename': file_name,
            'overlay_masks': True,
            'inference_output': inference_output,
            'opts': [],
            'load_predictions': load_predictions,
        }

        args = Namespace()
        for key, value in config.items():
            setattr(args, key, value)

        try:
            # Run the command
            if not evaluation_obj:
                evaluation_obj = Evaluation(args)
            evaluation_obj.args = args
            evaluation_obj.evaluate()
            print(f"demo.py executed successfully for {file_name}.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing demo.py for {file_name}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demo.py on grouped video data")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing video data")
    parser.add_argument("--output_path", required=True, help="Base directory for output results")
    parser.add_argument("--config_file", required=True, help="Path to the configuration file")
    parser.add_argument("--base_dir", required=True, help="Base directory for input video frames")
    parser.add_argument("--inference_output", required=False, default='',
                        help="Output directory for inference stats")
    parser.add_argument("--save_visualizations", required=False, default=False, help="Save visualizations")
    parser.add_argument("--load_predictions", default=False, help="Load saved predictions")
    args = parser.parse_args()

    main(args.csv_file, args.output_path, args.config_file, args.base_dir, args.inference_output, args.save_visualizations,
         args.load_predictions)


    # --csv_file="/home/adam/Documents/Experiments/Mask2Former/Test on different clip, same video January23_2025/train_split.csv"
    # --output_path="/home/adam/Documents/Experiments/Mask2Former/Test on different clip, same video January23_2025/results"
    # --config_file="/home/adam/PycharmProjects/Mask2Former/configs/ichilov/video_maskformer2_R50_bs16_8ep.yaml"
    # --base_dir="/home/adam/mnt/qnap/annotation_data/data/sam2"
