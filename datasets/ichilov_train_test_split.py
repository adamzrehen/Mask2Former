import pandas as pd
import os
import random


def train_test_split(csv_path, output_dir):
    # Read the input CSV
    df = pd.read_csv(csv_path)

    # Group videos
    grouped_videos = df.groupby(['Video'])
    summary = []

    # Initialize empty DataFrames for train and test sets
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for video, video_group in grouped_videos:
        # Group clips within each video
        grouped_clips = video_group.groupby(['Clip ID'])

        # Shuffle the group keys to ensure randomness
        grouped_clip_keys = list(grouped_clips.groups.keys())
        random.shuffle(grouped_clip_keys)

        # Calculate the split index
        split_index = int(0.8 * len(grouped_clip_keys))

        # Split into train and test groups
        train_clip_keys = grouped_clip_keys[:split_index]
        test_clip_keys = grouped_clip_keys[split_index:]

        # Add train and test groups to their respective DataFrames
        if len(train_clip_keys):
            train_df = pd.concat([train_df, pd.concat([grouped_clips.get_group(key) for key in train_clip_keys])])
        if len(test_clip_keys):
            test_df = pd.concat([test_df, pd.concat([grouped_clips.get_group(key) for key in test_clip_keys])])

        summary.append({
            'Video': video,
            'Train Clips': len(train_clip_keys),
            'Test Clips': len(test_clip_keys),
            'Total Clips': len(grouped_clip_keys),
            'Train Clip IDs': ", ".join([str(_) for _ in train_clip_keys]),
            'Test Clip IDs': ", ".join([str(_) for _ in test_clip_keys])
        })

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save train and test DataFrames to CSV files
    train_csv_path = os.path.join(output_dir, 'train_split.csv')
    test_csv_path = os.path.join(output_dir, 'test_split.csv')
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    # Save summary to a CSV file
    summary_df = pd.DataFrame(summary)
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Train and test CSV files saved to: {output_dir}")


if __name__ == "__main__":
    csv_path = '/home/adam/Documents/Experiments/Mask2Former/train.csv'
    output_dir = "/home/adam/Documents/Experiments/Mask2Former/split"
    train_test_split(csv_path, output_dir)