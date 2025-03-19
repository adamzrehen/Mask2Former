import pandas as pd
import os
import random
import tqdm


def train_test_split(csv_path, output_dir, ratio=0.8, mode='by_clip'):
    # Read the input CSV
    df = pd.read_csv(csv_path)
    summary = []

    # Initialize empty DataFrames for train and test sets
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    if mode == 'by_video':
        # Group by video_name: each video is a group
        groups = {video: group for video, group in df.groupby('video_name')}
        video_keys = list(groups.keys())
        random.shuffle(video_keys)
        split_index = int(ratio * len(video_keys))
        train_videos = video_keys[:split_index]
        test_videos = video_keys[split_index:]

        for video in train_videos:
            train_df = pd.concat([train_df, groups[video]], ignore_index=True)
        for video in test_videos:
            test_df = pd.concat([test_df, groups[video]], ignore_index=True)

        # Build summary information for each video
        for video in video_keys:
            summary.append({
                'Video': video,
                'Total Frames': len(groups[video]),
                'Train': video in train_videos,
                'Test': video in test_videos
            })

    elif mode == 'by_clip':
        # For each video, group by clip_id so that each clip is a group
        for video, video_group in df.groupby('video_name'):
            # Group by clip within this video
            clips = {clip: group for clip, group in video_group.groupby('clip_id')}
            clip_keys = list(clips.keys())
            # If not enough clips, skip (or you can choose to add all clips to one set)
            if len(clip_keys) < 2:
                continue
            random.shuffle(clip_keys)
            split_index = int(ratio * len(clip_keys))
            train_clips = clip_keys[:split_index]
            test_clips = clip_keys[split_index:]

            for clip in train_clips:
                train_df = pd.concat([train_df, clips[clip].assign(unique_name=f"{video}_{clip}")], ignore_index=True)
            for clip in test_clips:
                test_df = pd.concat([test_df, clips[clip].assign(unique_name=f"{video}_{clip}")], ignore_index=True)

            summary.append({
                'Video': video,
                'Total Clips': len(clip_keys),
                'Train Clips': len(train_clips),
                'Test Clips': len(test_clips),
                'Train Clip IDs': ", ".join(map(str, train_clips)),
                'Test Clip IDs': ", ".join(map(str, test_clips))
            })

    elif mode == 'within_clip':
        chunk_size = 20 # Define the size of each sequential chunk
        for (video, clip), group in df.groupby(['video_name', 'clip_id']):
            # Ensure the group is sorted by frame order
            group = group.sort_index()
            n_frames = len(group)

            # Create sequential chunks with a unique id for each chunk
            chunks = []
            num_chunks = 0
            for i in range(0, n_frames, chunk_size):
                num_chunks += 1
                chunk_df = group.iloc[i:i + chunk_size].copy()
                # Create a unique chunk name: video_clip_chunkNumber
                chunk_name = f"{video}_{clip}_{num_chunks}"
                # Add the unique chunk id to all rows in this chunk
                chunk_df['unique_name'] = chunk_name
                chunks.append(chunk_df)

            # Randomize the order of the chunks so train and test aren't sequential
            random.shuffle(chunks)

            # If there are fewer than 2 chunks, add them all to train
            if len(chunks) < 2:
                for chunk in chunks:
                    train_df = pd.concat([train_df, chunk], ignore_index=True)
                train_chunk_names = [chunk['unique_name'].iloc[0] for chunk in chunks]
                summary.append({
                    'Video': video,
                    'Clip': clip,
                    'Total Frames': n_frames,
                    'Total Chunks': len(chunks),
                    'Train Chunks': len(chunks),
                    'Test Chunks': 0,
                    'Train Chunk Names': train_chunk_names,
                    'Test Chunk Names': None,
                })
            else:
                # Split the randomized chunks into train and test using the provided ratio
                split_index = int(ratio * len(chunks))
                if split_index == 0:  # Ensure at least one chunk is in train if possible
                    split_index = 1
                train_chunks = chunks[:split_index]
                test_chunks = chunks[split_index:]

                for chunk in train_chunks:
                    train_df = pd.concat([train_df, chunk], ignore_index=True)
                for chunk in test_chunks:
                    test_df = pd.concat([test_df, chunk], ignore_index=True)

                train_chunk_names = [chunk['unique_name'].iloc[0] for chunk in train_chunks]
                test_chunk_names = [chunk['unique_name'].iloc[0] for chunk in test_chunks]

                summary.append({
                    'Video': video,
                    'Clip': clip,
                    'Total Frames': n_frames,
                    'Total Chunks': len(chunks),
                    'Train Chunks': len(train_chunks),
                    'Test Chunks': len(test_chunks),
                    'Train Chunk Names': train_chunk_names,
                    'Test Chunk Names': test_chunk_names,
                })

    else:
        raise ValueError("Invalid mode specified. Choose 'by_video', 'by_clip', or 'within_clip'.")

    # Ensure the output directory exists and save the train and test CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, 'train.csv')
    test_csv = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save summary to a CSV file
    summary_df = pd.DataFrame(summary)
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Train and test CSV files saved to: {output_dir}")


if __name__ == "__main__":
    csv_path = '/home/adam/Documents/Experiments/filtered_data.csv'
    output_dir = "/home/adam/Documents/Experiments/Mask2Former/split"
    train_test_split(csv_path, output_dir, ratio=0.8, mode='within_clip')
