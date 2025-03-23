import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(inference_file, summary_file, output_dir):
    df = pd.read_csv(inference_file)
    total_detections = df['Detections'].sum()
    total_misdetections = df['Misdetections'].sum()
    total_false_alarms = df['False Alarms'].sum()
    total_ok_images = df['OK Images'].sum()

    overall_det_rate = total_detections / (total_detections + total_misdetections) * 100
    overall_fa_rate = total_false_alarms / (total_false_alarms + total_ok_images) * 100

    print(f'Detection Rate: {overall_det_rate:0.2f} %')
    print(f'False Alarm Rate: {overall_fa_rate:0.2f} %')

    grouped = df.groupby('Video Name')
    results_per_video = {'Video': [], 'Detection Rate': [], 'False Alarm Rate': []}
    for video_name, sub_df in grouped:
        detections = sub_df['Detections'].sum()
        misdetections = sub_df['Misdetections'].sum()
        false_alarms = sub_df['False Alarms'].sum()
        ok_images = sub_df['OK Images'].sum()

        det_rate = detections / (detections + misdetections) * 100
        fa_rate = false_alarms / (false_alarms + ok_images) * 100
        results_per_video['Video'].append(video_name)
        results_per_video['Detection Rate'].append(det_rate)
        results_per_video['False Alarm Rate'].append(fa_rate)

    summary_per_video = pd.DataFrame(results_per_video)

    # Create the figure with subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Format the values to 2 decimal places
    summary_per_video['Detection Rate'] = summary_per_video['Detection Rate'].apply(
        lambda x: "-" if pd.isna(x) else f"{x:.2f}"
    )
    summary_per_video['False Alarm Rate'] = summary_per_video['False Alarm Rate'].apply(
        lambda x: "-" if pd.isna(x) else f"{x:.2f}"
    )

    # Prepare the table data
    table_data = summary_per_video[['Video', 'Detection Rate', 'False Alarm Rate']].values
    column_labels = ['Video', 'Detection Rate (%)', 'False Alarm Rate (%)']

    # Create the table and style it
    table = axes[0].table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')

    # Style the table to make it nicer
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width([0, 1, 2])  # Automatically adjust column width for better readability
    table.scale(1.2, 1.5)  # Increase the size of the table for better visibility

    # Set the table color
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f1f1f1')  # Light gray for header
        else:
            cell.set_fontsize(12)
            cell.set_text_props(weight='normal')
            cell.set_facecolor('#ffffff')  # White for data rows

    axes[0].axis('off')  # Hide the axes so only the table is visible

    # Plot for overall summary
    summary_data = {'Metric': ['Detection Rate', 'False Alarm Rate'], 'Value': [overall_det_rate, overall_fa_rate]}
    summary_df = pd.DataFrame(summary_data)

    sns.barplot(x='Metric', y='Value', data=summary_df, hue='Metric', ax=axes[1], legend=False)

    # Set y-axis to range 0-100 for the overall plot
    axes[1].set_ylim(0, 100)

    axes[1].set_title('Overall Detection and False Alarm Rates')
    axes[1].set_ylabel('Rate (%)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot as a single image
    output_file = os.path.join(output_dir, 'evaluation.png')
    plt.savefig(output_file)


def get_args():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument("--inference_file", help="A csv with inference results")
    parser.add_argument("--summary_file", help="A csv with summary of train/test split")
    parser.add_argument("--output_dir", help="A directory in which to save the evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    evaluate(args.inference_file, args.summary_file, args.output_dir)