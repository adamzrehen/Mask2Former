import os

# Paths to the image and annotation directories
image_dir = "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid/JPEGImages"
annotation_dir = "/home/adam/Documents/Data/KUMC Dataset/ytvis_format/valid/Annotations"

# Iterate through annotation sequences
for annotation_seq in os.listdir(annotation_dir):
    annotation_seq_path = os.path.join(annotation_dir, annotation_seq)
    image_seq_path = os.path.join(image_dir, annotation_seq)

    for img in os.listdir(annotation_seq_path):
        img_ = img.split('.xml')[0]
        if not os.path.exists(os.path.join(image_seq_path, img_ + '.jpg')):
            # If no corresponding image sequence exists, delete the annotation sequence
            os.remove(os.path.join(annotation_seq_path, img_ + '.xml'))