import numpy as np
from utils import compute_overlap


def compute_detection_statistics(masks, prediction_masks):
    for mask_id, masks_dict in enumerate(masks):
        for obj_label, mask in masks_dict.items():
            for key, prediction_mask in prediction_masks.items():
                if mask_id < len(prediction_mask):
                    pred_mask = prediction_mask[mask_id]
                    overlap = compute_overlap(mask, pred_mask)
                    if overlap > 0 and obj_label == key:
                        mask['matches'].get('matches', [])
                        mask['matches'].append({'object_label': key, 'overlap': overlap})
                    elif overlap > 0 and obj_label != key:
                        mask['mismatches'].get('mismatches', [])
                        mask['mismatches'].append({'object_label': key, 'overlap': overlap})

        # Scan mask for unmatched IDs
        masks_dict['unmatched'].get('unmatched', [])

    inference = {}
    for masks_dict in masks:
        for obj_label, mask in masks_dict.items():
            if obj_label not in inference:
                inference[obj_label] = {'misdetections': 0, 'detections': 0, 'false_alarms': 0, 'ok': 0, 'processed': 0}

                if len(mask['matches']) or len(mask['mismatches']):
                    inference[obj_label]['detections'] += 1
                    inference[obj_label]['processed'] += 1
                elif len(mask['matches']) == 0 and len(mask['mismatches'] == 0):
                    inference[obj_label]['misdetections'] += 1
                    inference[obj_label]['processed'] += 1

        # Process false alarms
        if len(masks_dict)['unmatched']:
            inference[-1]['false_alarms'] += 1

        if len(mask['matches']) == 0 and len(mask['mismatches'] == 0):
            inference[-1]['ok'] += 1
            inference[-1]['processed'] += 1
    return inference