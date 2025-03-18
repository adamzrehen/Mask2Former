from utils import compute_overlap


def compute_detection_statistics(masks, prediction_masks, min_overlap=50):
    # Initialize a new list to store matches information
    matches_info = []

    for mask_id, masks_dict in enumerate(masks):
        matches_data = {'matches_per_obj': {}, 'unmatched': {}}  # New dictionary to hold match-related info for this mask_id
        unmatched = set(prediction_masks.keys())  # Start with all keys as unmatched

        for obj_label, mask in masks_dict.items():
            # Create a separate dictionary for storing matches and mismatches
            match_info = {
                'matches': [],
                'mismatches': [],
                'misdetections': [],
            }

            matched_prediction = False
            for key, prediction_mask in prediction_masks.items():
                if mask_id >= len(prediction_mask):
                    unmatched.discard(key)
                    continue  # Skip if there's no mask at this index

                pred_mask = prediction_mask[mask_id]
                overlap = compute_overlap(mask, pred_mask)

                if overlap > min_overlap:
                    # If object labels match, add to 'matches'; otherwise, add to 'mismatches'
                    match_list = match_info['matches'] if obj_label == key else match_info['mismatches']
                    match_list.append({'object_label': key, 'overlap': overlap})
                    unmatched.discard(key)  # Remove from unmatched
                    matched_prediction = True

                # If the prediction was empty, remove from unmatched
                if pred_mask.sum() == 0:
                    unmatched.discard(key)

            if not matched_prediction and mask.sum():
                match_info['misdetections'].append({'object_label': obj_label, 'overlap': 0})

            matches_data['matches_per_obj'][obj_label] = match_info

        matches_data['unmatched'] = list(unmatched)

        # Append the matches-related data for the current mask_id
        matches_info.append(matches_data)

    # Now `matches_info` holds the matches, mismatches, and unmatched data

    inference = {}
    for match_info in matches_info:
        for obj_label, match_data in match_info['matches_per_obj'].items():
            if obj_label not in inference:
                inference[obj_label] = {'misdetections': 0, 'detections': 0, 'false_alarms': 0, 'ok': 0, 'processed': 0}

            # Optionally consider only matches
            if len(match_data['matches']) or len(match_data['mismatches']):
                inference[obj_label]['detections'] += 1
                inference[obj_label]['processed'] += 1
            elif len(match_data['misdetections']):
                inference[obj_label]['misdetections'] += 1
                inference[obj_label]['processed'] += 1

        # Process false alarms
        if len(match_info['unmatched']):
            inference[-1] = inference.get(-1, {'ok': 0, 'false_alarms': 0, 'processed': 0})
            inference[-1]['false_alarms'] += 1

        if (sum([len(_['matches']) + len(_['mismatches']) + len(_['misdetections']) for _ in
                 match_info['matches_per_obj'].values()]) == 0 and len(match_info['unmatched']) == 0):
            inference[-1] = inference.get(-1, {'ok': 0, 'false_alarms': 0, 'processed': 0})
            inference[-1]['ok'] += 1
            inference[-1]['processed'] += 1
    return inference