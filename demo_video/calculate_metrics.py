from utils import check_overlap


def compute_detection_statistics(masks, prediction_masks):

    inference = {}
    for mask_id, masks_dict in enumerate(masks):
        ok_image = True
        prediction = False
        if masks_dict is not None:
            for obj_label, mask in masks_dict.items():  # Iterate through each object label and its mask
                if obj_label not in inference:
                    inference[obj_label] = {'misdetections': 0, 'detections': 0, 'false_alarms': 0,
                                            'ok': 0, 'processed': 0}

                # Handle predictions when they are available
                overlap = 0
                if obj_label in prediction_masks:
                    if mask_id < len(prediction_masks[obj_label]):
                        pred_mask = prediction_masks[obj_label][mask_id]
                        overlap = check_overlap(mask, pred_mask)

                if mask is not None and mask.sum() > 0 and overlap:
                    inference[obj_label]['detections'] += 1
                    prediction = True
                    ok_image = False
                    inference[obj_label]['processed'] += 1
                elif mask is not None and mask.sum() > 0 and not overlap:
                    inference[obj_label]['misdetections'] += 1
                    ok_image = False
                    inference[obj_label]['processed'] += 1
        # Compute FAs
        for obj_label, prediction_mask in prediction_masks.items():
            if mask_id < len(prediction_mask):
                pred_mask = prediction_mask[mask_id]
                if pred_mask.sum() and (masks_dict is None or obj_label not in masks_dict or
                                    masks_dict[obj_label].sum() == 0):
                    inference[obj_label] = inference.get(obj_label, {'false_alarms': 0, 'processed': 0})
                    inference[obj_label]['false_alarms'] += 1
                    inference[obj_label]['processed'] += 1
                    prediction = True
    if not prediction and ok_image:
        inference[-1] = inference.get(-1, {'ok': 0, 'processed': 0})
        inference[-1]['ok'] += 1
        inference[-1]['processed'] += 1
    return inference