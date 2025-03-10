import torch

def box_iou(boxes1, boxes2):
    # Ensure boxes have proper shape
    if boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError(f"boxes1 should have shape [N, 4], but got {boxes1.shape}")
    if boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError(f"boxes2 should have shape [M, 4], but got {boxes2.shape}")

    if boxes1.size(0) == 0 or boxes2.size(0) == 0:
        # No boxes to compare, return empty IoU matrix
        return torch.zeros((boxes1.size(0), boxes2.size(0)), dtype=boxes1.dtype, device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def match_proposals_to_targets(proposals, gt_boxes, fg_iou_thresh=0.5, bg_iou_thresh=0.4):
    if gt_boxes.numel() == 0:
        device = proposals.device
        return (
            torch.full((proposals.shape[0],), -1, dtype=torch.int64, device=device),
            torch.zeros((proposals.shape[0],), dtype=torch.int64, device=device)
        )

    ious = box_iou(proposals, gt_boxes)
    max_iou, matched_gt_idxs = ious.max(dim=1)

    labels = torch.full((proposals.shape[0],), -1, dtype=torch.int64, device=proposals.device)
    labels[max_iou >= fg_iou_thresh] = 1
    labels[max_iou < bg_iou_thresh] = 0

    return matched_gt_idxs, labels

def subsample_labels(labels, batch_size_per_image=256, positive_fraction=0.5):
    positive_idxs = torch.nonzero(labels == 1).squeeze(1)
    negative_idxs = torch.nonzero(labels == 0).squeeze(1)

    num_pos = int(batch_size_per_image * positive_fraction)
    num_pos = min(positive_idxs.numel(), num_pos)

    num_neg = batch_size_per_image - num_pos
    num_neg = min(negative_idxs.numel(), num_neg)

    perm1 = torch.randperm(positive_idxs.numel(), device=labels.device)[:num_pos]
    perm2 = torch.randperm(negative_idxs.numel(), device=labels.device)[:num_neg]

    return torch.cat([positive_idxs[perm1], negative_idxs[perm2]], dim=0)

def apply_deltas_to_proposals(deltas, proposals):
    # deltas: (N, 4)
    # proposals: (N, 4)
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # CLAMP dw and dh to prevent overflow in exp()
    dw = torch.clamp(dw, min=-10, max=10)
    dh = torch.clamp(dh, min=-10, max=10)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=1)

    return pred_boxes
