from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances
from detectron2.layers import cat, batched_nms

from d2sgg.structures import SGInstances

def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill

    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return:
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec + num_classes * arange_inds] = fill
    return onehot_result

def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor

def pairwise_union(proposal1: Instances, proposal2: Instances):
    """
    Compute the union region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      union proposals
    """
    assert len(proposal1) == len(proposal2) and proposal1.image_size == proposal2.image_size
    boxes1, boxes2 = proposal1.pred_boxes.tensor, proposal2.pred_boxes.tensor
    union_boxes = cat((
        torch.min(boxes1[:,:2], boxes2[:,:2]),
        torch.max(boxes1[:,2:], boxes2[:,2:])
        ), dim=1)

    union_proposal = SGInstances(proposal1.image_size)
    union_proposal.union_boxes = union_boxes = Boxes(union_boxes)
    union_proposal.union_scores = union_scores = proposal1.scores * proposal2.scores
    return union_proposal

def resize_boxes(boxes: Boxes, orig_size: Tuple[int, int], output_size: Tuple[int, int]):
    """
    Returns a resized copy of bounding boxes
    orig_size: (height, width)
    output_size: (height, width)
    """

    if isinstance(output_size[0], torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_size[1].float()
        output_height_tmp = output_size[0].float()
    else:
        output_width_tmp = output_size[1]
        output_height_tmp = output_size[0]

    scale_x, scale_y = (
        output_width_tmp / orig_size[1],
        output_height_tmp / orig_size[0],
    )
    
    boxes.scale(scale_x, scale_y)
    boxes.clip(output_size)

    return boxes

def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:,:,:,0]*inter[:,:,:,1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:,2]- boxes_flat[:,0]+1.0)*(
        boxes_flat[:,3]- boxes_flat[:,1]+1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union

def obj_prediction_nms(boxes_per_cls, pred_logits, bg_class_idx, nms_thresh=0.3):
    """
    boxes_per_cls:               [num_obj, num_cls, 4]
    pred_logits:                 [num_obj, num_category]
    """
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0), 
                              boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh

    prob_sampled = F.softmax(pred_logits, dim=1).cpu().numpy()
    prob_sampled[:, bg_class_idx] = 0  # set bg to 0

    pred_label = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0  # This way we won't re-sample

    return pred_label