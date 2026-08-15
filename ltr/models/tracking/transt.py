import torch.nn as nn
from ltr import model_constructor

import numpy as np
import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        self.debug = 0
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    @staticmethod
    def _attention_to_heatmap(attn_map):
        """Convert 1D attention vector to 2D spatial heatmap."""
        if attn_map is None:
            return None

        # Convert tensor to numpy
        if hasattr(attn_map, 'cpu'):
            map_array = attn_map.detach().cpu().numpy()
        else:
            map_array = attn_map
        
        if map_array.ndim != 1:
            print(f"[debug] Attention map has unexpected shape: {map_array.shape}")
            return None

        # Compute grid size
        grid_size = int(round(np.sqrt(map_array.size)))
        if grid_size * grid_size != map_array.size:
            print(f"[debug] Cannot reshape {map_array.size} elements into square grid. size={map_array.size}, grid_size={grid_size}")
            return None

        # Reshape and normalize
        heatmap = map_array.reshape(grid_size, grid_size).astype(np.float32)
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.ones_like(heatmap) * 0.5
        
        return heatmap

    def forward(self, search, template):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        if self.debug >= 2:
            hs, decoder_attn = self.featurefusion_network(self.input_proj(src_template), mask_template,
                                                         self.input_proj(src_search), mask_search,
                                                         pos_template[-1], pos_search[-1], debug=True)
            debug_heatmap = self._attention_to_heatmap(decoder_attn)
            if debug_heatmap is not None:
                out = {'pred_logits': self.class_embed(hs)[-1], 'pred_boxes': self.bbox_embed(hs).sigmoid()[-1],
                       'debug_attention': debug_heatmap}
                return out
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search,
                                       pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def track(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None

        debug = getattr(self, 'debug', 0)
        print(f"[TransT.track] debug={debug}")
        if debug == 1:
            hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            print(f"[TransT.track] debug=1 path, no attention")
            return out

        if debug >= 2:
            print(f"[TransT.track] debug={debug} path, requesting attention")
            hs, decoder_attn = self.featurefusion_network(self.input_proj(src_template), mask_template,
                                                         self.input_proj(src_search), mask_search,
                                                         pos_template[-1], pos_search[-1], debug=True)
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            heatmap = self._attention_to_heatmap(decoder_attn)
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if heatmap is not None:
                print(f"[TransT.track] Extracted attention heatmap with shape {heatmap.shape}")
                out['debug_attention'] = heatmap
            else:
                print(f"[TransT.track] Failed to extract attention heatmap")
            return out

        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    # ensure device compatibility: fall back to cpu if cuda unavailable
    if settings.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA unavailable, switching device to cpu')
        device = torch.device('cpu')
    else:
        device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion