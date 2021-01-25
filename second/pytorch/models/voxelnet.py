import time
import pathlib
from enum import Enum
from functools import reduce
from second.core import box_np_ops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import spconv
import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)

from second.pytorch.models import rpn, middle, voxel_encoder, fusion
import pickle
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import second.utils.eval as se
import math
import second.data.kitti_common as kitti

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        # this part is acutually running
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_voxel_classifier=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 use_rc_net=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self.measure_time = measure_time
        vfe_class_dict = {
            "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
            "VoxelFeatureExtractorV3": voxel_encoder.VoxelFeatureExtractorV3,
            "SimpleVoxel": voxel_encoder.SimpleVoxel
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        self.voxel_feature_extractor = vfe_class(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance)
        if len(middle_num_filters_d2) == 0:
            if len(middle_num_filters_d1) == 0:
                num_rpn_input_filters = vfe_num_filters[-1]
            else:
                num_rpn_input_filters = middle_num_filters_d1[-1]
        else:
            num_rpn_input_filters = middle_num_filters_d2[-1]

        if use_sparse_rpn: # don't use this. just for fun.
            self.sparse_rpn = rpn.SparseRPN(
                output_shape,
                # num_input_features=vfe_num_filters[-1],
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2,
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=num_rpn_input_filters * 2,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)
        else:
            mid_class_dict = {
                "SparseMiddleExtractor": middle.SparseMiddleExtractor,
                "SpMiddleD4HD": middle.SpMiddleD4HD,
                "SpMiddleD8HD": middle.SpMiddleD8HD,
                "SpMiddleFHD": middle.SpMiddleFHD,
                "SpMiddleFHDV2": middle.SpMiddleFHDV2,
                "SpMiddleFHDLarge": middle.SpMiddleFHDLarge,
                "SpMiddleResNetFHD": middle.SpMiddleResNetFHD,
                "SpMiddleD4HDLite": middle.SpMiddleD4HDLite,
                "SpMiddleFHDLite": middle.SpMiddleFHDLite,
                "SpMiddle2K": middle.SpMiddle2K,
                "MiddleExtractor": middle.MiddleExtractor,
                "SpMiddleVision":middle.SpMiddleVision
            }
            mid_class = mid_class_dict[middle_class_name]
            mid_class_vision = mid_class_dict['SpMiddleVision']
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=middle_num_input_features,
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            rpn_class_dict = {
                "RPN": rpn.RPN,
                "RPNV2": rpn.RPNV2,
            }
            rpn_class = rpn_class_dict[rpn_class_name]
            self.rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}
    def start_timer(self, *names):
        if not self.measure_time:
            return
        for name in names:
            self._time_dict[name] = time.time()
        torch.cuda.synchronize()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def set_global_step(self,step_value):
        self.global_step = step_value

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example,detection_2d_path):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        #images = example["images"]
        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        # t = time.time()
        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points)
        self.end_timer("voxel_feature_extractor")
        # torch.cuda.synchronize()
        # print("vfe time", time.time() - t)
        batch_P2 = example["P2"]
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        if self._use_sparse_rpn:
            preds_dict = self.sparse_rpn(voxel_features, coors, batch_size_dev)
        else:
            self.start_timer("middle forward")
            spatial_features = self.middle_feature_extractor(
                voxel_features, coors, batch_size_dev)
            self.end_timer("middle forward")
            self.start_timer("rpn forward")
            if self._use_bev:
                preds_dict = self.rpn(spatial_features, example["bev_map"])
            else:
                preds_dict = self.rpn(spatial_features)
            self.end_timer("rpn forward")
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        # cls_preds shape [batch_size,200,176,2]
        if self.training:
            labels = example['labels']
            # labels shape: [batch_size,70400]
            # here label = 1 means positive, label = 0 means negative, label = -1 means dont care
            reg_targets = example['reg_targets']
            # reg_targets is the
            # reg_targets shape: [batch,70400,7]
            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            # pos_cls_weight and net_clas_weight are all 1.0
            # _loss_norm_type is: LossNormType.NormByNumPositives
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            #here unsqueeze(-1) means adding another dimention at the end,eg:[1,70400] ---> [1,70400,1]
            # cls_preds shape: [batch,200,176,2]
            # cls_targets shape : [batch,70400,1]
            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            # loc_loss shape: [batch,70400,7]
            # cls_loss shape: [batch,70400,1]
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            # this self._loc_loss_weight is the beta2 in the paper, the constant coefficient for localization loss
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            # here the loss is just a number in tensor format
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'],
                                                   reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(
                    batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(
                    dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight
            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            self.start_timer("predict")
            img_idx = example['image_idx'][0]
            detection_2d_result_path = pathlib.Path(detection_2d_path)
            detection_2d_file_name = f"{detection_2d_result_path}/{kitti.get_image_index_str(img_idx)}.txt"
            with open(detection_2d_file_name, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(' ') for line in lines]
            predicted_class = np.array([x[0] for x in content],dtype='object')
            predicted_class_index = np.where(predicted_class=='Car')
            detection_result = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
            score = np.array([float(x[15]) for x in content])  # 1000 is the score scale!!!
            f_detection_result=np.append(detection_result,score.reshape(-1,1),1)
            middle_predictions=f_detection_result[predicted_class_index,:].reshape(-1,5)
            top_predictions=middle_predictions[np.where(middle_predictions[:,4]>=-100)]
            res, iou_test, tensor_index = self.train_stage_2(example, preds_dict,top_predictions)
            self.end_timer("predict")

            return res, preds_dict, top_predictions, iou_test, tensor_index


    def train_stage_2(self, example, preds_dict,top_predictions):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_anchors_reshape = batch_anchors.reshape(1,200,176,14)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        batch_image_shape = example["image_shape"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1
        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()

            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
                #total_scores = cls_preds   # use this if you want to fuse raw log score
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            # finally generate predictions.
            final_box_preds = box_preds
            final_scores = total_scores
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)

            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            img_height = batch_image_shape[0,0]
            img_width = batch_image_shape[0,1]
            minxy[:,0] = torch.clamp(minxy[:,0],min = 0,max = img_width)
            minxy[:,1] = torch.clamp(minxy[:,1],min = 0,max = img_height)
            maxxy[:,0] = torch.clamp(maxxy[:,0],min = 0,max = img_width)
            maxxy[:,1] = torch.clamp(maxxy[:,1],min = 0,max = img_height)
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                #"label_preds": label_preds,
                "image_idx": img_idx,
            }
            predictions_dicts.append(predictions_dict)
            dis_to_lidar = torch.norm(box_preds[:,:2],p=2,dim=1,keepdim=True)/82.0
            box_2d_detector = np.zeros((200, 4))
            box_2d_detector[0:top_predictions.shape[0],:]=top_predictions[:,:4]
            box_2d_detector = top_predictions[:,:4]
            box_2d_scores = top_predictions[:,4].reshape(-1,1)
            time_iou_build_start=time.time()
            overlaps1 = np.zeros((900000,4),dtype=box_2d_preds.detach().cpu().numpy().dtype)
            tensor_index1 = np.zeros((900000,2),dtype=box_2d_preds.detach().cpu().numpy().dtype)
            overlaps1[:,:] = -1
            tensor_index1[:,:] = -1
            #final_scores[final_scores<0.1] = 0
            #box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0 
            iou_test,tensor_index, max_num = se.build_stage2_training(box_2d_preds.detach().cpu().numpy(),
                                                box_2d_detector,
                                                -1,
                                                final_scores.detach().cpu().numpy(),
                                                box_2d_scores,
                                                dis_to_lidar.detach().cpu().numpy(),
                                                overlaps1,
                                                tensor_index1)
            time_iou_build_end=time.time()
            iou_test_tensor = torch.FloatTensor(iou_test)  #iou_test_tensor shape: [160000,4]
            tensor_index_tensor = torch.LongTensor(tensor_index)
            iou_test_tensor = iou_test_tensor.permute(1,0)
            iou_test_tensor = iou_test_tensor.reshape(1,4,1,900000)
            tensor_index_tensor = tensor_index_tensor.reshape(-1,2)
            if max_num == 0:
                non_empty_iou_test_tensor = torch.zeros(1,4,1,2)
                non_empty_iou_test_tensor[:,:,:,:] = -1
                non_empty_tensor_index_tensor = torch.zeros(2,2)
                non_empty_tensor_index_tensor[:,:] = -1
            else:
                non_empty_iou_test_tensor = iou_test_tensor[:,:,:,:max_num]
                non_empty_tensor_index_tensor = tensor_index_tensor[:max_num,:]

        return predictions_dicts, non_empty_iou_test_tensor, non_empty_tensor_index_tensor


    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled,
                       vox_preds=None,
                       vox_labels=None,
                       vox_weights=None):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net

# attention! sin(a-b) = sin(a)cos(b) - cos(a)sin(b), this is the logic behind this
def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    # cls_preds before here is in shape of [batch,200,176,2]
    if encode_background_as_zeros:
        # we do encode background as zeros
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    # cls_preds here is in shape of [batch,70400,1]
    cls_targets = cls_targets.squeeze(-1)
    # the shape of cls_targets after squeeze is [batch,70400]
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    # shape of one_hot_targets here is [batch, 70400, 2]
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    # noted that for this focal loss function, alpha = 0.25, gamma = 2.0
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0 #becaues labels=-1 means dont't care
    # cared: [batch,70400]
    # cared: [N, num_anchors]
    positives = labels > 0 #because labels=1 means positive
    negatives = labels == 0  #because labels=0 means negative
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    #because only positive regressions are included in the localization loss
    #for classification loss, both positive and negative are included
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        # pos_normalizer shape: [batch,1]
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    #print("++++++++++++ anchors shape is",anchors.shape)
    #print("++++++++++++ reg_targets shape is",reg_targets.shape)
    anchors = anchors.view(batch_size, -1, 7)
    # reg_targets[...,-1] is theta_g -theta_a, anchors[...,-1] is theta_a, so plus them together will give you theta_g
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    #print("+++++++ rog_gt shape is:",rot_gt.shape)
    dir_cls_targets = (rot_gt > 0).long()
    #print("+++++++ dir_cls_targets shape is",dir_cls_targets.shape,"number of ones: ",dir_cls_targets.sum())
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    #print("+++++++ the final dir_cls_targets shape is",dir_cls_targets.shape,"number of ones: ",dir_cls_targets.sum())
    return dir_cls_targets
