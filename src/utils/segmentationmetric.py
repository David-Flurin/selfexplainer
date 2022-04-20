import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics


class MultiLabelSegmentationMetrics(torchmetrics.Metric):
    def __init__(self, eps=1e-5):
        super().__init__()

        self.seg_metric = SegmentationMetrics(activation='sigmoid')
        self.add_state("tp", torch.tensor(0.0))
        self.add_state("fp", torch.tensor(0.0))
        self.add_state("fn", torch.tensor(0.0))
        self.add_state("tn", torch.tensor(0.0))
        self.eps = eps

    def update(self, pred, true):
        with torch.no_grad():
            b, c, h, w = true.size()
            mask_true = torch.zeros((b, h, w), device=true.device)
                        
            for i in range(0, c):
                mask_true[:] += torch.where(true[:, i] > 0, i, 0)

            pred_activ = torch.sigmoid(pred)
            pred_max = torch.max(pred_activ, dim=1)[0]

            prediction = torch.clamp(torch.round(pred_max), 0, 1)
            mask_true = torch.clamp(torch.round(mask_true), 0, 1)

            # from matplotlib import pyplot as plt
            # plt.imshow(prediction[0])
            # plt.show()
            # plt.imshow(mask_true[0])
            # plt.show()

            self.fp += torch.sum(torch.eq(prediction, 1) & torch.eq(mask_true, 0))
            self.fn += torch.sum(torch.eq(prediction, 0) & torch.eq(mask_true, 1))
            self.tp += torch.sum(torch.eq(prediction, 1) & torch.eq(mask_true, 1))
            self.tn += torch.sum(torch.eq(prediction, 0) & torch.eq(mask_true, 0))

    def compute(self):
        self.pixel_acc = (self.tp + self.tn + self.eps) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        self.iou = (self.tp + self.eps) / (self.fn + self.fp + + self.tp +  self.eps)
        self.precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        self.recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        return {'Accuracy': self.pixel_acc.item(), 'IoU': self.iou.item(), 'Precision': self.precision.item(), 'Recall': self.recall.item()}

    def save(self, model, classifier_type, dataset):
        f = open(model + "_" + classifier_type + "_" + dataset + "_" + "test_metrics.txt", "w")
        f.write("Accuracy: " + str(self.accuracy) + "\n")
        f.write("Precision: " + str(self.precision) + "\n")
        f.write("Recall: " + str(self.recall) + "\n")
        f.write("IoU: " + str(self.iou))
        f.close()


class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.
    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.
    
    Pixel accuracy measures how many pixels in a image are predicted correctly.
    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.
    
    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need 
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.
    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.
    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.
    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5
        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.
        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.
        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.
    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.
    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, self.fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            # from matplotlib import pyplot as plt
            # fig = plt.figure(figsize=(8, 8))
            # fig.add_subplot(2,2,1)
            # plt.imshow(class_pred[0], vmin=0, vmax=1)
            # fig.add_subplot(2,2,2)
            # plt.imshow(class_gt[0], vmin=0, vmax=1)
            # fig.add_subplot(2,2,3)
            # plt.imshow(class_pred[1], vmin=0, vmax=1)
            # fig.add_subplot(2,2,4)
            # plt.imshow(class_gt[1], vmin=0, vmax=1)
            # plt.show()

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            self.fp = torch.sum(pred_flat) - tp
            self.fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # self.fp = np.sum(matrix[1, :])
        # self.fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) +self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]) + self.eps)
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        self.fp = torch.sum(output * (1 - target))  # FP
        self.fn = torch.sum((1 - output) * target)  # FN
        self.tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (self.tp + self.tn + self.eps) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + self.fp + self.fn + self.eps)
        precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        specificity = (tn + self.eps) / (tn + self.fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return [pixel_acc, dice, precision, specificity, recall]
