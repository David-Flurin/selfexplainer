import torch
import torchmetrics
import numpy as np

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .helper import get_class_dictionary

class SingleLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.add_state("true_positives", torch.zeros(num_classes))
        self.add_state("false_positives", torch.zeros(num_classes))
        self.add_state("true_negatives", torch.zeros(num_classes))
        self.add_state("false_negatives", torch.zeros(num_classes))

    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                self.true_negatives += 1
                top_class_prediction = batch_sample_logits.argmax(-1)
                label_idx = (labels[i] == 1).nonzero()
                if label_idx == top_class_prediction:
                    self.true_positives[label_idx] += 1
                    self.true_negatives[label_idx] -= 1
                else:
                    self.false_negatives[label_idx] += 1
                    self.false_positives[top_class_prediction] += 1
                    self.true_negatives[label_idx] -= 1
                    self.true_negatives[top_class_prediction] -= 1

            self.true_positives = self.true_positives.long()
            self.true_negatives = self.true_negatives.long()
            self.false_negatives = self.false_negatives.long()
            self.false_positives = self.false_positives.long()

    def compute(self):
        self.accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)).mean()
        self.precision = (self.true_positives / (self.true_positives + self.false_positives)).mean()
        self.recall = (self.true_positives / (self.true_positives + self.false_negatives)).mean()
        self.f_score = ((2 * self.true_positives) / (2 * self.true_positives + self.false_positives + self.false_negatives)).mean()

        print('Acc', self.accuracy.item())
        r = torch.round(self.accuracy, decimals=2)
        #i = r.item()
        print(r)
        return {'Accuracy': r, 'Precision': self.precision.item(), 'Recall': self.recall.item(), 'F-Score': self.f_score.item()}

class MultiLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if labels[i][j] == 1.0:
                        if batch_sample_logits[j] >= self.threshold:
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0
                    else:
                        if batch_sample_logits[j] >= self.threshold:
                            self.false_positives += 1.0
                        else:
                            self.true_negatives += 1.0

    def compute(self):
        self.accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives))
        self.precision = (self.true_positives / (self.true_positives + self.false_positives))
        self.recall = (self.true_positives / (self.true_positives + self.false_negatives))
        self.f_score = ((2 * self.true_positives) / (2 * self.true_positives + self.false_positives + self.false_negatives))

        return {'Accuracy': round(self.accuracy.item(), 2), 'Precision': self.precision.item(), 'Recall': self.recall.item(), 'F-Score': self.f_score.item()}

    def save(self, model, classifier_type, dataset):
        f = open(model + "_" + classifier_type + "_" + dataset + "_" + "test_metrics.txt", "w")
        f.write("Accuracy: " + str(self.accuracy.item()) + "\n")
        f.write("Precision: " + str(self.precision.item()) + "\n")
        f.write("Recall: " + str(self.recall.item()) + "\n")
        f.write("F-Score: " + str(self.f_score.item()))
        f.close()



class ClassificationMultiLabelMetrics():
    def __init__(self, threshold, num_classes, gpu, loss, classwise = False, dataset=None):
        device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, threshold=threshold).to(device)
        self.precision = torchmetrics.Precision(num_classes=num_classes, threshold=threshold).to(device)
        self.recall = torchmetrics.Recall(num_classes=num_classes, threshold=threshold).to(device)
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, threshold=threshold).to(device)
        if loss not in ['bce', 'ce']:
            raise ValueError('Unknown loss for metrics')
        self.loss = loss

        self.classwise = classwise

        if self.classwise:
            if dataset == None:
                raise ValueError('If calculated classwise, dataset must be specified.')
            self.class_list = list(get_class_dictionary(dataset).keys())
            self.class_accuracy = torchmetrics.Accuracy(num_classes=num_classes, threshold=threshold, average=None).to(device)
            self.class_precision = torchmetrics.Precision(num_classes=num_classes, threshold=threshold, average=None).to(device)
            self.class_recall = torchmetrics.F1Score(num_classes=num_classes, threshold=threshold, average=None).to(device)
            self.class_f1 = torchmetrics.Recall(num_classes=num_classes, threshold=threshold, average=None).to(device)
        
            

    def __call__(self, activations, targets):
        loss_fn = torch.sigmoid if self.loss == 'bce' else lambda x: torch.nn.functional.softmax(x, dim=-1)
        logits = loss_fn(activations)
        # if self.loss == 'ce':
        #     targets = targets.nonzero(as_tuple=True)[1]
        self.accuracy(logits, targets)
        self.precision(logits, targets)
        self.recall(logits, targets)
        self.f1(logits, targets)

        if self.classwise:
            self.class_accuracy(logits, targets)
            self.class_precision(logits, targets)
            self.class_recall(logits, targets)
            self.class_f1(logits, targets)


    def compute(self):
        if not self.classwise:
            return {'Accuracy': self.accuracy.compute(), 'Precision': self.precision.compute(), 'Recall': self.recall.compute(), 'F-Score': self.f1.compute()}
        else:
            return {'Total': {'Accuracy': self.accuracy.compute(), 'Precision': self.precision.compute(), 'Recall': self.recall.compute(), 'F-Score': self.f1.compute()},
            'Class': {'Accuracy': self.class_accuracy.compute(), 'Precision': self.class_precision.compute(), 'Recall': self.class_recall.compute(), 'F-Score': self.class_f1.compute()}}
        #return f'Acc: {self.accuracy.compute()'

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        if self.classwise:
            self.class_accuracy.reset()
            self.class_precision.reset()
            self.class_recall.reset()
            self.class_f1.reset()




### BELOW ARE JUST UTILITY FUNCTIONS, NOT THE ONES USED FOR THE RESULTS IN THE PAPER/THESIS ###

class MultiLabelTopPredictionAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("correct", torch.tensor(0.0))
        self.add_state("total", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                self.total += 1.0
                top_class_prediction = batch_sample_logits.argmax(-1)
                if (targets[i][top_class_prediction] == 1.0):
                    self.correct += 1.0

    def compute(self):
        return {'Top prediction accuracy': self.correct / self.total}

class MultiLabelPrecision(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if (batch_sample_logits[j] >= self.threshold):
                        if (targets[i][j] == 1.0):
                            self.true_positives += 1.0
                        else:
                            self.false_positives += 1.0

    def compute(self):
        return self.true_positives / (self.true_positives + self.false_positives)

class MultiLabelRecall(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, targets):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if (targets[i][j] == 1.0):
                        if (batch_sample_logits[j] >= self.threshold):
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0




class IoU():
    def __init__(self):
        self.iou = MeanAveragePrecision(iou_type="segm")

    def __call__(self, mask, seg, logits, target_vector):
        targets = []
        preds = []
        for b in range(mask.size(0)):
            target_dict = {}
            target_dict['labels'] = target_vector[b].nonzero()
            target_dict['masks'] = seg[b]
            targets.append(target_dict)

            objects = int(target_vector[b].sum().item())
            highest_idx = torch.sort(logits[b])[1][0:objects]
            pred_dict = {}
            pred_dict['labels'] = highest_idx
            pred_dict['masks'] = mask[b]
            preds.append(pred_dict)

        self.iou(preds, targets)
        
        
    
    def compute(self):
        return self.iou.compute

    def reset(self):
        self.iou.reset()



