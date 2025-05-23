import torch
import random

def get_targets_from_segmentations(segmentation, dataset, num_classes, include_background_class=True, gpu=0, synthetic_target='texture'):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")

    if segmentation.dim() == 3:
        b,h,w = segmentation.size()
    else:
        b,h,w,_ = segmentation.size()
    targets = torch.zeros((b, num_classes, h, w), device=device)

    if dataset == "SYN":
        for i in range(b):
            for c,color in enumerate(get_synthetic_class_colors(include_background_class=include_background_class, synthetic_target=synthetic_target)):
                targets[i, c] = torch.where(torch.all((segmentation[i] == torch.tensor(color, device=device)), dim=-1), 1., 0.)

    if dataset == "COLOR":
        rgb = segmentation.dim() == 4
        seg = torch.zeros((b, 2, h, w), device=device)
        
        for i in range(b):
            if rgb:
                seg[i, 0] = (segmentation[i] == torch.tensor([255., 0., 0.], device=segmentation.device)).all(dim=2).float()
                seg[i, 1] = (segmentation[i] == torch.tensor([0., 0., 255.], device=segmentation.device)).all(dim=2).float()
            else:
                seg[i, 0] = torch.where(segmentation[i] == 170., 1., 0.)
                seg[i, 1] = torch.where(segmentation[i] == 255., 1., 0.)
        if include_background_class:
            targets[:, 0] = torch.where(torch.all((segmentation == torch.tensor(0., device=device)), dim=-1), 1., 0.)
            targets[:, 1:2] = seg
        else:
            targets = seg

    return targets


# Only returns 1 filename, not an array of filenames
# Ônly used with batch size 1
def get_filename_from_annotations(annotations, dataset):
    if dataset in ["VOC", 'SMALLVOC', 'VOC2012', 'OI_SMALL', 'OI', 'OI_LARGE']:
        filename = annotations[0]['annotation']['filename']
    elif dataset in ["SYN", 'SYN_SAVED', 'SYN_MULTI', 'COLOR']:
        filename = annotations[0]['filename']
    else:
        raise Exception("Unknown dataset: " + dataset)

    return filename

def get_VOC_dictionary(include_background_class):
    if include_background_class:
        target_dict = {'background' : 0, 'aeroplane' : 1, 'bicycle' : 2, 'bird' : 3, 'boat' : 4, 'bottle' : 5, 'bus' : 6, 'car' : 7, 
                'cat' : 8, 'chair' : 9, 'cow' : 10, 'diningtable' : 11, 'dog' : 12, 'horse' : 13, 'motorbike' : 14, 'person' : 15, 
                'pottedplant' : 16, 'sheep' : 17, 'sofa' : 18, 'train' : 19, 'tvmonitor' : 20}
    else:
        target_dict = {'aeroplane' : 0, 'bicycle' : 1, 'bird' : 2, 'boat' : 3, 'bottle' : 4, 'bus' : 5, 'car' : 6, 
                'cat' : 7, 'chair' : 8, 'cow' : 9, 'diningtable' : 10, 'dog' : 11, 'horse' : 12, 'motorbike' : 13, 'person' : 14, 
                'pottedplant' : 15, 'sheep' : 16, 'sofa' : 17, 'train' : 18, 'tvmonitor' : 19}

    return target_dict

def get_OI_dictionary(include_background_class):
    target_dict = {'aeroplane' : 0, 'train' : 1, 'bird' : 2, 'motorbike' : 3, 'bottle' : 4, 'bus' : 5, 'car' : 6, 
            'cat' : 7, 'sofa' : 8, 'sheep' : 9, 'person' : 10, 'dog' : 11, 'horse' : 12  
            }

    if include_background_class:
        target_dict['background'] = len(target_dict.values())

    return target_dict

def get_large_OI_dictionary(include_background_class):
    
    target_dict = {'flower': 0, 'fish': 1, 'monkey': 2, 'cake': 3, 'sculpture': 4, 'lizard': 5, 'mobile phone': 6, 'camera': 7, 'bread': 8, 
                    'guitar': 9, 'snake': 10, 'handbag': 11, 'pastry': 12, 'ball': 13, 'flag': 14, 'piano': 15, 'rabbit': 16, 'book': 17, 'mushroom': 18, 'dress': 19}

    if include_background_class:
        target_dict['background'] = len(target_dict.values())

    return target_dict

def get_small_OI_dictionary(include_background_class):
    if include_background_class:
        target_dict = {'background' : 0, 'cat' : 1, 'dog' : 2, 'bird' : 3}
    else:
        target_dict = {'cat' : 0, 'dog' : 1, 'bird' : 2}

    return target_dict

def get_synthetic_target_dictionary(include_background_class, synthetic_target):
    if synthetic_target == 'texture':
        target_dict = {'bubblewrap' : 0, 'forest' : 1, 'fur' : 2, 'moss' : 3, 'paint' : 4, 'rock' : 5, 'wood' : 6, 'splatter': 7 
                }
    elif synthetic_target == 'shape':
        target_dict = {'circle' : 0, 'triangle' : 1, 'square' : 2, 'pentagon' : 3, 'hexagon' : 4, 'octagon' : 5, 'heart' : 6, 'star': 7, 'cross': 8
                }
    else:
        raise ValueError('Target type must be texture or shape')

    if include_background_class:
        target_dict['background'] = len(target_dict.values())
        
    return target_dict

def get_color_dictionary(include_background_class, rgb=True):
    if rgb:
        target = {'red': 0, 'blue':1}
    else:
        target =  {'gray': 1, 'white':2}

    if include_background_class:
        target['background'] = len(target.values())
    
    return target

def get_class_dictionary(dataset, include_background_class=False, synthetic_target='texture', rgb=True):
    if dataset in ['VOC', 'VOC2012']:
        return get_VOC_dictionary(include_background_class=include_background_class)
    elif dataset == 'OI':
        return get_OI_dictionary(include_background_class=include_background_class)
    elif dataset == 'OI_LARGE':
        return get_large_OI_dictionary(include_background_class=include_background_class)
    elif dataset in ['SMALLVOC', 'OI_SMALL']:
        return get_small_OI_dictionary(include_background_class=include_background_class)
    elif dataset in ['SYN', 'SYN_SAVED', 'SYN_MULTI']:
        return get_synthetic_target_dictionary(include_background_class=include_background_class, synthetic_target=synthetic_target)
    elif dataset == 'COLOR':
        return get_color_dictionary(include_background_class=include_background_class, rgb=rgb)
    else:
        raise ValueError('Dataset not known.')


def get_targets_from_annotations(annotations, dataset, include_background_class=False, gpu=0, synthetic_target='texture'):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")
    
    if dataset in ["VOC", 'VOC2012', 'OI_LARGE']:
        target_dict = get_class_dictionary(dataset, include_background_class)
        objects = [item['annotation']['object'] for item in annotations]

        batch_size = len(objects)
        target_vectors = torch.full((batch_size, 20), fill_value=0.0, device=device)
        for i in range(batch_size):
            object_names = [item['name'] for item in objects[i]]

            for name in object_names:
                index = target_dict[name]
                target_vectors[i][index] = 1.0

    elif dataset == 'OI':
        target_dict = get_class_dictionary(dataset, include_background_class)
        objects = [item['annotation']['object'] for item in annotations]

        batch_size = len(objects)
        target_vectors = torch.full((batch_size, 13), fill_value=0.0, device=device)
        for i in range(batch_size):
            object_names = [item['name'] for item in objects[i]]

            for name in object_names:
                index = target_dict[name]
                target_vectors[i][index] = 1.0

    elif dataset in ["SMALLVOC", 'OI_SMALL']:
        target_dict = get_class_dictionary(dataset, include_background_class)
        objects = [item['annotation']['object'] for item in annotations]

        batch_size = len(objects)
        target_vectors = torch.full((batch_size, len(target_dict)), fill_value=0.0, device=device)
        for i in range(batch_size):
            object_names = [item['name'] for item in objects[i]]

            for name in object_names:
                try:
                    index = target_dict[name]
                    target_vectors[i][index] = 1.0
                except KeyError:
                    pass
        if target_vectors.sum() < batch_size:
            raise ValueError('Target vector is all zero')

            
    elif dataset == "COCO":
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 91), fill_value=0.0, device=device)
        for i in range(batch_size):
            targets = annotations[i]['targets']
            for target in targets:
                target_vectors[i][target] = 1.0


    elif dataset in ["SYN", "SYN_SAVED", "SYN_MULTI"]:
        target_dict = get_synthetic_target_dictionary(include_background_class=False, synthetic_target=synthetic_target)
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 8), fill_value=0.0, device=device)
        for i in range(batch_size):
            targets = annotations[i]['objects']
            for obj in targets:
                if synthetic_target == 'texture':
                    name = obj[1]
                else:
                    name = obj[0]
                index = target_dict[name]
                target_vectors[i][index] = 1.0

    elif dataset == 'COLOR':
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 8), fill_value=0.0, device=device)
        for i in range(batch_size):
            target_vectors[i] = torch.Tensor(annotations[i]['logits'])

            target_vectors[i][annotations[i]] = 1.0

    return target_vectors

def get_synthetic_class_colors(include_background_class, synthetic_target):
    if synthetic_target == 'texture':
        class_colors = [[238, 30, 218], [11, 174, 227], [91, 187, 25], [104, 30, 191], [171, 88, 222], [253, 114, 104], [133, 10, 11], [230, 132, 230]]
    elif synthetic_target == 'shape':
        class_colors = [[208, 70, 121], [137, 218, 162], [115, 10, 147], [32, 201, 254], [215, 0, 57], [227, 161, 150], [135, 239, 205], [18, 222, 136], [111, 21, 62]]
    else:
        raise ValueError('Target type must be texture or shape')

    if include_background_class:
        class_colors = [[0,0,0]] + class_colors
        
    return class_colors

def extract_masks(segmentations, target_vectors, gpu=0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")

    batch_size, num_classes, h, w = segmentations.size()

    target_masks = torch.empty(batch_size, h, w, device=device)
    non_target_masks = torch.empty(batch_size, h, w, device=device)
    for i in range(batch_size):
        class_indices = target_vectors[i].eq(1.0)
        non_class_indices = target_vectors[i].eq(0.0)

        target_masks[i] = (segmentations[i][class_indices]).amax(dim=0)
       
        if torch.any(non_class_indices):
            non_target_masks[i] = (segmentations[i][non_class_indices]).amax(dim=0)
        else:
            non_target_masks[i] = torch.zeros((h, w))

    return target_masks.sigmoid(), non_target_masks.sigmoid()


class Distribution():
    def __init__(self):
        self.count = {}

    def update(self, name):
        if name in self.count:
            self.count[name] += 1
        else:
            self.count[name] = 1
    
    def print_distribution(self):
        total = sum(self.count.values())
        string = ''
        for name, i in self.count.items():
            string += f'{name}: {i/total:.2f}, '
        print(string)


class LogitStats():
    def __init__(self, classes):
        self.logits = [{'min': 1000, 'max':-1000} for i in range(classes)]

    def update(self, logits):
        max = torch.max(logits, dim=0)[0]
        min = torch.min(logits, dim=0)[0]
        for i,_ in enumerate(max):
            if self.logits[i]['min'] > min[i]:
                self.logits[i]['min'] = min[i].item()
            if self.logits[i]['max'] < max[i]:
                self.logits[i]['max'] = max[i].item()

    def plot(self, save_path, labels):
        from matplotlib import pyplot as plt
        
        x = range(len(self.logits))
        mins = [v['min'] for v in self.logits]
        maxs = [v['max'] for v in self.logits]
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x , mins, width, label='Min logit')
        rects2 = ax.bar(x, maxs, width, label='Max logit')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Logits value')
        ax.set_title('Min/Max logit for all classes')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # ax.bar_label(rects1, padding=3)
        # ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.savefig(save_path)


def get_class_weights(dataset):
    if dataset == 'VOC':
        stats = {
            "aeroplane": 112,
            "bicycle": 116,
            "bird": 180,
            "boat": 81,
            "bottle": 139,
            "bus": 97,
            "car": 376,
            "cat": 163,
            "chair": 224,
            "cow": 69,
            "diningtable": 97,
            "dog": 203,
            "horse": 139,
            "motorbike": 120,
            "person": 1025,
            "pottedplant": 133,
            "sheep": 48,
            "sofa": 111,
            "train": 127,
            "tvmonitor": 128
        }
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {target_dict[k]:v for k,v in stats.items()}
    elif dataset == 'VOC2012':
        stats = {'aeroplane': 432,
        'bicycle': 353,
        'bird': 560,
        'boat': 426,
        'bottle': 629,
        'bus': 292,
        'car': 1013,
        'cat': 605,
        'chair': 1178,
        'cow': 290,
        'diningtable': 304,
        'dog': 756,
        'horse': 350,
        'motorbike': 357,
        'person': 4194,
        'pottedplant': 484,
        'sheep': 400,
        'sofa': 281,
        'train': 313,
        'tvmonitor': 392}
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {target_dict[k]:v for k,v in stats.items()}
    elif dataset == 'OI_SMALL':
        stats = {
            'cat': 11116,
            'dog': 14283,
            'bird': 17730,
        }
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {k:stats[k] for k,v in target_dict.items()}
    elif dataset == 'OI':
        stats = {
            'person': 207352,
            'cat': 11116,
            'dog': 14283,
            'bird': 17730,
            'horse': 2709,
            'sheep': 990,
            'aeroplane': 10592,
            'bus': 3873,
            'car': 64519,
            'motorbike': 2350,
            'train': 8725,
            'bottle': 6974,
            'sofa': 598
        }
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {k:stats[k] for k,v in target_dict.items()}
    elif dataset == 'OI_LARGE':
        stats = {
            'flower': 60224,
            'fish': 5400,
            'monkey': 2118,
            'cake': 3263,
            'sculpture': 17690,
            'lizard': 1821,
            'mobile phone': 4182,
            'camera': 4879,
            'bread': 1629,
            'guitar': 17463,
            'snake': 1175,
            'handbag': 1725,
            'pastry': 892,
            'ball': 3441,
            'flag': 8177,
            'piano': 1175,
            'rabbit': 1146,
            'book': 7739,
            'mushroom': 1741,
            'dress': 29087
        }
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {k:stats[k] for k,v in target_dict.items()}
    elif dataset == 'OISMALL':
        stats = {
            'cat': 13936,
            'dog': 22885,
            'bird': 35586
        }
        target_dict = get_class_dictionary(dataset)
        numeral_stats = {k:stats[k] for k,v in target_dict.items()}
    else:
        target_dict = get_class_dictionary(dataset, include_background_class=False)
        return [2. for k in target_dict.keys()]

    return list(calc_class_weights(numeral_stats).values())
        

def calc_class_weights(stats):
    tot = sum(stats.values())
    mean = tot/len(stats)
    return {k: mean / v for k,v in stats.items()}
    

