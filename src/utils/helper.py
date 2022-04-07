import torch

def get_targets_from_annotations(annotations, dataset, include_background_class=False, gpu=0, toy_target='texture'):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")
    
    if dataset == "VOC":
        target_dict = get_target_dictionary(include_background_class)
        objects = [item['annotation']['object'] for item in annotations]

        batch_size = len(objects)
        target_vectors = torch.full((batch_size, 20), fill_value=0.0, device=device)
        for i in range(batch_size):
            object_names = [item['name'] for item in objects[i]]

            for name in object_names:
                index = target_dict[name]
                target_vectors[i][index] = 1.0

    elif dataset == "COCO":
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 91), fill_value=0.0, device=device)
        for i in range(batch_size):
            targets = annotations[i]['targets']
            for target in targets:
                target_vectors[i][target] = 1.0

    elif dataset == "CUB":
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 200), fill_value=0.0, device=device)
        for i in range(batch_size):
            target = annotations[i]['target']
            target_vectors[i][target] = 1.0

    elif dataset == "TOY":
        target_dict = get_toy_target_dictionary(include_background_class=False, toy_target=toy_target)
        batch_size = len(annotations)
        target_vectors = torch.full((batch_size, 8), fill_value=0.0, device=device)
        for i in range(batch_size):
            targets = annotations[i]['objects']
            for obj in targets:
                if toy_target == 'texture':
                    name = obj[1]
                else:
                    name = obj[0]
                index = target_dict[name]
                target_vectors[i][index] = 1.0

    return target_vectors

# Only returns 1 filename, not an array of filenames
# Ônly used with batch size 1
def get_filename_from_annotations(annotations, dataset):
    if dataset == "VOC":
        filename = annotations[0]['annotation']['filename']

    elif dataset == "COCO":
        filename = annotations[0]['filename']

    elif dataset == "CUB":
        filename = annotations[0]['filename']

    else:
        raise Exception("Unknown dataset: " + dataset)

    return filename

def get_target_dictionary(include_background_class):
    if include_background_class:
        target_dict = {'background' : 0, 'aeroplane' : 1, 'bicycle' : 2, 'bird' : 3, 'boat' : 4, 'bottle' : 5, 'bus' : 6, 'car' : 7, 
                'cat' : 8, 'chair' : 9, 'cow' : 10, 'diningtable' : 11, 'dog' : 12, 'horse' : 13, 'motorbike' : 14, 'person' : 15, 
                'pottedplant' : 16, 'sheep' : 17, 'sofa' : 18, 'train' : 19, 'tvmonitor' : 20}
    else:
        target_dict = {'aeroplane' : 0, 'bicycle' : 1, 'bird' : 2, 'boat' : 3, 'bottle' : 4, 'bus' : 5, 'car' : 6, 
                'cat' : 7, 'chair' : 8, 'cow' : 9, 'diningtable' : 10, 'dog' : 11, 'horse' : 12, 'motorbike' : 13, 'person' : 14, 
                'pottedplant' : 15, 'sheep' : 16, 'sofa' : 17, 'train' : 18, 'tvmonitor' : 19}

    return target_dict

def get_toy_target_dictionary(include_background_class, toy_target):
    if toy_target == 'texture':
        target_dict = {'bubblewrap' : 0, 'forest' : 1, 'fur' : 2, 'moss' : 3, 'paint' : 4, 'rock' : 5, 'wood' : 6
                }
    elif toy_target == 'shape':
        target_dict = {'circle' : 0, 'triangle' : 1, 'square' : 2, 'pentagon' : 3, 'hexagon' : 4, 'octagon' : 5, 'heart' : 6, 'star': 7, 'cross': 8
                }
    else:
        raise ValueError('Target type must be texture or shape')

    if include_background_class:
        target_dict['background'] = len(target_dict.values())
        
    return target_dict

def extract_masks(segmentations, target_vectors, gpu=0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else "cpu")

    batch_size, num_classes, h, w = segmentations.size()

    target_masks = torch.empty(batch_size, h, w, device=device)
    non_target_masks = torch.empty(batch_size, h, w, device=device)
    for i in range(batch_size):
        class_indices = target_vectors[i].eq(1.0)
        non_class_indices = target_vectors[i].eq(0.0)

        target_masks[i] = (segmentations[i][class_indices]).amax(dim=0)
        
        non_target_masks[i] = (segmentations[i][non_class_indices]).amax(dim=0)

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

