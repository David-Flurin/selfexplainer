import torch
import pytorch_lightning as pl
import os

from torch import nn, softmax
from torch.optim import Adam
from pathlib import Path

from copy import deepcopy

import pickle

#from torchviz import make_dot

from utils.helper import get_class_dictionary, get_filename_from_annotations, get_targets_from_annotations, extract_masks, Distribution, get_targets_from_segmentations, LogitStats
from utils.image_display import save_all_class_masks, save_mask, save_masked_image, save_background_logits, save_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss, mask_similarity_loss, weighted_loss, bg_loss, background_activation_loss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics
from utils.weighting import softmax_weighting

import GPUtil
from matplotlib import pyplot as plt

class BaseModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, weighting_koeff=1, pretrained=False, use_similarity_loss=False, similarity_regularizer=1.0, use_entropy_loss=False, use_weighted_loss=False,
    use_mask_area_loss=True, use_mask_variation_loss=True, mask_variation_regularizer=1.0, ncmask_total_area_regularizer=0.3, mask_area_constraint_regularizer=1.0, class_mask_min_area=0.04, 
                 class_mask_max_area=0.3, mask_total_area_regularizer=0.1, save_masked_images=False, use_perfect_mask=False, count_logits=False, save_masks=False, save_all_class_masks=False, 
                 gpu=0, profiler=None, metrics_threshold=-1.0, save_path="./results/", objective='classification', class_loss='bce', frozen=False, freeze_every=20, background_activation_loss=False):

        super().__init__()

        self.gpu = gpu
        self.profiler = profiler

        self.learning_rate = learning_rate
        self.weighting_koeff = weighting_koeff

        self.frozen = None
        self.dataset = dataset
        self.num_classes = num_classes

        self.use_similarity_loss = use_similarity_loss
        self.similarity_regularizer = similarity_regularizer
        self.use_entropy_loss = use_entropy_loss
        self.use_weighted_loss = use_weighted_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer

        self.use_background_activation_loss = background_activation_loss

        self.save_path = save_path
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks

        self.objective = objective

        self.test_background_logits = []
        self.class_loss = class_loss

        #self.automatic_optimization = False
        self.frozen = frozen
        self.freeze_every = freeze_every

        #DEBUG
        self.i = 0.
        self.use_perfect_mask = use_perfect_mask
        self.count_logits = count_logits

        self.global_image_mask = None
        self.global_object_mask = None
        self.first_of_epoch = True
        self.same_images = {}

        if self.dataset == 'TOY':
            self.f_tex_dist = Distribution()
            self.b_text_dist = Distribution()
            self.shapes_dist = Distribution()

        self.setup_losses(class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)


    def setup_losses(self, class_mask_min_area, class_mask_max_area):
        if self.class_loss == 'ce':
            self.classification_loss_fn = nn.CrossEntropyLoss()
        elif self.class_loss == 'bce':
            self.classification_loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'Classification loss argument {self.class_loss} not known')
        

        self.total_variation_conv = TotalVariationConv()
        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area, gpu=self.gpu)


    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset == "COLOR":
            self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

        if self.count_logits:
            self.logit_stats = {'image': LogitStats(self.num_classes)}
            if self.use_similarity_loss:
                self.logit_stats['object'] = LogitStats(self.num_classes)
            if self.use_entropy_loss:
                self.logit_stats['background'] = LogitStats(self.num_classes)

    def forward(self, image, targets, perfect_mask = None):
        output = {}
        output['image'] = self._forward(image, targets)

        i_mask = output['image'][1]
        if perfect_mask != None:
            i_mask = perfect_mask
        
        if self.use_similarity_loss:
            masked_image = i_mask.unsqueeze(1) * image
            output['object'] = self._forward(masked_image, targets, frozen=self.frozen)
        
        if self.use_entropy_loss:   
            target_mask_inversed = torch.ones_like(i_mask) - i_mask
            if image.dim() > 3:
                target_mask_inversed = target_mask_inversed.unsqueeze(1)
            inverted_masked_image = target_mask_inversed * image
            # from matplotlib import pyplot as plt
            # fig = plt.figure(figsize=(10,10))
            # for b in range(image.size()[0]):
            #     fig.add_subplot(b+1,3,b*3+1)
            #     plt.imshow(image[b].detach().transpose(0,2))
            #     fig.add_subplot(b+1,3,b*3+2)
            #     plt.imshow(i_mask[b].detach().transpose(0,1))
            #     fig.add_subplot(b+1,3,b*3+3)
            #     plt.imshow(inverted_masked_image[b].detach().transpose(0,2))
            # plt.show()
            output['background'] = self._forward(inverted_masked_image, targets, frozen=self.frozen)
            
        return output

    def _forward(self, image, targets, frozen=False):
        if frozen:
            segmentations = self.frozen(image)
        else:
            segmentations = self.model(image) # [batch_size, num_classes, height, width]
        

        target_mask, non_target_mask = extract_masks(segmentations, targets, gpu=self.gpu) # [batch_size, height, width]

        
        weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
        logits = weighted_segmentations.sum(dim=(2,3))
        #logits = segmentations.mean((2,3))
        

        return segmentations, target_mask, non_target_mask, logits

    def measure_weighting(self, segmentations):
        with self.profiler.profile("softmax_weighing"):
            weighted_segmentations = softmax_weighting(segmentations, self.weighting_koeff)
            logits = weighted_segmentations.sum(dim=(2,3))
        return logits
        
    def training_step(self, batch, batch_idx):
        #GPUtil.showUtilization()
        image, seg, annotations = batch
        targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)

        # bb = image.size()[0]
        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # for b in range(bb):
        #     fig.add_subplot(1, bb, b+1)
        #     plt.imshow(image[b].permute(1,2,0))
        # plt.show()

        '''
        for b in range(image.size()[0]):
            s = image[b].sum().item()
            if s in self.same_images.keys():
                self.same_images[s] += 1
            else:
                self.same_images[s] = 1
        '''

        # if self.first_of_epoch:
        #     from matplotlib import pyplot as plt
        #     fig = plt.figure()
        #     batch_size = image.size()[0]
        #     for b in range(batch_size):
        #         fig.add_subplot(1, batch_size, b+1)
        #         plt.imshow(image[b].permute(1,2,0))
        #     plt.show()
        #     self.first_of_epoch = False
        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(10, 5))
        # fig.add_subplot(1,3,1)
        # plt.imshow(image[0])
        # fig.add_subplot(1,3,2)
        # plt.imshow(targets[0][0])
        # fig.add_subplot(1,3,3)
        # plt.imshow(targets[0][1])
        # plt.show()
        # if self.dataset == 'TOY':
        #     for a in annotations:
        #         for obj in a['objects']:
        #             self.shapes_dist.update(obj[0])
        #             self.f_tex_dist.update(obj[1])
        #     self.b_text_dist.update(a['background'])
        
        if self.frozen and self.i % self.freeze_every == 0 and (self.use_similarity_loss or self.use_entropy_loss):
           self.frozen = deepcopy(self.model)
           for _,p in self.frozen.named_parameters():
               p.requires_grad_(False)
        
        if self.use_perfect_mask:
            output = self(image, target_vector, torch.max(targets, dim=1)[0])
        else:
            output = self(image, target_vector)

        
        #print(output['image'][3])
        #print(target_vector)

        if self.use_entropy_loss:
            self.test_background_logits.append(output['background'][3].sum().item())

        #GPUtil.showUtilization()
        # perfect_mask = torch.max(targets, dim=1)[0].unsqueeze(1)
        # masked_img = image*perfect_mask
        # inv_masked_img = image*(torch.ones_like(perfect_mask) - perfect_mask)
        # output1 = self(masked_img, target_vector)
        # output2 = self(inv_masked_img, target_vector)
        # # from matplotlib import pyplot as plt
        # # fig = plt.figure()
        # # bb = image.size()[0]
        # # for b in range(image.size()[0]):
        # #     fig.add_subplot(bb, 5, (b*5)+1)
        # #     plt.imshow(image[b].permute(1,2,0))
        # #     fig.add_subplot(bb, 5, (b*5)+2)
        # #     plt.imshow(perfect_mask[b].permute(1,2,0))
        # #     fig.add_subplot(bb, 5, (b*5)+3)
        # #     plt.imshow((torch.ones_like(perfect_mask) - perfect_mask)[b].permute(1,2,0))
        # #     fig.add_subplot(bb, 5, (b*5)+4)
        # #     plt.imshow(output1['image'][1][b].detach(), vmin=0, vmax=1)
        # #     fig.add_subplot(bb, 5, (b*5)+5)
        # #     plt.imshow(output2['image'][1][b].detach(), vmin=0, vmax=1)
        # # plt.show()
        # mask_loss = self.classification_loss_fn(output1['image'][3], target_vector)
        # inv_mask_loss = bg_loss(output2['image'][0], use_softmax=True)
        # self.log('mask_loss', mask_loss.item())
        # self.log('inv_mask_loss', inv_mask_loss.item())
        # loss =  mask_loss + inv_mask_loss 


        if self.objective == 'classification':
            classification_loss_initial = self.classification_loss_fn(output['image'][3], target_vector)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)
        elif self.objective == 'segmentation':
            # if self.i > 800:
            #     from matplotlib import pyplot as plt
            #     fig = plt.figure(figsize=(10, 5))
            #     fig.add_subplot(2,2,1)
            #     plt.imshow(output['image'][0][0][0].detach())
            #     fig.add_subplot(2,2,2)
            #     plt.imshow(output['image'][0][0][1].detach())
            #     fig.add_subplot(2,2,3)
            #     plt.imshow(targets[0][0])
            #     fig.add_subplot(2,2,4)
            #     plt.imshow(targets[0][1])
            #     plt.show()
            targets = targets.to(torch.float64)
            classification_loss_initial = self.classification_loss_fn(output['image'][0], targets)
        else:
            raise ValueError('Unknown objective')
        
        #print(classification_loss_initial)

        classification_loss = classification_loss_initial
        self.log('classification_loss', classification_loss)
        #if classification_loss.item() > 0.5 and self.i > 20 and self.i % 2 == 0:
        # b_s,_,_,_ = image.size()
        # for b in range(2):
        #     f, axes = plt.subplots(3,8) 
        #     f.set_size_inches(12,8)
        #     for s in range(7):
        #         axes[0][s].imshow(output['image'][0][b][s].detach(), vmin=-5, vmax=5)
        #         axes[1][s].imshow(output['object'][0][b][s].detach(),vmin=-5, vmax=5)
        #     axes[0][7].imshow(output['image'][1][b].detach(), vmin=0, vmax=1)
        #     axes[1][7].imshow(output['object'][1][b].detach(), vmin=0, vmax=1)
        #     axes[2][0].imshow(image[b].T)
        # plt.show()

        loss = classification_loss
        
        #loss = torch.zeros(1, device=image.device, requires_grad=True)

        
        obj_back_loss = torch.zeros((1), device=loss.device)
        if self.use_similarity_loss:
            similarity_loss = self.similarity_regularizer * mask_similarity_loss(output['image'][1], output['object'][1])
            #similarity_loss = self.classification_loss_fn(output['object'][3], target_vector)
            self.log('similarity_loss', similarity_loss)

            obj_back_loss += similarity_loss

        if self.use_entropy_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = entropy_loss(output['background'][3])
            elif self.bg_loss == 'distance':
                if self.class_loss == 'ce':
                    background_entropy_loss = bg_loss(output['background'][0])
                else:
                    background_entropy_loss = bg_loss(output['background'][0])
            self.log('background_entropy_loss', background_entropy_loss)
            obj_back_loss += background_entropy_loss # Entropy loss is negative, so is added to loss here but actually its subtracted


        

        mask_loss = torch.zeros((1), device=loss.device)
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(output['image'][1])) #+ self.total_variation_conv(s_mask))
            mask_loss += mask_variation_loss

        if self.use_mask_area_loss:
            #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], target_vector)) #+ self.class_mask_area_loss_fn(output['object'][0], target_vector))
            mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean()) #+ output['object'][1].mean())
            #mask_area_loss += self.ncmask_total_area_regularizer * (output['image'][2].mean()) #+ output['object'][2].mean())
            self.log('mask_area_loss', mask_area_loss)
            mask_loss += mask_area_loss


        if self.use_similarity_loss or self.use_entropy_loss or self.use_mask_variation_loss or self.use_mask_area_loss:
            if self.use_weighted_loss:
                loss = weighted_loss(loss, obj_back_loss + mask_loss, 2, 0.2)
            else:
                loss = loss + obj_back_loss + mask_loss

        if self.use_background_activation_loss:
            bg_logits_loss = background_activation_loss(output['image'][1])
            self.log('bg_logits_loss', bg_logits_loss)
            loss += weighted_loss(loss, bg_logits_loss, 2, 0.1)
        
        
        self.i += 1.
        self.log('iterations', self.i)

        self.log('loss', float(loss))
       
        self.train_metrics(output['image'][3], target_vector)

        #DEBUG

        #Save min and max logits
        if self.count_logits:
            for k, v in output.items():
                self.logit_stats[k].update(v[3])

        # self.global_image_mask = output['image'][1]
        # self.global_object_mask = output['object'][1]
                
        #GPUtil.showUtilization()  
        #d = make_dot(loss, params=dict(self.model.named_parameters())) 
        #d.render('backward_graph_unfrozen', format='png')  
        # output['image'][1].retain_grad()
        # output['object'][1].retain_grad()

        # o = self.optimizers()
        # self.manual_backward(loss)
        # o.step()
        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute(), prog_bar=(self.dataset=='TOY'))
        self.train_metrics.reset()
        '''
        self.f_tex_dist.print_distribution()
        self.b_text_dist.print_distribution()
        self.shapes_dist.print_distribution()
        '''
        
        for g in self.trainer.optimizers[0].param_groups:
            self.log('lr', g['lr'], prog_bar=True)

        self.first_of_epoch = True

        #print("Checking for same images")
        #print("Size of dict", len(self.same_images))
        #for k,v in self.same_images.items():
        #    if v > 1:
        #        print(k, v)

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)
        
        self.frozen = deepcopy(self.model)
        for _,p in self.frozen.named_parameters():
            p.requires_grad_(False)
        
        output = self(image, targets)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_initial = self.classification_loss_fn(output['image'][3], labels)
            #classification_loss_object = self.classification_loss_fn(o_logits, labels)
            #classification_loss_background = self.classification_loss_fn(b_logits, labels)
        else:
            classification_loss_initial = self.classification_loss_fn(output['image'][3], targets)
            #classification_loss_object = self.classification_loss_fn(o_logits, targets)
            #classification_loss_background = self.classification_loss_fn(b_logits, targets)

        classification_loss = classification_loss_initial #+ classification_loss_object + classification_loss_background

        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            loss += similarity_loss 

        if self.use_entropy_loss:
            background_entropy_loss = entropy_loss(output['background'][3])
            self.log('background_entropy_loss', background_entropy_loss)
            loss += background_entropy_loss
        # if self.use_mask_variation_loss:
        #     mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
        #     loss += mask_variation_loss

        if self.use_mask_area_loss:
            #mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(output['image'][0], targets) + self.class_mask_area_loss_fn(output['object'][0], targets))
            mask_area_loss = self.mask_total_area_regularizer * (output['image'][1].mean() + output['object'][1].mean())
            #mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            self.log('mask_area_loss', mask_area_loss)
            loss += mask_area_loss

        # if self.use_mask_coherency_loss:
        #     mask_coherency_loss = (t_mask - s_mask).abs().mean()
        #     loss += mask_coherency_loss

        self.log('val_loss', loss)
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.valid_metrics(output['object'][3], labels)
        else:
            self.valid_metrics(output['object'][3], targets) 
   
    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, seg, annotations = batch
        targets = get_targets_from_segmentations(seg, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu, include_background_class=False)
        target_vector = get_targets_from_annotations(annotations, dataset=self.dataset, num_classes=self.num_classes, gpu=self.gpu)
        output = self(image, target_vector)

        

        # from matplotlib import pyplot as plt
        # plt.imshow(image[0].permute(1,2,0))
        # plt.show()
        

        # from matplotlib import pyplot as plt
        # fig = plt.figure(figsize=(10, 5))
        # fig.add_subplot(1,3,1)
        # plt.imshow(image[0])
        # fig.add_subplot(1,3,2)
        # plt.imshow(output['image'][0][0][0])
        # fig.add_subplot(1,3,3)
        # plt.imshow(output['image'][0][0][1])
        # plt.show()

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, output['image'][1], filename, self.dataset)
            filename = Path(self.save_path) / "inverse_masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            inverse = torch.ones_like(output['image'][1]) - output['image'][1]
            save_masked_image(image, inverse, filename, self.dataset)

            filename = Path(self.save_path) / "images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_image(image, filename, self.dataset)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            for k, v in output.items():
                save_mask(v[1], Path(self.save_path) / f'masks_{k}_pass' / filename, self.dataset)


        # if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
        #     filename = self.save_path / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
        #     save_all_class_masks(image, t_seg, filename)



        classification_loss = self.classification_loss_fn(output['image'][3], target_vector)
        # print(classification_loss)
        # print(output['image'][3])
        # print(target_vector)
        loss = classification_loss
        if self.use_similarity_loss:
            similarity_loss = mask_similarity_loss(output['image'][1], output['object'][1])
            loss += similarity_loss

        if self.use_entropy_loss:
            if self.bg_loss == 'entropy':
                background_entropy_loss = entropy_loss(output['background'][3])
            elif self.bg_loss == 'distance':
                if self.class_loss == 'ce':
                    background_entropy_loss = bg_loss(output['background'][0])
                else:
                    background_entropy_loss = bg_loss(output['background'][0])
            loss += background_entropy_loss

        os.makedirs(os.path.dirname(Path(self.save_path) / "test_losses" / filename), exist_ok=True)
        with open(Path(self.save_path) / "test_losses" / filename, 'w') as f:
            f.write(f'classification loss: {classification_loss}\n')
            if self.use_similarity_loss:
                f.write(f'similarity loss: {similarity_loss}\n')
            if self.use_entropy_loss:
                f.write(f'background loss: {background_entropy_loss}\n')

        '''
        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
            mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            loss += mask_area_loss

        if self.use_mask_coherency_loss:
            mask_coherency_loss = (t_mask - s_mask).abs().mean()
            loss += mask_coherency_loss
        '''

        self.log('test_loss', loss)

        



        self.test_metrics(output['image'][3], target_vector)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()
        #save_background_logits(self.test_background_logits, Path(self.save_path) / 'plots' / 'background_logits.png')


        #DEBUG
        if self.count_logits and self.save_path:
            dir = self.save_path + '/logit_stats'
            os.makedirs(dir)
            class_dict = get_class_dictionary(self.dataset)
            for k,v in self.logit_stats.items():
                v.plot(dir + f'/{k}.png', list(class_dict.keys()))


    # def configure_optimizers(self):
    #     return Adam(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=50, threshold=0.001, min_lr=1e-6)
        lr_scheduler_config = {
        "scheduler": lr_scheduler,
        "interval": "epoch",
        "frequency": 1,
        "monitor": "loss",
        "strict": True,
        "name": None,
        }
        return {'optimizer': optim, 'lr_scheduler': lr_scheduler_config}

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('frozen'):
                del checkpoint['state_dict'][k]


