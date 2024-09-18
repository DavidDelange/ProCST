import os.path as osp
from data_handlers.domain_adaptation_dataset import domainAdaptationDataSet
from PIL import Image
import numpy as np
from core.constants import RESIZE_SHAPE
import os
import yaml
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from core.transforms_torch import CustomGaussianBlur,RandomCropInsideBoundingBox, BinarizeTargets, CustomRandomContrast
from torchvision.transforms import InterpolationMode

class SynthDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_image_label_pyramid=False, get_filename=False, get_original_image=False):
        super(SynthDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.domain_resize = RESIZE_SHAPE['synth']
        self.get_image_label_pyramid = get_image_label_pyramid
        self.get_filename = get_filename
        self.get_original_image = get_original_image
        self.id_to_trainid = {1: 1, 2: 2, 3: 3}
        self.num_labels = len(self.id_to_trainid.keys())
        if images_list_path != None:
            self.images_list_file = osp.join(images_list_path, '%s.txt' % set)
            self.img_ids = [image_id.strip().split('.')[0] for image_id in open(self.images_list_file)]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        image, label = self.getitem_seescans(index)

        label_copy, labels_pyramid = None, None 
        if self.get_image_label or self.get_image_label_pyramid:
            if self.get_image_label_pyramid:
                labels_pyramid =  self.GeneratePyramid(label, is_label=True)
                labels_pyramid = [self.convert_to_class_ids(label_scale) for label_scale in labels_pyramid]
            else:
                label = self.convert_to_class_ids(label)
        else: 
            label = None

        scales_pyramid = self.GeneratePyramid(image)
        if self.get_image_label:
            return scales_pyramid, label
        elif self.get_image_label_pyramid:
            return scales_pyramid, labels_pyramid
        else:
            return scales_pyramid if not self.get_filename else scales_pyramid, self.img_ids[index]

    def getitem_seescans(self, index):
        """Get image and mask from annotation file

        Parameters
        ----------
        index : int
            Index of the annotation file to load the sample and ground
            truth (image and mask)

        Returns
        -------
        image : numpy.ndarray
            RGB image tensor of $[H, W, C]$ (channel last approach)
        mask : numpy.ndarray
            Binary mask tensor of $[H, W]$
        """
        # Get sample and target by index
        sample, sample_length = self.get_sample(index)     
        self.sample_shape = sample.shape
        target = self.get_target(index)
        sample_channels = sample.shape[-1]

        # Concatenate sample and targets
        # Shape is [num_channels_sample + num_channels_target, H, W]
        composition = np.concatenate((sample, target), axis=2)

        # Apply transforms if set
        if self.get_original_image:
            pass
        else:
            transforms_pipeline = Compose(
            [
                ToTensor(),
                # CustomRandomContrast(alpha_range=[0.5, 1.5], p=0.3, num_labels=3),
                RandomRotation(degrees=10, expand=False),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomCropInsideBoundingBox(image_dist_meters=sample_length, target_crop_meters=10, min_percentage=0.3, p=1, stage='train', verbose=False),
                Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST),
                # CustomGaussianBlur(kernel_size_range=[5, 9], sigma_range=[0.5, 2], p=0.3, num_labels=3),
                BinarizeTargets(self.num_labels)
            ])            
            transformed_composition = transforms_pipeline(composition)

            # Return sample and target tuple
            # First sample_channels channels correspond to sample, rest correspond to target
            sample = transformed_composition[:sample_channels]
            target = transformed_composition[sample_channels:]

            sample = F.to_pil_image(sample.cpu())
            target = F.to_pil_image(target.cpu())
        return sample, target
        

    def get_sample(self, index):
        """Load the input image as RGB tensor.

        Parameters
        ----------
        index : int
            Index of the annotation file to load the sample

        Returns
        -------
        numpy.ndarray
            RGB image tensor of $[H, W, C]$ (channel last approach)
        """
        # Get path to sample
        sample_name = self.img_ids[index] + '.png'
        sample_path = osp.join(self.root, "%s/images/%s" % (self.set, sample_name))
        
        sample_length = self.get_sample_length(index)

        # Load sample and convert to np.darray
        # TODO: check if .convert RGB to the PIL image is required
        # TODO: check if returning PIL image only (not numpy) works
        sample = np.expand_dims(self._read_image_as_numpy(sample_path)[:,:,0], axis=-1)

        # print(sample.shape)
        return sample, sample_length

    def get_target(self, index):
        """Load ground truth binary mask

        Each ground truth has 1 defect per channel (RGB)

        Parameters
        ----------
        index : int
            Index of the annotation file to load the target

        Returns
        -------
        numpy.ndarray
            Binary mask tensor of $[H, W, C]$

        Dataset for gt consists in Name_image_0
                                   Name_image_1
                                   Name_image_2
        where 0,1,2 are the labels number 
        """
        targets = np.zeros(shape=(self.sample_shape[0], self.sample_shape[1],  self.num_labels), dtype=np.uint8)
        
        if self.get_image_label is not False:
            target_name = self.img_ids[index] + '.png'
            target_path = osp.join(self.root, "%s/labels/%s" % (self.set, target_name))
            
            # Load sample and convert to np.darray
            # TODO: check if .convert RGB to the PIL image is required
            # TODO: check if returning PIL image only (not numpy) works

            targets = ([self._read_image_as_numpy(target_path)[:, :, label_channel]
                        for label_channel, _ in enumerate(self.id_to_trainid.keys())])
            
            # Concatenate all masks in a single np.darray, channel last approach
            targets = np.stack(targets, axis=-1)
        
        return targets

    def get_sample_length(self, index):
        name = self.img_ids[index] + '.yaml'
        yaml_path = osp.join(self.root, "%s/yaml/%s" % (self.set, name))
        sample_length = 0
        if os.path.exists(yaml_path):
            with open(yaml_path, 'rb') as f:
                sample_conf = yaml.safe_load(f)
            if (sample_conf.get('dim_x') is not None) and (sample_conf.get('dim_y') is not None):
                sample_length = np.max([sample_conf['dim_x'], sample_conf['dim_y']])        
            else:
                sample_length = 0

        return sample_length

    def _read_image_as_numpy(self, path_to_image):
        try:
            return np.asarray(Image.open(path_to_image).convert('RGB'))
        except Exception as e:
            print(f"Failed to read image at path: {path_to_image}")
            raise e


if __name__ == '__main__':
    root = '/home/ddel/workspace/data/seescans/synth'
    images_list_path = '/home/ddel/workspace/repositories/ProCST/dataset/synth_list'
    scale_factor = 0.5
    num_scales = 2
    curr_scale = 0
    set = 'train'
    get_image_label=False
    get_image_label_pyramid=True
    get_filename=False
    get_original_image=False
    dataset = SynthDataSet(root, 
                        images_list_path, 
                        scale_factor, 
                        num_scales, 
                        curr_scale, 
                        set, 
                        get_image_label,
                        get_image_label_pyramid,
                        get_filename, 
                        get_original_image)
    print(dataset[0])
    print(len(dataset))