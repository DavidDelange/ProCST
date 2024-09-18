import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sedl.sedl_datasets.SEDL import Dataset
from sedl.exceptions import BadConfigurationError
import yaml
import torch


class UTIMAP_DatasetRealGSX(Dataset):
"""Utility mapping dataset with N masks as ground truth

    Sample : RGB image tensor of $[H, W, C]$ (channel last approach)
    Target : Mmasks tensor of $[H, W, N]$ (values in $[0, 1]$),
             where N is the number of labels to train on
    """

    def __init__(self, root_dir, labels, target_type, num_channels, stage, 
                 background=False, transforms=None, visualization=None):
        """Builds the Utility mapping dataset with N masks as ground truth

        Parameters
        ----------
        root_dir : str
            Path to dataset folder
        labels : list
            For instance: manholes, trenches, pipes
        stage : str
            One of either "train", "valid" or "test"
        background : boolean
            Flag to include or not background mask
        transforms : torchvision transforms
            Pipeline for preprocessing image and mask
        """
        if stage not in ('train', 'valid', 'test'):
            raise BadConfigurationError("Stage is not train nor valid nor test.")

        # Instantiate SEDLDataset
        super().__init__(root_dir, [], num_channels, stage)

        # Attributes
        self.root_dir = root_dir 
        self.labels = labels
        self.target_type = target_type
        self.stage = stage 
        self.num_labels = len(labels)
        self.transforms = transforms
        self.visualization = visualization
        self.dataset_name = os.path.split(root_dir)[-1]
        self.background = background

        # Set paths to data
        self.root_stage = os.path.join(self.root_dir, self.stage)
        self.root_sample = os.path.join(self.root_stage, 'samples') #path to stage/samples
        self.root_yaml = os.path.join(self.root_stage, 'yaml') # path to stage/yaml
        if self.target_type is not None:
            self.root_target = os.path.join(self.root_stage, self.target_type) #path to stage/targets
        self.data_sample = os.listdir(self.root_sample)   # list filenames of samples 
        if self.background:
            self.root_background = os.path.join(self.root_stage, 'targets_road_with_gaps')
          
        
    def __len__(self):
        return len(self.data_sample)

    def __getitem__(self, index):
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
        if self.transforms:
            if sample_length:
                for i, t in enumerate(self.transforms.transforms):
                    if 'RandomCropInsideBoundingBox' in str(t):
                        self.transforms.transforms[i].image_dist_meters = sample_length 
                        break      
                    if 'SequentialCropInsideBoundingBox' in str(t):
                        self.transforms.transforms[i].image_dist_meters = sample_length
                        break      
                    if 'OverlapCropInsideBoundingBox' in str(t):
                        self.transforms.transforms[i].image_dist_meters = sample_length
                        break       
 
            transformed_composition = self.transforms(composition)

            # Return sample and target tuple
            # First sample_channels channels correspond to sample, rest correspond to target
            sample = transformed_composition[:sample_channels]
            target = transformed_composition[sample_channels:]

        if self.visualization:
            self.visualize_during_training(sample, target, index)
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
        sample_name = self.get_filename(index)
        sample_path = os.path.join(self.root_sample, sample_name + '.png')
        if self.background:
            background_path = os.path.join(self.root_background, sample_name + '.png')
        
        sample_length = self.get_sample_length(sample_name)

        # Load sample and convert to np.darray
        # TODO: check if .convert RGB to the PIL image is required
        # TODO: check if returning PIL image only (not numpy) works
        sample = [np.expand_dims(self._read_image_as_numpy(sample_path)[:,:,0], axis=-1)]

        if self.background:
            background = np.expand_dims(self._read_image_as_numpy(background_path)[:, :, 0], axis=-1)
            sample.append(background)
        sample = np.concatenate(sample, axis=-1)

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
        targets = np.zeros(shape=(self.sample_shape[0], self.sample_shape[1], self.num_labels), dtype=np.uint8)
        
        if self.target_type is not None:
            target_path = self.data_sample[index]
            target_path = os.path.join(self.root_target, target_path)
            
            # Load sample and convert to np.darray
            # TODO: check if .convert RGB to the PIL image is required
            # TODO: check if returning PIL image only (not numpy) works

            targets = ([self._read_image_as_numpy(target_path)[:, :, label_channel]
                        for label_channel, _ in enumerate(self.labels)])
            
            # Concatenate all masks in a single np.darray, channel last approach
            targets = np.stack(targets, axis=-1)
        
        return targets
    

    def get_filename(self, index):
        return self.data_sample[index].split('.')[-2]
    

    def get_sample_length(self, sample_name):

        yaml_path = os.path.join(self.root_yaml, sample_name + '.yml')
        sample_length = 0
        if os.path.exists(yaml_path):
            with open(yaml_path, 'rb') as f:
                sample_conf = yaml.safe_load(f)
            if (sample_conf.get('dim_x') is not None) and (sample_conf.get('dim_y') is not None):
                sample_length = np.max([sample_conf['dim_x'], sample_conf['dim_y']])        
            else:
                sample_length = 0

        return sample_length


    def get_crops_boundaries(self, index):
        sample, sample_length = self.get_sample(index)
        sample_tensor = torch.from_numpy(np.transpose(sample, (2, 0, 1)))

        for i, t in enumerate(self.transforms.transforms):
            if 'OverlapCropInsideBoundingBox' in str(t):
                self.transforms.transforms[i].image_dist_meters = sample_length
                self.transforms.transforms[i].remapping_mode = True
                out = self.transforms.transforms[i].forward(sample_tensor)

                break

        return out
        
    
    def get_rows_and_cols(self, index):
        
        sample, sample_length = self.get_sample(index)
        
        for i, t in enumerate(self.transforms.transforms):
            if 'SequentialCropInsideBoundingBox' in str(t):
                target_crop_meters = self.transforms.transforms[i].target_crop_meters  
                break
        image_res = np.max(sample.shape) 
        target_crop = int(image_res/sample_length*target_crop_meters)
        h, w = sample.shape[0], sample.shape[1]

        def equal_or_ceil(num):
            if int(num)== num:
                return int(num)
             
            return int(num)+1
        n_rows = equal_or_ceil(h/target_crop)
        n_cols = equal_or_ceil(w/target_crop)    
        
        return n_cols, n_rows
    
    def get_sample_depth(self, sample_name):

        yaml_path = os.path.join(self.root_yaml, sample_name + '.yml')
        sample_depth = 0
        if os.path.exists(yaml_path):
            with open(yaml_path, 'rb') as f:
                sample_conf = yaml.safe_load(f)
            if sample_conf.get('depth') is not None:
                sample_depth = sample_conf['depth']        
            else:
                # Hardcoded for raw-real-gsx-001-inference
                sample_depth = round(float(sample_name.split('_')[-1]) * 0.02 + 0.02, 2)

        return sample_depth        
    

    def get_dataset_yaml(self):
        yaml_files = [file for file in os.listdir(self.root_dir) if file.endswith((".yaml", ".yml"))]
    
        if not yaml_files:
            print("No YAML files found in the specified directory.")
            return None
        
        first_yaml_file = yaml_files[0]
        file_path = os.path.join(self.root_dir, first_yaml_file)
        with open(file_path, 'r') as file:
            try:
                dataset_yaml = yaml.safe_load(file)
                return dataset_yaml
            except yaml.YAMLError as e:
                print(f"Error reading YAML file {file_path}: {e}")
                return None

    def _read_image_as_numpy(self, path_to_image):
        try:
            return np.asarray(Image.open(path_to_image).convert('RGB'))
        except Exception as e:
            print(f"Failed to read image at path: {path_to_image}")
            raise e

    def visualize_during_training(self, sample, target, idx):
        
        # visualize transformed sample and transformed heatmaps for target
        filename = self.get_filename(idx)
        while len(sample.shape)<4:
            sample = sample.unsqueeze(1)
            target = target.unsqueeze(1)
        sample = np.array(sample)
        target = np.array(target)
        for i in range(sample.shape[1]):
            sample_i = sample[:,i,:,:]
            target_i = target[:,i,:,:]
            sample_i = np.transpose(sample_i, (1, 2, 0))
            target_i = np.transpose(target_i, (1, 2, 0))
        
            # Drawing
            fig, ax = plt.subplots(ncols=4, tight_layout=True, 
                        figsize=(36,9), gridspec_kw={'width_ratios': [3, 3, 3, 3]})
            
            ax[0].imshow(sample_i, cmap='gray')#, extent=extent, aspect=aspect)
            ax[0].set_title('Training sample')

            for channel in range(target_i.shape[-1]):
                ax[channel+1].imshow(target_i[:,:,channel]*255, cmap='gray')    
            ax[1].set_title('Target label Manhole')
            ax[2].set_title('Target label Trech')
            ax[3].set_title('Target label Pipe')

            # Saving
            os.makedirs(f'{self.__class__.__name__}/{self.dataset_name}', exist_ok=True)    
            file_path = f'{self.__class__.__name__}/{self.dataset_name}/sample_in_training_{filename}_{i}.png'

            fig.savefig(file_path)
            plt.close('all')


if __name__ == '__main__':
    # import torch
    from torchvision.transforms import Compose
    from torchvision.transforms import ToTensor, Resize
    from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
    # from seescans.processing.transforms_torch import CustomGaussianBlur, CustomRandomContrast, SequentialCropInsideBoundingBox
    from seescans.processing.transforms_torch import RandomCropInsideBoundingBox, BinarizeTargets

    # Testing if dataset is being loaded correctly
    root_dir = '/home/mcla/workspace/data/GPR/processed_real_gsx_4930_150cm_002_bipolar_debug'

    # Expliciting dataset labels
    labels = ['manhole', 'trench', 'pipe']
    
    # Expliciting target type ['targets_distorted', 'targets_distorted_mismatched', 'targets_perfect', 'targets_perfect_mismatched', 'targets_road', 'targets_road_with_gaps']
    target_type = 'targets_distorted_mismatched'
    
    # Expliciting dataset input number of channels
    num_channels = 3

    # Expliciting dataset stage ['train', 'valid', 'test']
    stage = 'test'

    # Expliciting dataset transforms
    transforms_pipeline = Compose(
        [
            ToTensor(),
            # CustomRandomContrast(alpha_range=[0.5, 1.5], p=0.3, num_labels=3),
            RandomRotation(degrees=10, expand=False),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomCropInsideBoundingBox(image_dist_meters=25, target_crop_meters=10, min_percentage=0.3, p=1, stage=stage, verbose=False),
            Resize(size=(512, 512)),
            # CustomGaussianBlur(kernel_size_range=[5, 9], sigma_range=[0.5, 2], p=0.3, num_labels=3),
            BinarizeTargets(len(labels))
        ]
    )
       
    # Instancing data loader class
    ds = UTIMAP_DatasetRealGSX(root_dir=root_dir, 
                                labels=labels, 
                                target_type=target_type, 
                                num_channels=num_channels, 
                                stage=stage, 
                                transforms=transforms_pipeline,
                                visualization=True)
    
    for idx, (sample, target) in enumerate(ds):
        print(f'Processed image {idx}')
        print(f'Sample shape after transforms: {sample.shape}')
        sample_values = torch.unique(sample)
        print(f'Unique values for sample: {sample_values}')
        print(f'Targets shape after transforms: {target.shape}')
        target_values = torch.unique(target)
        print(f'Unique values for targets: {target_values}')