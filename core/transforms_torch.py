"""
Each transform is define as subclass that will inherit from the torch.nn.Module parent class
and uses tensor operations with the torch package.

Usage: Transformations are built using Compose from torchvision.transforms. 
In order to script the transformations, use torch.nn.Sequential as below.

Important considerations:
-Make sure to use only scriptable transformations, i.e. that work with torch.Tensor,
does not require lambda functions or PIL.Image.

"""

import torch
import numpy as np
import random
from torchvision.transforms import RandomAffine, InterpolationMode, ToTensor, Resize, RandomEqualize, RandomAutocontrast, GaussianBlur
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import glob
import torch.nn.functional as F
from torchvision.utils import save_image

class CustomRandomAffine(torch.nn.Module):
    """ scale:  (a, b), then scale is randomly sampled from the range a, b.
    interpolation, if input is Tensor, only NEAREST, BILINEAR are supported."""
    def __init__(self, zoom_min, zoom_max, p=0):
        super().__init__()
        
        self.scale = (zoom_min, zoom_max)
        self.p = p
        
    def forward(self, image):
        if random.random() < self.p:
            image = RandomAffine(degrees = 0, translate=None, scale= self.scale, shear=None, 
                                            interpolation=InterpolationMode.NEAREST, fill=0, center=None)(image)
        return image
        

class CustomResize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.NEAREST):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, image):
        return Resize(self.size, interpolation=self.interpolation)(image)


class NormalizeImage(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, image):
        # Ensuring no zero-division runtime error
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]
        
        sample_max = max(torch.max(torch.abs(sample)), 1e-8)
        sample_norm = sample / sample_max
        return torch.cat((sample_norm, targets), dim=0)


class NormalizeImage_to_dataset(torch.nn.Module):
    """ Performs normalization of data according to 
    the global values of a dataset (not sample by sample). """
    def __init__(self, min_dataset, max_dataset, num_labels):
        super().__init__()
        self.min = min_dataset
        self.max = max_dataset
        self.num_labels = num_labels

    def forward(self, image):
        # Ensuring no zero-division runtime error
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]
       
        sample_norm = (sample - self.min)/(self.max  - self.min)
        return torch.cat((sample_norm, targets), dim=0)
    

class StandardizeImage(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, image):
        # Ensuring no zero-division runtime error
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]

        sample_stand = (sample - torch.mean(sample))/(torch.std(sample)+1e-10)
        return torch.cat((sample_stand, targets), dim=0)

  
class StandardizeImage_to_dataset(torch.nn.Module):
    """ Performs standardization of data according to 
    the global values of a dataset (not sample by sample). """
    def __init__(self, mean_dataset, std_dataset, num_labels):
        super().__init__()
        self.mean = mean_dataset
        self.std = std_dataset
        self.num_labels = num_labels

    def forward(self, image):
        # Ensuring no zero-division runtime error       
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]

        sample_stand = (sample - self.mean)/(self.std + 1e-10)        
        return torch.cat((sample_stand, targets), dim=0)
    

class CustomRandomEqualize(torch.nn.Module):
    """Equalize the histogram of the given image randomly with a given probability.

    Args:
        p (float): probability of the image being equalized. Default value is 0.5
        num_labels (int): number of dedicated channels for labels
    """

    def __init__(self, p=0.5, num_labels=3):
        super().__init__()
        self.p = p
        self.num_labels = num_labels

    def forward(self, image):        
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]
        sample_eq = RandomEqualize(self.p)(sample.to(torch.uint8))
        return torch.cat((sample_eq, targets), dim=0)

class CustomRandomAutocontrast(torch.nn.Module):
    """
    Args:
        p (float): probability of the image being equalized. Default value is 0.5
        num_labels (int): number of dedicated channels for labels
    """

    def __init__(self, p=0.5, num_labels=3):
        super().__init__()
        self.p = p
        self.num_labels = num_labels

    def forward(self, image):        
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]
        sample_eq = RandomAutocontrast(self.p)(sample)
        return torch.cat((sample_eq, targets), dim=0) 
    
class CustomRandomContrast(torch.nn.Module):
    """
    Args:
        alpha_range (list): If alpha < 1, it will decrease the contrast, making the image more uniform and less vibrant.
        If alpha > 1, it will increase the contrast, making the dark areas darker and the bright areas brighter.
        p (float): probability of the image being equalized. Default value is 0.5
        num_labels (int): number of dedicated channels for labels
    """
    def finddata(self, image):
        # Find the indices where values are greater than the threshold        
        indices = torch.nonzero(image > self.threshold, as_tuple=False)
        if indices.numel() == 0:
            return None
        return indices
        
    def __init__(self, alpha_range, threshold=0.3, p=0.5, num_labels=3, verbose=False):
        super().__init__()
        # Sanity check
        assert isinstance(alpha_range, list), 'should be a list (range)'
        self.alpha_range = alpha_range
        self.threshold = threshold
        self.p = p
        self.num_labels = num_labels
        self.verbose = verbose

    def forward(self, image):   
        if random.random() < self.p:          
            sample = image[:-self.num_labels, :, :]
            targets = image[-self.num_labels:, :, :]
            
            indices = self.finddata(sample)            
            if indices is not None:
                sample_contrast = sample.clone()  
                mean = sample_contrast[indices[:, 0], indices[:, 1], indices[:, 2]].mean()
                alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
                sample_contrast[indices[:, 0], indices[:, 1], indices[:, 2]] = (sample_contrast[indices[:, 0], indices[:, 1], indices[:, 2]] - mean) * alpha + mean
                sample_contrast[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.clamp(sample_contrast[indices[:, 0], indices[:, 1], indices[:, 2]], 0, 1)
            else:
                sample_contrast = sample.clone()
                mean = sample_contrast.mean()
            
            if self.verbose:
                sample_plot_min = sample.clone()                  
                sample_plot_min[indices[:, 0], indices[:, 1], indices[:, 2]] = (sample_plot_min[indices[:, 0], indices[:, 1], indices[:, 2]] - mean) * self.alpha_range[0] + mean
                sample_plot_min[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.clamp(sample_plot_min[indices[:, 0], indices[:, 1], indices[:, 2]], 0, 1)
            
                sample_plot_max = sample.clone()                  
                sample_plot_max[indices[:, 0], indices[:, 1], indices[:, 2]] = (sample_plot_max[indices[:, 0], indices[:, 1], indices[:, 2]] - mean) * self.alpha_range[1] + mean
                sample_plot_max[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.clamp(sample_plot_max[indices[:, 0], indices[:, 1], indices[:, 2]], 0, 1)
                
                visualize_contrast(sample, sample_plot_min, sample_plot_max)

            return torch.cat((sample_contrast, targets), dim=0) 
        return image
    
class RandomLevelsPreset(torch.nn.Module):
    """
    Args:
        levels_range (list)
        gamma (float)
        num_labels (int)
    """        
    def __init__(self, levels_min_range=[80, 90], levels_max_range=[205, 215], p=1, num_labels=3, verbose=False):
        super().__init__()
        # Sanity check
        assert isinstance(levels_min_range, list), 'should be a list (range)'
        self.levels_min_range = [x / 255.0 for x in levels_min_range]        
        assert isinstance(levels_max_range, list), 'should be a list (range)'
        self.levels_max_range = [x / 255.0 for x in levels_max_range]
        self.p = p
        self.num_labels = num_labels
        self.verbose = verbose

    def forward(self, image):   
        if random.random() < self.p:          
            sample = image[:-self.num_labels, :, :]
            targets = image[-self.num_labels:, :, :]
            
            # Apply input range adjustment
            levels_min = random.uniform(self.levels_min_range[0], self.levels_min_range[1])
            levels_max = random.uniform(self.levels_max_range[0], self.levels_max_range[1])
            
            sample_levels = torch.clamp((sample - levels_min) / (levels_max - levels_min), 0, 1)

            if self.verbose:
                sample_plot_min = sample.clone()    
                sample_plot_min = torch.clamp((sample - self.levels_min_range[0]) / (self.levels_max_range[-1] - self.levels_min_range[0]), 0, 1)

                sample_plot_max = sample.clone()   
                sample_plot_max = torch.clamp((sample - self.levels_min_range[-1]) / (self.levels_max_range[0] - self.levels_min_range[-1]), 0, 1)

                visualize_levels(sample, sample_plot_min, sample_plot_max)

            return torch.cat((sample_levels, targets), dim=0) 
        return image
    
class BipolarPreset(torch.nn.Module):
    """
    Args:
        levels_range (list)
        gamma (float)
        num_labels (int)
    """        
    def __init__(self, num_labels=3, verbose=False):
        super().__init__()
        
        self.num_labels = num_labels
        self.verbose = verbose

    def forward(self, image):   
              
        sample = image[:-self.num_labels, :, :]
        targets = image[-self.num_labels:, :, :]
        
        # Apply input range adjustment
        # sample_bipolar = sample.clone() * 100
        # sample_bipolar = 3.23e-9 - 1.61e-9 * sample + 3e-4 * torch.pow(sample, 2) - 2e-6 * torch.pow(sample, 3) + 9.13e-14 * torch.pow(sample, 4) - 4.92e-16 * torch.pow(sample, 5) + 4.06e-19 * torch.pow(sample, 6)
        
        with open('/home/mcla/workspace/repos/seescans/seescans/processing/colormap_bipolar.txt', 'r') as file:
            first_values = []  # List to store the first values
            
            # Iterate over each line in the file
            for line in file:
                # Split the line by commas to get the three values
                values = line.strip().split(',')
                
                # Assuming each row has three values, add the first value to the list
                if len(values) >= 1:
                    first_values.append(float(values[0]))  # Convert to float if needed
                else:
                    print("Error: Invalid format in line:", line)


        # Convert numpy array to PyTorch tensor
        array_tensor = torch.tensor(first_values)

        # Assuming your HxW tensor is called `tensor_hw`
        # tensor_hw = torch.tensor(...)  # Your HxW tensor here

        # Find the closest index in the numpy array for each value in the HxW tensor
        closest_indices = torch.round(sample.float() * (len(first_values) - 1)).long()

        # Gather the values from the numpy array based on the closest indices
        closest_values = torch.gather(array_tensor, 0, closest_indices.view(-1))

        # Reshape the closest values to match the shape of your HxW tensor
        closest_values = closest_values.view(sample.size())

        # Perform element-wise multiplication
        sample_bipolar = sample * closest_values

        if self.verbose:

            image = np.array(sample)
            if len(image.shape) > 2:
                image = np.transpose(image, (1, 2, 0))

            image_bipolar = np.array(sample_bipolar)
            if len(image_bipolar.shape) > 2:
                image_bipolar = np.transpose(image_bipolar, (1, 2, 0))
            
            # Drawing
            fig0, ax0 = plt.subplots(ncols=1, tight_layout=True, 
                        figsize=(image.shape[0]/100, image.shape[1]/100))

            ax0.imshow(image, cmap='gray')    
            ax0.set_title('Original')

            fig1, ax1 = plt.subplots(ncols=1, tight_layout=True, 
                        figsize=(image_bipolar.shape[0]/100, image_bipolar.shape[1]/100))

            ax1.imshow(image_bipolar, cmap='gray')    
            ax1.set_title('Bipolar preset')
        
            # Saving
            save_path = './test_torchvision_transf'
            os.makedirs(save_path, exist_ok=True)
            files = glob.glob(save_path + '/*.png')
            
            file_path0 = os.path.join(save_path, f'sample_in_training_{str(len(files))}_og.png')
            fig0.savefig(file_path0)
            
            file_path1 = os.path.join(save_path, f'sample_in_training_{str(len(files))}_bipolar.png')
            fig1.savefig(file_path1)
            plt.close('all')
            

        return torch.cat((sample_bipolar, targets), dim=0) 

class CustomGaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.

    Args:
        kernel_size (list): Size of the Gaussian kernel.
        sigma (list): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        Tensor: Gaussian blurred version of the input image.
    """

    def __init__(self, kernel_size_range, sigma_range, p=0.5, num_labels=3, verbose=False):
        super().__init__()
        # Sanity check
        assert isinstance(kernel_size_range, list), 'should be a list (range)'
        assert isinstance(sigma_range, list), 'should be a list (range)'

        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p
        self.num_labels = num_labels
        self.verbose = verbose

    def forward(self, image):   
        if random.random() < self.p:     
            sample = image[:-self.num_labels, :, :]
            targets = image[-self.num_labels:, :, :]

            # set the variables
            kernel_size = random.randrange(self.kernel_size_range[0], self.kernel_size_range[1] + 1, step=2)
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])

            sample_eq = GaussianBlur(kernel_size=kernel_size, sigma=sigma)(sample)

            if self.verbose:
                sample_plot_min = sample.clone()       
                sample_plot_min = GaussianBlur(kernel_size=self.kernel_size_range[0], sigma=self.sigma_range[0])(sample_plot_min)

                sample_plot_max = sample.clone()       
                sample_plot_max = GaussianBlur(kernel_size=self.kernel_size_range[1], sigma=self.sigma_range[1])(sample_plot_max)
                visualize_gaussianblur(sample, sample_plot_min, sample_plot_max)

            return torch.cat((sample_eq, targets), dim=0)   
        return image

class BinarizeTargets(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

    def forward(self, image):
        threshold = 0.5
        if torch.max(image[-self.num_labels:, :, :]) > 1 + 1e-6:
            threshold = 128
        targets = torch.where(image[-self.num_labels:, :, :] > threshold, 1, 0)
        return torch.cat((image[:-self.num_labels, :, :], targets), dim=0)
        
  
class RandomCropInsideBoundingBox(torch.nn.Module):
    def __init__(self, 
                 image_dist_meters=35, 
                 target_crop_meters=10, 
                 sample_dim=1, threshold=0.01, 
                 min_percentage=0.3, 
                 max_iterations=100, 
                 p=1,
                 stage='train',
                 verbose=False):
        
        super().__init__()        
        self.image_dist_meters = image_dist_meters
        self.target_crop_meters = target_crop_meters
        self.sample_dim = sample_dim 
        self.threshold = threshold
        self.min_percentage = min_percentage
        self.max_iterations = max_iterations
        self.p = p
        self.stage=stage
        self.verbose = verbose

    def minboundingbox(self, image):
        # Find the indices where values are greater than the threshold
        # TODO: Artifacts appear at the bottom of the image generated by substance. 
        if self.stage == 'train':
            # # Removing padding of 40 pixel dye to artifacts from substance
            guard_margin = 40

            image[:, :guard_margin, :] = 0
            image[:, -guard_margin:, :] = 0
            image[:, :, :guard_margin] = 0
            image[:, :, -guard_margin:] = 0

            indices = (image > self.threshold).nonzero()
        
        else:
            indices = (image > self.threshold).nonzero()

        # Check if any values greater than the threshold were found
        if indices.size(0) == 0:
            return None
        
        # Calculate the bounds of the rectangle
        min_row = indices[:, 1].min().item()
        min_col = indices[:, 2].min().item()
        max_row = indices[:, 1].max().item()
        max_col = indices[:, 2].max().item()
        
        # Calculate the dimensions of the minimum rectangle
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        
        rect = (min_row, min_col, width, height)
        return rect

    def forward(self, image):
        if random.random() < self.p:            
            # Initialize variables
            cropped_flag = False
            iterations = 0
            max_percentage_so_far = 0
            image_res = np.max(image.shape)                      
            target_crop = int(image_res/self.image_dist_meters*self.target_crop_meters)
            
            rect = self.minboundingbox(image[:self.sample_dim,:,:])

            if rect is None:
                return image
            
            min_row, min_col, rect_width, rect_height = rect
            
            while cropped_flag is False and iterations < self.max_iterations:
                # Randomize the position of the crop within the minimum rectangle
                if (min_row + rect_height - target_crop) > min_row:
                    crop_bottom = np.random.randint(min_row, min_row + rect_height - target_crop)
                else:
                    crop_bottom = min_row
                if (min_col + rect_width - target_crop) > min_col:
                    crop_left = np.random.randint(min_col, min_col + rect_width - target_crop)
                else:
                    crop_left = min_col

                # Perform the crop
                cropped_image = image[:, crop_bottom : crop_bottom + target_crop, crop_left : crop_left + target_crop]                
                cropped_rect = (crop_bottom, crop_left, target_crop, target_crop)

                # Check if the cropped region has a minimum percentage of values greater than threshold
                valid_values = (cropped_image[:self.sample_dim, :, :] > self.threshold).sum().item()
                total_values = cropped_image[:self.sample_dim, :, :].numel()
                percentage_valid_values = valid_values / total_values

                # if self.verbose > 0:
                #     print(f'\tSample with a valid values ratio of {round(percentage_valid_values,2)}')

                if percentage_valid_values >= self.min_percentage:
                    max_percentage_so_far = percentage_valid_values
                    cropped_image_so_far = cropped_image
                    cropped_rect_so_far = cropped_rect
                    cropped_flag = True
                else:
                    if percentage_valid_values > max_percentage_so_far:
                        max_percentage_so_far = percentage_valid_values
                        cropped_image_so_far = cropped_image
                        cropped_rect_so_far = cropped_rect      
                    iterations += 1

            if self.verbose:
                print(f'\tExit after {iterations} iterations with a valid values ratio of {round(max_percentage_so_far, 2)}')
                print(f'\tCrop X: {crop_left}\tCrop Y: {crop_bottom}')
                visualize_boundingbox(image, rect, cropped_image_so_far, cropped_rect_so_far, self.sample_dim)
        else:
            cropped_image_so_far = image 
        
        return cropped_image_so_far

class SequentialCropInsideBoundingBox(torch.nn.Module):
    def __init__(self, 
                 image_dist_meters=25, 
                 target_crop_meters=10, 
                 sample_dim=1,  
                 p=1,
                 stage='test',
                 verbose=False):
        
        super().__init__()        
        self.image_dist_meters = image_dist_meters
        self.target_crop_meters = target_crop_meters
        self.sample_dim = sample_dim 
        self.p = p
        self.stage=stage
        self.verbose = verbose

    
    def forward(self, image):
        if random.random() < self.p:            
            # Initialize variables

            image_res = np.max(image.shape)                      
            target_crop = int(image_res/self.image_dist_meters*self.target_crop_meters)
            h, w = image.shape[1], image.shape[2]      
            
            cropped_images = []
            for x in range(0, h, target_crop):
                for y in range(0, w, target_crop):
                    x_start = x
                    x_end = min(x + target_crop, h)
                    y_start = y
                    y_end = min(y + target_crop, w)
                    crop_zeros = torch.zeros(image.shape[0], target_crop, target_crop)
                    crop_zeros[:, :x_end - x_start, :y_end - y_start]=image[:, x_start:x_end, y_start:y_end]
                    cropped_images.append(crop_zeros)

            stacked_images = torch.stack(cropped_images, dim=1)
           
            if self.verbose:
                visualize_sequential_crop(stacked_images, self.sample_dim)
        else:
            return image
        
        return stacked_images
    
class OverlapCropInsideBoundingBox(torch.nn.Module):
    def __init__(self, 
                 image_dist_meters=25, 
                 target_crop_meters=10,
                 multiplier=.5,
                 data_threshold=35,
                 sample_dim=1,  
                 p=1,
                 stage='test',
                 remapping_mode=False,
                 verbose=False):
        
        super().__init__()        
        self.image_dist_meters = image_dist_meters
        self.target_crop_meters = target_crop_meters
        self.multiplier = multiplier
        self.data_threshold = data_threshold
        self.sample_dim = sample_dim 
        self.p = p
        self.stage=stage
        self.remapping_mode=remapping_mode
        self.verbose = verbose

    
    def forward(self, image):

        image_res = np.max(image.shape)                      
        target_crop = int(image_res/self.image_dist_meters*self.target_crop_meters)
        grid_size = int(target_crop * self.multiplier)
         
        # Pad input tensor
        image = F.pad(image, (target_crop // 2, target_crop // 2, target_crop // 2, target_crop // 2), value=0)

        # Get the dimensions of the input tensor
        height, width = image.shape[-2], image.shape[-1]

        # Calculate the number of squares in each dimension
        num_squares_h = (height + grid_size - 1) // grid_size
        num_squares_w = (width + grid_size - 1) // grid_size

        # Initialize an empty list to store the cropped squares
        double_cropped_squares = []
        boundaries = []
        double_boundaries = []
        # Loop through each square
        for i in range(num_squares_h):
            for j in range(num_squares_w):
                # Calculate the starting indices for the current square
                start_h = i * grid_size
                start_w = j * grid_size

                # Crop the square from the padded tensor
                cropped_square = image[..., start_h:start_h+grid_size, start_w:start_w+grid_size]

                # Check if the cropped square contains non-zero values
                if cropped_square.nonzero().numel() > 0:
                    # Save boundaries
                    boundaries.append((start_h, start_w, grid_size))

                    # Center indices for the new double-sized crops
                    center_h = start_h + grid_size // 2
                    center_w = start_w + grid_size // 2

                    # Adjust starting indices for double-sized crops
                    start_h_double = center_h - target_crop // 2
                    start_w_double = center_w - target_crop // 2

                    # Crop the double-sized square from the padded tensor
                    double_cropped_square = image[..., start_h_double:start_h_double+target_crop,
                                                    start_w_double:start_w_double+target_crop]

                    
                    if double_cropped_square.shape[-1] == target_crop and double_cropped_square.shape[-2] == target_crop:
                        # Count the number of non-zero elements in the crop
                        num_non_zero = torch.count_nonzero(double_cropped_square[0,:,:])

                        # Calculate the total number of elements in the crop
                        total_elements = double_cropped_square[0,:,:].numel()

                        # Calculate the percentage of data inside the crop
                        percentage = (num_non_zero / total_elements) * 100
                        if percentage > self.data_threshold:                           
                            # Append the double-sized cropped square to the list
                            double_cropped_squares.append(double_cropped_square)

                            # Save boundaries for the double-sized crop
                            double_boundaries.append((start_h_double, start_w_double, target_crop))

        # Stack the cropped squares along a new dimension to form the final tensor
        if self.remapping_mode:
            return double_boundaries, [height, width]
        
        return torch.stack(double_cropped_squares, dim=1)

            
def visualize_sequential_crop(image, sample_dim):
    # Saving
    save_path = './test_torchvision_transf'
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + '/*.png')
   
    
    sample = image[:sample_dim,:,:,:].squeeze()
    for i in range(sample.shape[0]): 
        file_path = os.path.join(save_path, f'sample_in_training_{str(len(files))}_bbcropcrop_{str(i)}.png')
        save_image(sample[i] , file_path)
        
        
def visualize_boundingbox(image, bb, cropped_image, cropped_rect, sample_dim):
    
    # visualize transformed sample and transformed heatmaps for target
    image = np.array(image)
    if len(image.shape) > 2:
        image = np.transpose(image, (1, 2, 0))
        sample = image[:, :, :sample_dim]
        # target = image[:, :, sample_dim:]
    
    cropped_image = np.array(cropped_image)
    if len(cropped_image.shape) > 2:
        cropped_image = np.transpose(cropped_image, (1, 2, 0))
        cropped_sample = cropped_image[:, :, :sample_dim]
        cropped_target = cropped_image[:, :, sample_dim:]

    rect = patches.Rectangle((bb[1], bb[0]), bb[2], bb[3],
                             linewidth=2, edgecolor='r', facecolor='none')
    crop_rect = patches.Rectangle((cropped_rect[1], cropped_rect[0]), cropped_rect[2], cropped_rect[3],
                                  linewidth=2, edgecolor='y', facecolor='none')

    # Drawing
    fig, ax = plt.subplots(ncols=3, tight_layout=True, 
                figsize=(20,9), gridspec_kw={'width_ratios': [3, 3, 3]})
    
    ax[0].imshow(sample, cmap='gray')#, extent=extent, aspect=aspect)
    ax[0].add_patch(rect)
    ax[0].add_patch(crop_rect)
    ax[0].set_title('Sample with Minimum Rectangle (red) and Crop (yellow)')

    ax[1].imshow(cropped_sample, cmap='gray')    
    ax[1].set_title('Randomized Crop Inside Minimum Rectangle')

    ax[2].imshow(cropped_target, cmap='gray')    
    ax[2].set_title('Target cropped')

    # Saving
    save_path = './test_torchvision_transf'
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + '/*.png')
    file_path = os.path.join(save_path, f'sample_in_training_{str(len(files))}_bbcrop.png')
    fig.savefig(file_path)
    plt.close('all')

def visualize_contrast(sample, sample_plot_min, sample_plot_max):
    
    sample = np.array(sample)
    if len(sample.shape) > 2:
        image = np.transpose(sample, (1, 2, 0))

    sample_plot_min = np.array(sample_plot_min)
    if len(sample_plot_min.shape) > 2:
        image_min = np.transpose(sample_plot_min, (1, 2, 0))

    sample_plot_max = np.array(sample_plot_max)
    if len(sample_plot_max.shape) > 2:
        image_max = np.transpose(sample_plot_max, (1, 2, 0))
       
    # Drawing
    fig, ax = plt.subplots(ncols=3, tight_layout=True, 
                figsize=(20,9), gridspec_kw={'width_ratios': [3, 3, 3]})
    
    ax[0].imshow(image, cmap='gray')    
    ax[0].set_title('Original contrast')

    ax[1].imshow(image_min, cmap='gray')    
    ax[1].set_title('Minimum contrast selected')

    ax[2].imshow(image_max, cmap='gray')    
    ax[2].set_title('Maximum constrast selected')

    # Saving
    save_path = './test_torchvision_transf'
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + '/*.png')
    file_path = os.path.join(save_path, f'sample_in_training_{str(len(files))}_contrast.png')
    fig.savefig(file_path)
    plt.close('all')

def visualize_gaussianblur(sample, sample_plot_min, sample_plot_max):
    
    sample = np.array(sample)
    if len(sample.shape) > 2:
        image = np.transpose(sample, (1, 2, 0))

    sample_plot_min = np.array(sample_plot_min)
    if len(sample_plot_min.shape) > 2:
        image_min = np.transpose(sample_plot_min, (1, 2, 0))

    sample_plot_max = np.array(sample_plot_max)
    if len(sample_plot_max.shape) > 2:
        image_max = np.transpose(sample_plot_max, (1, 2, 0))
       
    # Drawing
    fig, ax = plt.subplots(ncols=3, tight_layout=True, 
                figsize=(20,9), gridspec_kw={'width_ratios': [3, 3, 3]})
    
    ax[0].imshow(image, cmap='gray')    
    ax[0].set_title('Original contrast')

    ax[1].imshow(image_min, cmap='gray')    
    ax[1].set_title('Minimum blur selected')

    ax[2].imshow(image_max, cmap='gray')    
    ax[2].set_title('Maximum blur selected')

    # Saving
    save_path = './test_torchvision_transf'
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + '/*.png')
    file_path = os.path.join(save_path, f'sample_in_training_{str(len(files))}_gaussblur.png')
    fig.savefig(file_path)
    plt.close('all')

def visualize_levels(sample, sample_plot_min, sample_plot_max):
    
    sample = np.array(sample)
    if len(sample.shape) > 2:
        image = np.transpose(sample, (1, 2, 0))

    sample_plot_min = np.array(sample_plot_min)
    if len(sample_plot_min.shape) > 2:
        image_min = np.transpose(sample_plot_min, (1, 2, 0))

    sample_plot_max = np.array(sample_plot_max)
    if len(sample_plot_max.shape) > 2:
        image_max = np.transpose(sample_plot_max, (1, 2, 0))
        
    # Drawing
    fig, ax = plt.subplots(ncols=3, tight_layout=True, 
                figsize=(20,9), gridspec_kw={'width_ratios': [3, 3, 3]}, dpi = 200)

    ax[0].imshow(image, cmap='gray')    
    ax[0].set_title('Original')

    ax[1].imshow(image_min, cmap='gray')    
    ax[1].set_title('Minimum levels preset')

    ax[2].imshow(image_max, cmap='gray')    
    ax[2].set_title('Maximum levels preset')

    # Saving
    save_path = './test_torchvision_transf'
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(save_path + '/*.png')
    file_path = os.path.join(save_path, f'sample_in_training_{str(len(files))}_levels.png')
    fig.savefig(file_path, dpi = 'figure')
    plt.close('all')


def process_UTIMAP_inputdata(path, image, num_labels, target_type):
    sample = np.asarray(Image.open(os.path.join(path, 'samples', image)).convert('RGB'))
    sample = np.expand_dims(sample[:, :,  0], axis=-1)

    targets = np.zeros(shape=(sample.shape[0], sample.shape[1], num_labels), dtype=np.uint8)
    
    if target_type is not None:
        target_path = os.path.join(path, 'targets_distorted_mismatched', image)
        targets = ([np.asarray(Image.open(target_path).convert('RGB'))[:, :, label_channel]
                        for label_channel in range(num_labels)])    
        targets = np.stack(targets, axis=-1)


    composition = np.concatenate((sample, targets), axis=2)
    return composition    
        

if __name__ == '__main__':    
    from PIL import Image
    from torchvision.transforms import Compose
    from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
    
    # path = '/home/mcla/workspace/data/GPR/processed_synth_gsx_2k_3m_002_debug_02/train'
    # path = '/home/mcla/workspace/data/GPR/processed_synth_gsx_2k_3m_004/train'
    # path = '/home/mcla/workspace/data/GPR/processed_real_gsx_2k_3m_001_streer/test'
    # path = '/home/ddel/Workspace/Data/gps/pointcloud/real_cscans/processed_real_gsx_4930_150cm_80_behind_building/test'
    path = '/home/ddel/Workspace/Data/gps/pointcloud/real_cscans/raw-real-gsx-002-inference/test'
    labels = ['manhole', 'trench', 'pipe']
    stage = 'train'

    sample_path = os.path.join(path, 'samples')
    images = os.listdir(sample_path)

    transforms_pipeline = Compose(
        [            
            ToTensor(),
            SequentialCropInsideBoundingBox(image_dist_meters=62,
                 target_crop_meters=10, 
                 sample_dim=1, 
                 p=1,
                 stage='test',
                 verbose=True),
            # RandomLevelsPreset(levels_min_range=[5, 30], levels_max_range=[225, 250], p=1, num_labels=3, verbose=True), 
            # CustomRandomContrast(alpha_range=[0.7, 1.3], p=0.3, num_labels=3, verbose=True),
            RandomRotation(degrees=10, expand=False),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            # RandomCropInsideBoundingBox(image_dist_meters=25, target_crop_meters=10, min_percentage=0.3, p=1, stage=stage, verbose=True),
            Resize(size=(512, 512))
            # CustomGaussianBlur(kernel_size_range=[5, 9], sigma_range=[0.5, 2], p=0.3, num_labels=3, verbose=True),
            # BinarizeTargets(len(labels))
        ]
    )
            
    for idx, image in enumerate(images):
    
        print(f'Processing frame {image} as image {idx}')
        composition = process_UTIMAP_inputdata(path, image, len(labels), target_type=None)
        transformed_image = transforms_pipeline(composition)
        print(f'Shape after transforms: {transformed_image.shape}')
        # sample_values = torch.unique(transformed_image[:-len(labels), :, :])
        # print(f'Unique values for sample: {sample_values}')
        targets_values = torch.unique(transformed_image[-len(labels):, :, :])
        print(f'Unique values for targets: {targets_values}')
