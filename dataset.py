from pathlib import Path
import torch
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        super().__init__()
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params

    @staticmethod # Means it never accesses any class attributes and doesn't need the 'self' argument
    def extract_files(root):
        files = []

        for subject in root.glob('*'):
            slice_path = subject / 'data'

            for slice_data in slice_path.glob('*.npy'):
                files.append(slice_data)

        return files

    @staticmethod
    def change_img_to_label_path(path):
        '''
        Replaces 'imagesTr' with 'labelsTr'
        '''
        parts = list(path.parts) # get all directories within the path
        parts[parts.index('data')] = 'masks' # Replace 'imagesTr' with 'labelsTr'

        return Path(*parts) # Combine list back into a Path object
    
    def augment(self, slice_data, mask):
        random_seed = torch.randint(0, 100000, (1, )).item()
        imgaug.seed(random_seed)

        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(image=slice_data, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()

        return slice_aug, mask_aug
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        file_path = self.all_files[index]
        mask_path = self.change_img_to_label_path(file_path)
        slice_data = np.load(file_path).astype(np.float32)
        mask = np.load(mask_path)

        if self.augment_params:
            slice_data, mask = self.augment(slice_data, mask)

        return np.expand_dims(slice_data, 0), np.expand_dims(mask, 0)
