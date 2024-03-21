
import time
import torch
import numpy as np


class ChopBatchAggregate2d:
    """
    Example Usage:
    ```
        for volume, mask in test_set:
            CA = ChopAndAggregate(volume, (96, 96), (20, 20))
            
            for batch in CA:
                X = batch.cuda()
                Y = model(X)
                CA.add_batch(Y)
            
            preds = CA.aggregate()
            dice = get_dice(preds, mask)
    ```
    """
    
    def __init__(
            self, 
            image_tensor, 
            patch_size, 
            patch_overlap, 
            batch_size,
            num_classes, 
            device='cpu'
            ):
        """ Does the chopping where grid locations are calculated.
        Args:
            volume_tensor: HxC 2D image
            patch_size: (d, h, c) patch sizes
            patch_overlap: (d, h, c) overlaps. If negative, then the whole image
                is inputted.
            batch_size: inference batch size
            device: if 'cuda' then aggregate divide makes use of GPU.
                This only applies to 1 operation in aggregate, nothing else!!
        """
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        else:
            assert image_tensor.ndim == 3
        self.tensor = image_tensor
        self.device = device
        self.tensor_shape = image_tensor.shape[-2:]
        
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        
        # N x 6 array where each row is a crop: (d1,h1,x1,d2,h2,x2)
        if sum(self.patch_overlap) >= 0:
            self.grid_locations = get_grid_locations_2d(
                self.tensor_shape, self.patch_size, self.patch_overlap)
            self.num_patches = self.grid_locations.shape[0]
        else:
            iH, iW = image_tensor.shape[-2:]
            self.grid_locations = np.array([0, 0, iH, iW])[None]
            self.num_patches = 1

        self.num_classes = num_classes
        mask_shape = [num_classes] + list(self.tensor_shape)
        self.accum_tensor = torch.zeros(mask_shape, dtype=torch.float32,
                                        device=self.device, requires_grad=False)
        self.average_mask = torch.zeros(mask_shape, dtype=torch.uint8,
                                        device=self.device, requires_grad=False)
    

    def __len__(self):
        """ Returns the total number of batches. """
        N_patches = self.num_patches
        additional_batch = int( (N_patches % self.batch_size) > 0 )
        return N_patches // self.batch_size + additional_batch


    def __iter__(self):
        """ Initializes batch iterator. """
        self.batch_counter = 0
        self.patch_counter = 0
        self.num_aggregated_batches = 0
        return self
        
        
    def __next__(self):
        """ Gets a batch of crops. 
        Returns:
            tuple(batch_tensor, batch_locations)
                batch_tensor: Bx1xDxHxW of crops
                batch_locations = Bx6 array of upper & lower locations.
        """
        if self.batch_counter >= len(self):
            raise StopIteration
        
        # iH, iW = self.tensor.shape[-2:]
        # cH, cW = self.patch_size[-2:]
        # if len(self) == 1 and iH == cH and iW == cW:
        #     loc = np.array([0, 0, iH, iW])
        #     self.batch_counter += 1
        #     return self.tensor[None], loc[None]
            
        idx_start = self.batch_size * self.batch_counter
        idx_exl_end = min(idx_start + self.batch_size, self.num_patches)
        batch_patches = []
        for n in range(idx_start, idx_exl_end):
            lower = self.grid_locations[n,:2]
            upper = self.grid_locations[n,2:]
            patch = self.tensor[:, 
                                lower[0]: upper[0],
                                lower[1]: upper[1]]  # no clone, no Δ to data
            batch_patches.append(patch)
        self.batch_counter += 1
        batch_tensor = torch.stack(batch_patches, dim=0)
        batch_locations = self.grid_locations[idx_start:idx_exl_end]
        return batch_tensor, batch_locations


    def add_batch_predictions(self, batch, locations, act='none'):
        """
        Args:
            batch: BxCxHxW prediction tensor (C=#classes)
            act: activation for model predictions
                'none' means that batch tensor are logits
                'softmax' means apply softmax to the class dimension
                'sigmoid' means apply a sigmoid to entire tensor
        """
        N, C = batch.shape[:2]
        assert batch.ndim == 4
        assert C == self.num_classes
        assert N == locations.shape[0]
        assert locations.shape[1] == 4
        act = act.lower()

        with torch.no_grad():
            batch = batch.to(self.device)
            if 'softmax' in act:
                batch = batch.softmax(1)
            elif 'sigmoid' in act:
                batch = batch.sigmoid()

            for n in range(N):
                lower = locations[n, :2]
                upper = locations[n, 2:]

                self.accum_tensor[:, lower[0]:upper[0],
                                     lower[1]:upper[1]] += batch[n]
                self.average_mask[:, lower[0]:upper[0],
                                     lower[1]:upper[1]] += 1
    

    def aggregate(self, act='none', ret='none', cpu=True, numpy=True):
        """
        Args:
            act: 'none', 'softmax', 'sigmoid'
            ret: 'none', 'one_hot'/'1_h'/'1_hot', 'id'/'id_map'
        """
        start = time.time()
        act = act.lower()
        ret = ret.lower()
        
        with torch.no_grad():
            # agg_pred = torch.div(self.accum_tensor, self.average_mask)
            self.accum_tensor /= self.average_mask # in-place to save memory
            agg_pred = self.accum_tensor
            # agg_pred = torch.nan_to_num(agg_pred, nan=0, posinf=0, neginf=0)
            if 'cpu' not in self.device:  # clear accum tensors to save mem
                # del self.accum_tensor
                self.accum_tensor = None
                del self.average_mask
                self.average_mask = None
                # with torch.cuda.device(self.device):
                #     torch.cuda.empty_cache()
            
            if 'softmax' in act:
                agg_pred = agg_pred.softmax(0) 
            elif 'sigmoid' in act:
                agg_pred = agg_pred.sigmoid()
            
            # print('Aggregate (divide): ', time.time() - start)
            if 'hot' in ret or '1' in ret:  # one hot
                oh_shape = agg_pred.shape
                agg_pred = agg_pred.to(torch.float16)
                agg_pred = agg_pred.argmax(0).unsqueeze(0)
                if cpu:
                    agg_pred = agg_pred.cpu()
                    zeros = torch.zeros(oh_shape, dtype=torch.uint8)
                else:
                    zeros = torch.zeros(oh_shape, dtype=torch.uint8,
                                        device=agg_pred.device)
                agg_pred = torch.zeros(oh_shape, dtype=torch.uint8).scatter_(
                                       0, agg_pred, 1)
            elif 'id' in ret:
                fin_type = torch.int16
                if self.num_classes <= 255:
                    fin_type = torch.uint8
                elif self.num_classes > 32767:
                    fin_type = torch.int32
                if cpu:
                    agg_pred = agg_pred.argmax(0).to(fin_type).cpu()
                else:
                    agg_pred = agg_pred.argmax(0).to(fin_type)
            else:
                if cpu:
                    agg_pred = agg_pred.cpu()
            
            if numpy:
                agg_pred = agg_pred.detach().cpu().numpy()
        # print('Aggregate: ', time.time() - start)
        return agg_pred
        


def get_grid_locations_2d(image_size, patch_size, patch_overlap, sort=False):
    """ 
    Args:
        image_size: sequence or array of same dimension as image
        patch_size: patch lengths among all dims (same shape as image_size)
        patch_overlap: (same shape as image_size)
    """
    indices = []
    zipped = zip(image_size, patch_size, patch_overlap)
    for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
        end = im_size_dim + 1 - patch_size_dim
        step = patch_size_dim - patch_overlap_dim
        indices_dim = list(range(0, end, step))
        if indices_dim[-1] != im_size_dim - patch_size_dim:
            indices_dim.append(im_size_dim - patch_size_dim)
        indices.append(indices_dim)
    indices_ini = np.array(np.meshgrid(*indices)).reshape(2, -1).T
    indices_ini = np.unique(indices_ini, axis=0)
    indices_fin = indices_ini + np.array(patch_size)
    locations = np.hstack((indices_ini, indices_fin))
    
    if sort:
        return np.array(sorted(locations.tolist()))
    return locations
    