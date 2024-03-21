

# Default library tools
import pathlib
import numbers
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.functional as F
from einops import repeat, rearrange

# Default training components / utils that can easily be overrided below
import data, lib, experiments
from lib.utils import devices, timers, statistics
from lib.utils.io import output, files
from lib.utils.devices import mem, ram
from lib.utils.train.configs import create_namespace
from lib.utils.train.distributed import synchronize
from lib.utils.train.criterion import get_criterion 
from lib.utils.train.optimizers import get_optimizer
from lib.utils.train.schedulers import get_scheduler
from lib.utils.train.serialization import save_checkpoint

# 🔶🔶 Custom Imports 🔶🔶
from lib.metrics.seg_metrics import batch_metrics
from data.transforms2d.crops.inference import ChopBatchAggregate2d as CBA


# --- Training Tools and Constants (all-cap) --- #
curr_dir = pathlib.Path(__file__).parent.absolute()
save_dir = curr_dir / 'artifacts'
watch = timers.StopWatch()



class Trainer:
    """
    Note:
        - Does not support distributed training (no need for rank input)
    """
    data_dim = 2
    
    def __init__(self, config, model, data_d, tracker):
        
        # Configuration
        self.config = config
        self.device = config.experiment.device
        self.gpu_indices = config.experiment.gpu_idxs
        self.debug_config = config.experiment.debug
        self.use_amp = config.experiment.amp and bool(self.gpu_indices)
        self.task_config = config.task
        
        self.point_targ_type = self.task_config.points_target
        
        # Data 
        self.data_d = data_d         
        self.df = data_d['df']
        self.train_df = data_d['train_df']
        self.train_set = data_d['train_set']
        self.train_loader = data_d['train_loader']
        self.val_df = data_d['val_df']
        self.val_set = data_d['val_set']
        self.test_df = data_d['test_df']
        self.test_set = data_d['test_set']
        
        # Model & Optimizer
        
        self.model = model.to(self.device)
        if 'occ' in self.point_targ_type:
            self.criterion = get_criterion(config).to(self.device)
        else:
            from lib.losses.inr_losses import SDFLoss
            self.criterion = SDFLoss()
        self.optimizer = get_optimizer(config, self.model.parameters())
        self.scheduler = get_scheduler(config, self.optimizer)
        
        # Summarization
        self.tracker = tracker
        self.global_iter = 0
        
        
    def train_epoch(self, epoch):
        watch.tic('epoch')
        epmeter = statistics.EpochMeters()
        
        self.model.train()
        device = self.device
        
        # Print
        tot_epochs = self.config.train.epochs - self.config.train.start_epoch
        sec_header = (f'Starting Epoch Index {epoch} / {tot_epochs} '
                      f'(lr: {self.scheduler.lr:.7f}, amp: {self.use_amp})')
        output.subsection(sec_header)
        ram(disp=True)  # Print process RAM usage.
        watch.tic('iter')
        
        # Train Epoch
        for it, batch in enumerate(self.train_loader):
            
            X = batch['images'].float().to(device, non_blocking=True)
            Y_id = batch['masks'].to(torch.uint8).to(device, non_blocking=True)
            X_p = batch['X_p'].to(device)
            Y_p = batch['Y_p'].to(device)
                        
            out_d = self.model(X, p=X_p)
            out = out_d['out'] if isinstance(out_d, dict) else out_d  
            
            orig_targ = Y_p if 'occ' in self.point_targ_type \
                        else batch['Y_sdf'].to(device)
            targ = orig_targ
            # if self.task_config.num_extensions:
            #     targ = repeat(targ, 'B P -> B (rep P)', 
            #                   rep=self.task_config.num_extensions+1)
                
            # loss_d = self.criterion(
            #     out,  # B x classes x #pts
            #     targ # B x #pts
            # )  
            
            # import IPython; IPython.embed(); 
            
            loss_d = self.criterion(
                    out[:,:,:orig_targ.shape[-1]],  # B x classes x #pts
                    orig_targ  # B x #pts
            )
            loss = loss_d['loss'] if isinstance(loss_d, dict) else loss_d 
            
            if self.task_config.num_extensions:
                ext_targ = repeat(orig_targ, 'B P -> B (rep P)', 
                                  rep=self.task_config.num_extensions)
                ext_loss_d = self.criterion(
                        out[:,:,orig_targ.shape[-1]:],  # B x classes x #pts
                        ext_targ  # B x #pts
                )
                ext_wt = self.config.task.ext_loss_wt
                loss = (1 - ext_wt) * loss + ext_wt * ext_loss_d['loss']
                # targ = repeat(orig_targ, 'B P -> B (rep P)', 
                #               rep=self.task_config.num_extensions+1)
            
            # loss_d = self.criterion(out,  # B x classes x #pts
            #                         Y_p.unsqueeze(1).float())  # B x #pts
            
            if 'loss_str' in out_d:
                loss_str = out_d['loss_str']
            elif hasattr(self.criterion, 'get_loss_string'):
                loss_str = self.criterion.get_loss_string(loss_d)
            else:
                loss_str = f'loss {loss.item():.3f}'
            
            # Global Auxiliary Loss
            wt = self.config.task.global_loss_wt
            if self.model.global_decoder is not None and wt:
                glob_out = out_d['out_global']
                glob_loss_d = self.criterion(glob_out, orig_targ) 
                glob_loss = glob_loss_d['loss']
                loss = (1 - wt) * loss + wt * glob_loss
                
                smooth_wt = self.config.task.smoothness_loss_wt
                suffix = ''
                if smooth_wt:
                    smooth_loss = ((out - glob_out.detach()) ** 2).mean() * \
                                  smooth_wt
                    loss = loss + smooth_loss
                    suffix = f', smooth {smooth_loss.item():.2f}'
                
                if 'occ' in self.point_targ_type:
                    loss_str = (f'loss {loss.item():.2f} ('
                                f'ce {loss_d["ce"].item():.2f}, '
                                f'dc {loss_d["dc"].item():.2f}, '
                                f'glob {glob_loss.item():.2f}{suffix})')
                else:
                    loss_str = (f'loss {loss.item():.2f} ('
                                f'local {loss_d["loss"].item():.2f}, '
                                f'glob {glob_loss.item():.2f}{suffix})')
            
            # Embedding Loss
            emb_wt = self.config.task.embedding_loss_wt
            if emb_wt:
                emb_loss = emb_wt * (out ** 2).sum()
                if self.model.global_decoder is not None and wt:
                    glob_out = out_d['out_global']
                    emb_loss = emb_loss + emb_wt * (glob_out ** 2).sum()
                loss = loss + emb_loss
                loss_str = loss_str[:-1] + f', emb {emb_loss.item():.2f})'
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # --- Iteration Tracking, Metrics, and Summary --- # 
            
            # Batch Metrics
            # targs = batch['masks_1h'].to(torch.uint8)
            batch_targs = Y_p.unsqueeze(1).cpu()
            if self.task_config.num_extensions:
                batch_targs = repeat(batch_targs, 'B 1 P -> B 1 (rep P)', 
                                     rep=self.task_config.num_extensions+1)
            with torch.no_grad():
                # pred_ids = out.sigmoid() >= 0.5
                # preds = torch.zeros(targs.shape, dtype=torch.uint8,
                #                     device=self.device)
                # preds.scatter_(1, pred_ids, 1)
                # preds = preds.cpu()
                if 'occ' in self.point_targ_type:
                    preds = (out.sigmoid() >= 0.5).cpu().to(torch.uint8)
                else:
                    preds = (out.tanh() <= 0).cpu().to(torch.uint8)
            iter_metrics_d = batch_metrics(preds, batch_targs, 
                                           naive_avg=False,
                                           ignore_background=False)
            
            # Record Loss
            iter_metrics_d['loss'] = loss.item()

            # Print Iter Info
            break_train_iter = self.debug_config.break_train_iter
            print_every = max(1, len(self.train_loader) // 4)
            debug_print = self.debug_config.mode or break_train_iter
            if it % print_every == 0 or debug_print:
                iter_time = watch.toc('iter', disp=False)
                    
                print(
                    f"\n    Iter {it+1}/{len(self.train_loader)} "
                    f"({iter_time:.1f} sec, {mem(self.gpu_indices):.1f} GB) - "
                    f"{loss_str} \n      "
                    f"Jaccard (agg): {iter_metrics_d.jaccard_summary:.3f} "
                    f"({iter_metrics_d.jaccard_summary_agg:.3f}), "
                    f"Dice (agg): {iter_metrics_d.dice_summary:.3f} "
                    f"({iter_metrics_d.dice_summary_agg:.3f})\n       "
                    f"Class Dices: {iter_metrics_d['dice_class']}"
                )
            epmeter.update({k: iter_metrics_d[k] 
                            for k in self.config.serialize.train_metrics})
            self.global_iter += 1
            watch.tic('iter')
            
            # Break Training Iter for Debugging
            is_bti_num = isinstance(break_train_iter, numbers.Number)
            if break_train_iter:
                if it >= break_train_iter if is_bti_num else 14:
                    break
        
        del loss; del out_d; del preds;   # save mem for inference
        self.scheduler.step(epoch=epoch, value=0)
        return epmeter
        
    @torch.no_grad()
    def infer(
            self, 
            dataset, 
            epoch,
            name='test', 
            overlap_perc=0.2,
            save_predictions=False
            ):
        """ Returns metrics for a test or validation set & saves predictions.
        Args:
            num_examples: needed for DDP since rank 0 worker needs a total count
                of examples so it knows how many times to pull metrics from Q.
        """
        # save_predictions = True
        print(f' *Save Predictions: {save_predictions}')
        config = self.config
        lthresh = config.test.logits_thresh
        device = config.experiment.device
        cba_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if len(config.experiment.gpu_idxs) > 1:
            cba_device = f'cuda:{len(config.experiment.gpu_idxs) - 1}'

        watch.tic(name.title())
        epmeter = statistics.EpochMeters()
        self.model.eval()

        watch.tic(f'{name}_iter')
        metric_results = []
        tracked_metrics, k = [], 'dice_summary'
        N_test_samples = 2 if config.experiment.debug.break_test_iter \
                        else len(dataset) 
        for i in range(N_test_samples):
            example_d = dataset[i]
            image = example_d['images']
            mask = example_d['masks_1h']  #  image: float32, mask: uint8

            # Create Chop-Batch-Aggregate inference helper
            test_batch_size = config.test.batch_size
            num_classes = len(dataset.classnames)
            patch_size = config.data[config.data.name].net_in_size
            overlap = [int(overlap_perc * s) for s in patch_size]
            cba = CBA(image, patch_size, overlap, 
                      test_batch_size, 1, device=cba_device)
            glob_cba = CBA(image, patch_size, overlap, 
                           test_batch_size, 1, device=cba_device)
            
            size_mult = torch.tensor(patch_size, device=device)[None, None] - 1
            
            # Get Patch Predictions and Update Aggregator
            for bidx, batch in enumerate(cba):
                crops, locations = batch
                crops = crops.to(device)
                out_d = self.model(crops)
                
                # Convert point outputs to image
                logits = out_d['out']
                points = out_d['gp'].flip(dims=(-1,))  # xy to yx
                # import IPython; IPython.embed(); 
                
                C = logits.shape[1] 
                assert C == 1, f'Not supported, {logits.shape}'
                shape = crops.shape
                pred_image = torch.zeros((shape[0], C, shape[-2], shape[-1]),
                                         device=crops.device)
                for b in range(pred_image.shape[0]):
                    b_logits = logits[b]             # 1 x #pts
                    b_coords = ((points[b] + 1) / 2) # 1 x #pts x 2
                    b_coords = torch.round(b_coords * size_mult).long()
                    
                    pred_image[b,0,b_coords[0,:,0],b_coords[0,:,1]] = b_logits[0]                
                cba.add_batch_predictions(pred_image, locations, 
                                          act=config.test.pred_agg_activation)
                
                # Convert global outputs to image
                glogits = out_d['out_global']           
                if glogits is not None:     
                    gpred_image = torch.zeros((shape[0], C, shape[-2], shape[-1]),
                                            device=crops.device)
                    for b in range(gpred_image.shape[0]):
                        b_logits = glogits[b]             # 1 x #pts
                        b_coords = ((points[b] + 1) / 2) # 1 x #pts x 2
                        b_coords = torch.round(b_coords * size_mult).long()
                        
                        gpred_image[b,0,b_coords[0,:,0],
                                   b_coords[0,:,1]] = b_logits[0]
                    glob_cba.add_batch_predictions(gpred_image, locations, 
                            act=config.test.pred_agg_activation)

            # Get Aggregated Predictions
            del crops; del logits; del out_d
            agg_predictions = cba.aggregate(ret='none', cpu=True, numpy=False,
                                            act='none') 
            if 'occ' in self.point_targ_type:
                agg_predictions = (agg_predictions >= lthresh).to(torch.uint8)
            else:
                agg_predictions = (agg_predictions <= 0).to(torch.uint8)

            # Compute Metrics
            mets = batch_metrics(agg_predictions[None], 
                                 mask[-1][None, None], 
                                 naive_avg=True,
                                 ignore_background=True)
            metric_results.append(mets)
            tracked_metrics.append(mets[k])
            
            # Save Predictions in Artifacts Folder (?)
            num_epochs = config.train.epochs
            
            if save_predictions:
                curr_path = pathlib.Path(__file__).parent
                exp_path = curr_path.parent
                pred_dir = f'{name}_epochindex_{epoch}'
                save_dp = exp_path / 'artifacts' / config.experiment.id / pred_dir
                save_path = save_dp / f'imageindex_{i}.png'
                
                pred_arr = agg_predictions[-1]
                mask_arr = mask[-1]
                
                gpred_arr, gtitle = None, ''
                if glogits is not None:
                    if 'occ' in self.point_targ_type:
                        gpred_arr = (glob_cba.aggregate(ret='none', cpu=True, 
                                     numpy=False, act='none') >= lthresh).to(torch.uint8)
                    else:
                        gpred_arr = (glob_cba.aggregate(ret='none', cpu=True, 
                                     numpy=False, act='none') <= 0).to(torch.uint8)
                        
                    gmets = batch_metrics(mask[-1][None, None], 
                                          gpred_arr[None], 
                                          naive_avg=True,
                                          ignore_background=True)
                    gtitle = f'GP (dice: {gmets[k]:.2f})'
                
                save_planar_image_result(save_path, image, pred_arr, mask_arr,
                                         glob_arr=gpred_arr,
                                         pred_title=f'LP (dice: {mets[k]:.2f})',
                                         glob_title=gtitle)
            
            # Print and Record Metrics
            if config.experiment.rank == 0 and i % 10 == 0:
                elaps = watch.toc(f'{name}_iter', disp=False)
                N = len(dataset)
                print(f'🖼️  Inference ({name.title()}) Image {i+1} / {N} '
                    f'({elaps:.2f} sec) \n'
                    f'       Dice: {float(mets["dice_summary"]):.4f} '
                    f' {mets["dice_class"]}',
                    flush=True) 
            epmeter.update({k: mets[k] 
                            for k in self.config.serialize.test_metrics})
            
            watch.tic(f'{name}_iter')
            del cba; del agg_predictions

        # Save plot
        if save_predictions:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(2, 1, 1)
            x, y  = list(range(len(dataset))), tracked_metrics
            ax.plot(x, y)
            ax.set_xticks(list(range(0, len(dataset), 2)), minor=True)
            for i, j in zip(x, y):
                ax.annotate(f'{round(j * 100)}', xy=(i-0.05, j+0.1))
            ax.set_ylim(0, 1.0)
            ax = fig.add_subplot(2, 1, 2)
            y  = sorted([[m, i] for i, m in zip(x, tracked_metrics)], 
                        reverse=True)
            ax.plot(x, [v[0] for v in y])
            ax.set_xticks(list(range(0, len(dataset), 2)), minor=True)
            ax.set_ylim(0, 1.0)
            for i, j in zip(x, y):
                ax.annotate(f'{j[1]}', xy=(i, j[0]))
            plt.savefig(str(save_dp / f'_{k}.png'), dpi=120)
            plt.close()
        
        # Compute Set-wise Metrics
        final_mets_d = epmeter.avg(no_avg=['tps', 'fps', 'fns'])
        tp, fp, fn = (final_mets_d[k] for k in ['tps', 'fps', 'fns'])
        final_mets_d['dice_summary_agg'] = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        final_mets_d['jaccard_summary_agg'] = tp / (tp + fp + fn + 1e-7)
        
        watch.toc(name.title(), disp=True)
        return final_mets_d
    


# ========================================================================== #
# * ### * ### * ### *           Misc Utilities           * ### * ### * ### * #
# ========================================================================== #
    
    
def save_planar_image_result(save_path, image, pred_prob, targ,
                             dpi=120, glob_arr=None, 
                             pred_title='', glob_title=''):
    
    def process_input(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Conver to numpy array with canonical image shapes
        if image.ndim == 3:
            if image.shape[0] in (1, 3):
                if image.shape[0] == 1:
                    image = image[0]
                else:
                    assert image.shape[0] == 3, f'Invalid shape {image.shape}'
                    image = np.moveaxis(image, 0, -1)
            else:
                assert image.shape[-1] in (1, 3), f'Invalid shape {image.shape}'
                if image.shape[-1] == 1:
                    image = image[..., 0]
        else:
            assert image.ndim == 2
            
        # Intensity Processing (image is now HxW or HxWx3)
        image = image.astype('float32')
        
        imin, imax = image.min(), image.max()
        imin = 0. if imin == imax else imin
        if imin != 0:
            image -= imin 
        if imax - imin != 1:
            if imax - imin != 0:
                image /= imax - imin 
        
        image = np.round(image * 255).astype('uint8')
        
        return image 
    
    def add_axis(fig, axis_loc, image, title=''):
        ax = fig.add_subplot(axis_loc)
        cmap = 'gray' if image.ndim == 2 else None
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        return ax
    
    image_arr = process_input(image) 
    pred_arr = process_input(pred_prob) 
    targ_arr = process_input(targ)
    
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    ptitle = 'Prediction' if not pred_title else pred_title
    gtitle = 'Global Prediction' if not glob_title else glob_title
    
    if glob_arr is not None:
        gpred_arr = process_input(glob_arr) 
        fig = plt.figure(figsize=(16,8))
        ax = add_axis(fig, 141, image_arr, title='Image')
        ax = add_axis(fig, 142, pred_arr, title=ptitle)
        ax = add_axis(fig, 143, targ_arr, title='Ground Truth')
        ax = add_axis(fig, 144, gpred_arr, title=gtitle)
    else:
        fig = plt.figure(figsize=(14,8))
        ax = add_axis(fig, 131, image_arr, title='Image')
        ax = add_axis(fig, 132, pred_arr, title=ptitle)
        ax = add_axis(fig, 133, targ_arr, title='Ground Truth')
    
    plt.savefig(str(save_path), dpi=dpi)
    plt.close()
            
