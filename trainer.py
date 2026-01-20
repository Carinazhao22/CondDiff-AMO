
import glob
import os
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.evaluation import evaluation_image, AverageMeter
from model.train_val_forward import simple_train_val_forward 
from torchvision.utils import save_image

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cal_mae(gt, res, thresholding, save_to=None, n=None):
    res = F.interpolate(res.unsqueeze(0), size=gt.shape, mode='bilinear', align_corners=False)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res > 0.5).float() if thresholding else res
    res = res.cpu().numpy().squeeze()
    if save_to is not None:
        plt.imsave(os.path.join(save_to, n), res, cmap='gray')
    return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res

    return wrapper


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_val_forward_fn=simple_train_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        """
            Initialize the accelerator.
        """
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )
        project_name = getattr(cfg, "project_name", 'ResidualDiffsuion-v7')
        self.accelerator.init_trackers(project_name) # config=cfg removed for deleting error message
        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        """
            Initialize the model and parameters.
        """
        self.model = model
         
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        # calculate training steps
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer
        self.train_dataset = cfg.datasets
        if self.accelerator.is_main_process:
            # save results in wandb folder if results_folder is not specified
            self.results_folder = Path(results_folder if results_folder
                                       else os.path.join(self.accelerator.get_tracker('wandb', unwrap=True).dir, "../"))
            self.results_folder.mkdir(exist_ok=True)
        
        self.vis_path = cfg.vis_path
        
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0
        self.iteration = 0
        self.sample_iter = 0
        # prepare model, dataloader, optimizer with accelerator
        self.model,self.opt, self.scheduler, self.train_loader, self.test_loader \
            = self.accelerator.prepare(self.model, self.opt, scheduler, self.train_loader, self.test_loader)

            
    def align_raw_size(self, full_mask, obj_position, vm_pad, data):
        vm_np_crop = data["vm_no_crop"].squeeze()
        H, W = vm_np_crop.shape[-2], vm_np_crop.shape[-1]
        bz, seq_len = full_mask.shape[:2]
        new_full_mask = torch.zeros((bz, seq_len, H, W)).to(torch.float32).cuda()
        if len(vm_pad.shape)==3:
            vm_pad = vm_pad[0]
            obj_position = obj_position[0]
        for b in range(bz):
            paddings = vm_pad[b]
            position = obj_position[b]
            new_fm = full_mask[
                b, :,
                :-int(paddings[0]) if int(paddings[0]) !=0 else None,
                :-int(paddings[1]) if int(paddings[1]) !=0 else None
            ]
            vx_min = int(position[0])
            vx_max = min(H, int(position[1])+1)
            vy_min = int(position[2])
            vy_max = min(W, int(position[3])+1)
            resize = transforms.Resize([vx_max-vx_min, vy_max-vy_min])
            try:
                new_fm = resize(new_fm)
                new_full_mask[b, :, vx_min:vx_max, vy_min:vy_max] = new_fm[0]
            except:
                new_fm = new_fm
        return new_full_mask

    def save(self, epoch, latest = True, max_to_keep=10):
        """
        Delete the old checkpoints to save disk space.
        """
        if not self.accelerator.is_local_main_process:
            return
        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        # keep the last n-1 checkpoints
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)
        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            # 'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        if latest:
            save_name = str(self.results_folder / f'model-latest.pt')
            torch.save(data, save_name)
        else:
            save_name = str(self.results_folder / f'model-{epoch}.pt')
            last_save_name = str(self.results_folder / f'model-{epoch}-last.pt')

            # if save file exists, rename it to last_save_name
            if os.path.exists(save_name):
                os.remove(last_save_name) if os.path.exists(last_save_name) else None
                os.rename(save_name, last_save_name)

            torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)

            self.cur_epoch = data['epoch']
            # self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either milestone or path')
        if self.scheduler is not None:
            # step scheduler to the last epoch
            for _ in range(self.cur_epoch):
                self.scheduler.step()
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

    def to_cuda(self, meta, device):
        for k in meta:
            if  meta[k] is not None:
                meta[k] = meta[k].to(device)
        return meta
    
    def visualize(self, pred_vm, pred_fm, meta, mode, iteration):
        pred_fm = pred_fm.squeeze()
        pred_vm = pred_vm.squeeze()
        gt_vm = meta["vm_no_crop"].squeeze()
        gt_fm = meta["fm_no_crop"].squeeze()
        to_plot = torch.cat((pred_vm, pred_fm, gt_vm, gt_fm)).cpu().numpy()
        save_dir = os.path.join('./output_img', '{}_samples'.format(mode))
        image_id, anno_id= meta["img_id"], meta["anno_id"]
        plt.imsave("{}/{}_{}_{}.png".format(save_dir, iteration, int(image_id.item()), int(anno_id.item())), to_plot)
    
    
 
    def overlay_mask_with_contour(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_color: tuple = (0.9, 0.9, 0.1),  # RGB color for mask interior (0-1 range)
        contour_color: tuple = (0.9, 0.9, 0.1),  # RGB color for mask outline (0-1 range)
        mask_alpha: float = 0.5,  # Transparency for mask interior (0-1)
        contour_alpha: float = 0.5,  # Transparency for mask outline (0-1)
        contour_width: int = 0 # Pixel width of the outline
    ):
        """
        Overlay mask with interior and contour on input image
        Args:
            image: Input image tensor ([3, H, W], range: 0-1)
            mask: Binary mask tensor ([1, H, W] or [H, W], range: 0-1)
            ... (other parameters as above)
        Returns:
            Overlayed image tensor ([3, H, W])
        """
        # Basic tensor preparation
        mask = (mask > 0.5).to(torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [H, W] → [1, H, W]
        mask = mask.expand_as(image[:1])  # Match image's spatial dims
        
        # 1. Create color layers
        mask_color = torch.tensor(mask_color, device=image.device).view(3, 1, 1)
        contour_color = torch.tensor(contour_color, device=image.device).view(3, 1, 1)
        
        # 2. Detect mask contours using morphological operations
        kernel = torch.ones(1, 1, contour_width*2+1, contour_width*2+1, device=image.device)
        mask_dilated = (F.conv2d(mask.unsqueeze(0), kernel, padding=contour_width)[0] > 0.5).to(torch.int64)
        contour = (mask_dilated - mask) > 0  # Outline = Dilated mask - Original mask
        
        # 3. Blend layers
        # Interior blending
        overlay = image * (1 - mask_alpha * mask) + mask_color * (mask_alpha * mask)
        
        # Contour blending (on top of interior)
        overlay = overlay * (1 - contour_alpha * contour) + contour_color * (contour_alpha * contour)
        
        return overlay.clamp(0, 1)

    @torch.inference_mode()
    @run_on_seed
    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name = data['image'], data['gt'], data['name'] 
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device).squeeze(1)
            out = self.train_val_forward_fn(model, image=image, verbose=False)
            res = out["pred"].detach().cpu()
            maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, res, name)]
        # gather all the results from different processes
        accelerator.wait_for_everyone()
        mae = accelerator.gather(torch.tensor(maes).mean().to(device))
        mae = mae.mean().item()
        # mae = mae_sum / test_data_loader.dataset.size
        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    @torch.inference_mode()
    @run_on_seed
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_in_iou
        if '_best_in_iou' not in globals():
            _best_in_iou = 0

        # # Mean Absolute Error
        # def cal_mae(gt, res, thresholding, save_to=None, n=None):
        #     res = res.cpu().numpy().squeeze()
        #     if save_to is not None:
        #         plt.imsave(os.path.join(save_to, n), res, cmap='gray')
        #     return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        # iter = 0 
        # for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
        with torch.no_grad():
            data = next(test_data_loader)
            data = self.to_cuda(data, device)
            image, gt, img_id, anno_id, vm = data['image'], data['gt'], data["img_id"], data["anno_id"], data['seg'] # , data['image_for_post']
            random_points, gauss_map  = data['random_points'], data['gauss_map']
            gt = gt.squeeze(1)
            gt = [np.array(x.cpu(), np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device).squeeze(1)
            ensem_out = self.train_val_forward_fn(model, image=image, random_points = random_points, gauss_map = gauss_map, seg = vm, time_ensemble=True,
                                                  gt_sizes=[g.shape for g in gt], verbose=False)
            # pred_fm_crop: B,1,256,256
            pred_fm_crop = torch.cat(ensem_out["pred"], dim=0)
            
            # pred_vm = self.align_raw_size(data['vm_crop_gt'], data['obj_position'], data["vm_pad"], data)
            
            pred_fm =pred_fm_crop.to("cuda") # self.align_raw_size(pred_fm_crop, data['obj_position'], data["vm_pad"], data)
        
            # visualization
            # self.visualize(pred_vm, pred_fm, data, 'test', iter)
            # iter +=1

            # ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]
            loss_eval = self.loss_and_evaluation(pred_fm, data)
            iou = loss_eval['iou'].item() / (loss_eval['iou_count'].item() + 1e-7)
            invisible_iou_ = loss_eval['invisible_iou_'].item() / (loss_eval['iou_count'].item() + 1e-7)
            output = {
                    "img_id" : int(img_id[0].cpu().detach().numpy()),
                    "anno_id": int(anno_id[0].cpu().detach().numpy()),
                    "iou": iou,
                    "in_iou": invisible_iou_}

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        # ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()

        _best_in_iou = min(_best_in_iou, invisible_iou_)
        return output, _best_in_iou
    
    @torch.inference_mode()
    @run_on_seed
    def test_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        iter = 0
 
    
        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device

        intersection_rec = AverageMeter()
        union_rec = AverageMeter()
        c_intersection_rec = AverageMeter()
        c_union_rec = AverageMeter()
        inv_intersection_rec =  AverageMeter()
        inv_union_rec = AverageMeter()
        list_iou = []
        list_inv_iou = []
        save_image_id, save_ann_id = [], []
        vis_count = 50
        total_time = 0
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
                data = self.to_cuda(data, device)
                iter += 1
                image, gt, vm = data['image'], data['gt'], data['seg'] #data['category_id'] # , data['image_for_post']
                img_id = int(data['img_id'].item())
                ann_id = int(data['anno_id'].item())
                # occlude_rate = data['occlude_rate'].item()
                random_points, gauss_map  = data['random_points'], data['gauss_map']
                # save_image(image[0], "original.jpg")
                # save_image([gt[0], vm[0]], "gt.jpg")
                gt = gt.squeeze(1)
                # gt2 = gt
                gt = [np.array(x.cpu(), np.float32) for x in gt]
                gt = [x / (x.max() + 1e-8) for x in gt]
                image = image.to(device).squeeze(1)
                
                ensem_out, total_time= self.train_val_forward_fn(model, total_time = total_time, cur_idx = iter, image=image, random_points = random_points, gauss_map = gauss_map, seg = vm, time_ensemble=True,
                                                        gt_sizes=[g.shape for g in gt], verbose=False)
                 
                # pred_fm_crop: B,1,256,256
                pred_fm_crop = torch.cat(ensem_out["pred"], dim=0)
                pred_cfm_crop = ensem_out["vm"]
              
                pred_fm = self.align_raw_size(pred_fm_crop, data['obj_position'], data["vm_pad"], data)
                pred_cfm = self.align_raw_size(pred_cfm_crop, data['obj_position'], data["vm_pad"], data)
                # self.visualize(pred_vm, pred_fm, data, 'test', iter)
                # ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]
                
                pred_fm = pred_fm.squeeze()
                pred_cfm = pred_cfm.squeeze()
                fm_no_crop = data["fm_no_crop"].squeeze()
                vm_no_crop = data["vm_no_crop"].squeeze()
                # print(pred_fm.shape)
                # print("fm no crop", fm_no_crop.shape)
                
                # pred_vm = pred_vm.squeeze()
                 
                pred_fm = (pred_fm > 0.5).to(torch.int64)
                pred_cfm = (pred_cfm > 0.5).to(torch.int64)
                
                intersection = ((pred_fm == 1) & (fm_no_crop == 1)).sum()
                union = ((pred_fm == 1) | (fm_no_crop == 1)).sum()
                intersection_rec.update(intersection)
                union_rec.update(union)
                
                c_intersection = ((pred_cfm == 1) & (fm_no_crop == 1)).sum()
                c_union = ((pred_cfm == 1) | (fm_no_crop == 1)).sum()
                c_intersection_rec.update(c_intersection)
                c_union_rec.update(c_union)

                # for invisible mIoU
                 
                inv_intersection = ((pred_fm == 1) & (fm_no_crop == 1) & (vm_no_crop == 0)).sum()
                inv_union = (((pred_fm == 1) | (fm_no_crop == 1)) & (vm_no_crop == 0)).sum()
               
                if ((fm_no_crop - vm_no_crop).sum()) != 0:
                    inv_intersection_rec.update(inv_intersection)
                    inv_union_rec.update(inv_union)
                    Flag = 1
                else:
                    Flag = 0
                

                # loss_eval = self.loss_and_evaluation(pred_fm, data)
                # iou += loss_eval['iou']
                # iou_post += loss_eval['iou_post']
                # iou_count += loss_eval['iou_count']
                # invisible_iou_ += loss_eval['invisible_iou_']
                # invisible_iou_post += loss_eval['invisible_iou_post']
                # occ_count += loss_eval['occ_count']
                iou = intersection/(union+1e-6)
                occ_iou = inv_intersection/(inv_union+1e-6)
                iou_ciou = c_intersection/(c_union+1e-6)
                list_iou.append(iou)
                list_inv_iou.append(occ_iou)
                image_no_crop = data["img_no_crop"].squeeze().permute(2,0,1)/255
                
                # noise1 = ensem_out["noise1"]  
                # noise2 = ensem_out["noise2"]
                # vm_mask = data['seg']
                # save_image(noise1,"./vis/noise1_{}_{}.png".format(img_id, ann_id) )
                # save_image(noise2,"./vis/noise2_{}_{}.png".format(img_id, ann_id) )
                # save_image(vm_mask, "./vis/vm_mask_{}_{}.png".format(img_id, ann_id))  
                
                # if len(save_image_id) < vis_count:
                #     if  iou- iou_ciou > 5: 
                #         self.logger.info("#########Saving #{} visualization #########". format(len(save_image_id) + 1))
                #         # amodal_mask = data['gt']
                #         image = data["image"]
                       
                #         image_point = data["img_point"]
                #         # vm_mask = data['seg']
 
                #         # noise1 = ensem_out["noise1"]  
                #         # noise2 = ensem_out["noise2"]  
                #         vis_ours = self.overlay_mask_with_contour(image_no_crop.to("cpu"), pred_fm.squeeze().to("cpu"))
                #         vis_fm = self.overlay_mask_with_contour(image_no_crop.to("cpu"), fm_no_crop.squeeze().to("cpu"))
                #         vis_vm = self.overlay_mask_with_contour(image_no_crop.to("cpu"), vm_no_crop.squeeze().to("cpu"))
                #         vis_cfm = self.overlay_mask_with_contour(image_no_crop.to("cpu"), pred_cfm.squeeze().to("cpu"))
                        
                #         save_image_id.append(img_id)
                #         save_ann_id.append(ann_id)
                        
                #         save_image(image_point, os.path.join(self.vis_path, "img_point_{}_{}.png".format(img_id,ann_id)))
                #         save_image(image_no_crop, os.path.join(self.vis_path, "img_{}_{}.png".format(img_id,ann_id)))
                #         save_image(vis_fm, os.path.join(self.vis_path, "fm_{}_{}.png".format(img_id,ann_id)))
                #         save_image(vis_vm, os.path.join(self.vis_path, "vm_{}_{}.png".format(img_id,ann_id)))
                #         save_image(vis_ours, os.path.join(self.vis_path, "pred_{}_{}.png".format(img_id,ann_id)))
                #         save_image(vis_cfm, os.path.join(self.vis_path, "pred_coarse_{}_{}.png".format(img_id,ann_id)))
                #         # save_image(noise1,"./vis/noise1_{}_{}.png".format(img_id, ann_id) )
                #         # save_image(noise2,"./vis/noise2_{}_{}.png".format(img_id, ann_id) )
                #         # save_image(amodal_mask, "./vis/amodal_mask_{}_{}.png".format(img_id, ann_id))
                #         # save_image(image, "./vis/image_{}_{}.png".format(img_id, ann_id))
                #         # save_image(vm_mask, "./vis/vm_mask_{}_{}.png".format(img_id, ann_id))
                #         # save_image(pred_fm_crop, "./vis/pred_fm{}_{}.png".format(img_id, ann_id))
                # else:
                #     with open('save_vis_{}_id.txt'.format(self.train_dataset), 'w') as f:
                #         for img_id1, ann_id1  in zip(save_image_id, save_ann_id):
                #             f.write("{},{}\n".format(img_id1, ann_id1))
                #     break

                self.logger.info('img_id: {}, anno_id: {}, IoU: {}, occ_IoU: {}, Flag: {}'.format(img_id, ann_id, iou, 
                occ_iou, Flag))

                # self.logger.info('img_id: {}, anno_id: {}, IoU: {}, IoU_post: {}, occ_IoU: {}, occ_IoU_post: {}'.format(data['img_id'].item(), data['anno_id'].item(), loss_eval['iou'].item(), 
                # loss_eval['iou_post'].item(), loss_eval['invisible_iou_'].item(), loss_eval['invisible_iou_post'].item()))

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        # ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()
        
        miou = intersection_rec.sum / (union_rec.sum + 1e-10) # mIoU
        inv_miou = inv_intersection_rec.sum / (inv_union_rec.sum + 1e-10) # mIoU

        self.logger.info('meanIoU: {}'.format(miou))
        # self.logger.info('meanIoU post-process: {}'.format(iou_post.item() / iou_count.item()))
        self.logger.info('meanIoU invisible: {}'.format(inv_miou))
        self.logger.info('per instance time:{}'.format(total_time/(iter-50)))
        # self.logger.info('meanIoU invisible post-process: {}'.format(invisible_iou_post.item() / occ_count.item()))
        # self.logger.info('iou_count: {}'.format(iou_count))
        # self.logger.info('occ_count: {}'.format(occ_count))
        # if len(save_ann_id)< vis_count:
        #     with open('save_vis_{}_id.txt'.format(self.train_dataset), 'w') as f:
        #                 for img_id1, ann_id1  in zip(save_image_id, save_ann_id):
        #                     f.write("{},{}\n".format(img_id1, ann_id1))
       
        return miou , inv_miou 
    
    @torch.inference_mode()
    @run_on_seed
    def test_time_ensemble1(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        iter = 0
 
    
        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device

        intersection_rec = AverageMeter()
        union_rec = AverageMeter()

        intersection_rec_0 = AverageMeter()
        union_rec_0 = AverageMeter()

        intersection_rec_1 = AverageMeter()
        union_rec_1 = AverageMeter()

        intersection_rec_2 = AverageMeter()
        union_rec_2 = AverageMeter()

        intersection_rec_3 = AverageMeter()
        union_rec_3 = AverageMeter()

        intersection_rec_4 = AverageMeter()
        union_rec_4 = AverageMeter()
        
        inv_intersection_rec =  AverageMeter()
        inv_union_rec = AverageMeter()
        list_iou = []
        list_inv_iou = []

        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
                data = self.to_cuda(data, device)
                iter += 1
                image, gt, vm = data['image'], data['gt'], data['seg'] #data['category_id'] # , data['image_for_post']
                occlude_rate = data['occlude_rate'].item()
                # print("occlude_rate: ", occlude_rate)
                # save_image(image[0], "original.jpg")
                # save_image([gt[0], vm[0]], "gt.jpg")
                gt = gt.squeeze(1)
                gt = [np.array(x.cpu(), np.float32) for x in gt]
                gt = [x / (x.max() + 1e-8) for x in gt]
                image = image.to(device).squeeze(1)
                ensem_out = self.train_val_forward_fn(model, image=image, seg = vm, time_ensemble=True,
                                                        gt_sizes=[g.shape for g in gt], verbose=False)
                # pred_fm_crop: B,1,256,256
                pred_fm_crop = torch.cat(ensem_out["pred"], dim=0)
                 
                    
                 
                pred_fm = self.align_raw_size(pred_fm_crop, data['obj_position'], data["vm_pad"], data)
                # self.visualize(pred_vm, pred_fm, data, 'test', iter)
                # ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]
                
                pred_fm = pred_fm.squeeze()
                fm_no_crop = data["fm_no_crop"].squeeze()
                vm_no_crop = data["vm_no_crop"].squeeze()
                # print(pred_fm.shape)
                # print("fm no crop", fm_no_crop.shape)
                
                # pred_vm = pred_vm.squeeze()
                 
                pred_fm = (pred_fm > 0.5).to(torch.int64)
        
                intersection = ((pred_fm == 1) & (fm_no_crop == 1)).sum()
                union = ((pred_fm == 1) | (fm_no_crop == 1)).sum()

                intersection_rec.update(intersection)
                union_rec.update(union)

                if occlude_rate == 0.0:
                    intersection_rec_0.update(intersection)
                    union_rec_0.update(union)
                elif occlude_rate >= 0.01 and occlude_rate < 0.2:
                    intersection_rec_1.update(intersection)
                    union_rec_1.update(union)
                elif occlude_rate >= 0.2 and occlude_rate < 0.4:
                    intersection_rec_2.update(intersection)
                    union_rec_2.update(union)
                elif occlude_rate >= 0.4 and occlude_rate < 0.7:
                    intersection_rec_3.update(intersection)
                    union_rec_3.update(union)
                elif occ_iou >= 0.7:
                    intersection_rec_4.update(intersection)
                    union_rec_4.update(union)

                # for invisible mIoU
                 
                inv_intersection = ((pred_fm == 1) & (fm_no_crop == 1) & (vm_no_crop == 0)).sum()
                inv_union = (((pred_fm == 1) | (fm_no_crop == 1)) & (vm_no_crop == 0)).sum()
               
                if ((fm_no_crop - vm_no_crop).sum()) != 0:
                    inv_intersection_rec.update(inv_intersection)
                    inv_union_rec.update(inv_union)
                    Flag = 1
                else:
                    Flag = 0

                # loss_eval = self.loss_and_evaluation(pred_fm, data)
                # iou += loss_eval['iou']
                # iou_post += loss_eval['iou_post']
                # iou_count += loss_eval['iou_count']
                # invisible_iou_ += loss_eval['invisible_iou_']
                # invisible_iou_post += loss_eval['invisible_iou_post']
                # occ_count += loss_eval['occ_count']
                iou = intersection/(union+1e-6)
                occ_iou = inv_intersection/(inv_union+1e-6)
                list_iou.append(iou)
                list_inv_iou.append(occ_iou)
                self.logger.info('img_id: {}, anno_id: {}, IoU: {}, occ_IoU: {}, Flag: {}'.format(data['img_id'].item(), data['anno_id'].item(), iou, 
                occ_iou, Flag))

                # self.logger.info('img_id: {}, anno_id: {}, IoU: {}, IoU_post: {}, occ_IoU: {}, occ_IoU_post: {}'.format(data['img_id'].item(), data['anno_id'].item(), loss_eval['iou'].item(), 
                # loss_eval['iou_post'].item(), loss_eval['invisible_iou_'].item(), loss_eval['invisible_iou_post'].item()))

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        # ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()
        mean_iou = intersection_rec.sum / (union_rec.sum + 1e-10) # mIoU
        miou_0 = intersection_rec_0.sum / (union_rec_0.sum + 1e-10) # mIoU
        miou_1 = intersection_rec_1.sum / (union_rec_1.sum + 1e-10) # mIoU
        miou_2 = intersection_rec_2.sum / (union_rec_2.sum + 1e-10) # mIoU
        miou_3 = intersection_rec_3.sum / (union_rec_3.sum + 1e-10) # mIoU
        miou_4 = intersection_rec_4.sum / (union_rec_4.sum + 1e-10) # mIoU
        miou = (miou_0 + miou_1 + miou_2 + miou_3 + miou_4) / 5
        inv_miou = inv_intersection_rec.sum / (inv_union_rec.sum + 1e-10) # mIoU

        self.logger.info('meanIoU: {}/{}, meanIoU_0: {}, meanIoU_1: {}, meanIoU_2: {}, meanIoU_3: {}, meanIoU_4: {}'.format(miou, mean_iou, miou_0, miou_1, miou_2, miou_3, miou_4))
        # self.logger.info('meanIoU post-process: {}'.format(iou_post.item() / iou_count.item()))
        self.logger.info('meanIoU invisible: {}'.format(inv_miou))
        # self.logger.info('meanIoU invisible post-process: {}'.format(invisible_iou_post.item() / occ_count.item()))
        # self.logger.info('iou_count: {}'.format(iou_count))
        # self.logger.info('occ_count: {}'.format(occ_count))
        return miou , inv_miou 



    def loss_and_evaluation(self, pred_fm, meta, pred_vm=None):
        loss_eval = {}
        pred_fm = pred_fm.squeeze()
        counts = meta["counts"].reshape(-1).to(pred_fm.device)
        fm_no_crop = meta["fm_no_crop"].squeeze()
        vm_no_crop = meta["vm_no_crop"].squeeze()
        # pred_vm = pred_vm.squeeze()
        # post-process
        pred_fm = (pred_fm > 0.5).to(torch.int64)
        # pred_vm = (pred_vm > 0.5).to(torch.int64)
        
        iou, invisible_iou_, iou_count = evaluation_image((pred_fm > 0.5).to(torch.int64), fm_no_crop, counts, meta)
        loss_eval["iou"] = iou
        loss_eval["invisible_iou_"] = invisible_iou_
        loss_eval["occ_count"] = iou_count
        loss_eval["iou_count"] = torch.Tensor([1]).cuda()
        pred_fm_post = pred_fm + vm_no_crop
        
        pred_fm_post = (pred_fm_post>0.5).to(torch.int64)
        iou_post, invisible_iou_post, iou_count_post = evaluation_image(pred_fm_post, fm_no_crop, counts, meta)
        loss_eval["iou_post"] = iou_post
        loss_eval["invisible_iou_post"] = invisible_iou_post
        return loss_eval
    
    @torch.inference_mode()
    @run_on_seed
    def val_batch_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        ensemble_maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, vm = data['image'], data['gt'], data['name'], data['seg']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / (x.max() + 1e-8) for x in gt]
            image = image.to(device).squeeze(1)
            batch_res = []
            for i in range(5):
                ensem_out = self.train_val_forward_fn(model, image=image, seg = vm, time_ensemble=True, verbose=False)
                ensem_res = ensem_out["pred"].detach().cpu()
                batch_res.append(ensem_res)
            batch_res = torch.mean(torch.concat(batch_res, dim=1), dim=1, keepdim=True)
            for g, r, n in zip(gt, batch_res, name):
                ensemble_maes.append(cal_mae(g, r, thresholding, save_to, n))

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()

        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae

 
    def train(self):
        self.logger.info(f'Training on {self.train_dataset} Dataset.')
        accelerator = self.accelerator
         
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            # Train
            self.model.train()
        
            loss_sm = SmoothedValue(window_size=10) 
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for data in self.train_loader:
                    
                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        counts = data['counts'].squeeze().bool().sum()
                        if counts==0:
                            continue
                        loss  = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)
                        self.iteration+=1
                        accelerator.backward(loss)

                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    self.accelerator.log({'loss': loss_sm.avg, 'lr': self.opt.param_groups[0]['lr']})
                    
                    pbar.update()
                    # if self.iteration % 100 == 0:
                    #     output, _best_in_iou = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                    #     iou, invisible_iou_, img_id, anno_id = output['iou'], output['in_iou'], output['img_id'], output['anno_id']
                    #     self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch}, img_id: {img_id}, anno_id: {anno_id}, iou: {iou:.4f} invisible_iou:{invisible_iou_:.4f}')
                    #     accelerator.log({ 
                    #                     "img_id": output['img_id'],
                    #                     "anno_id": output['anno_id'],
                    #                     "iou": iou,
                    #                     "in_iou": invisible_iou_,})
                        # if iou == _best_in_iou:
                        #     self.save("best")

                    # if loss_sm.count >= 20:
                    #     break
            if self.scheduler is not None:
                self.scheduler.step()

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')
            
            # Val
            self.model.eval()
            # if (epoch + 1) % 10 == 0: #or (epoch >= self.train_num_epoch * 0.7):
            output, _best_in_iou = self.val_time_ensemble(self.model, self.test_loader, accelerator)
            iou, invisible_iou_, img_id, anno_id = output['iou'], output['in_iou'], output['img_id'], output['anno_id']
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch}, img_id: {img_id}, anno_id: {anno_id}, iou: {iou:.4f} invisible_iou:{invisible_iou_:.4f}')
            accelerator.log({ 
                                "img_id": output['img_id'],
                                "anno_id": output['anno_id'],
                                "iou": iou,
                                "in_iou": invisible_iou_,})
            
            self.save(self.cur_epoch, latest= True)
            if (epoch+1) % 5 == 0: 
                self.save(self.cur_epoch, latest= False)

            # Visualize
            with torch.inference_mode():
                if accelerator.is_main_process:
                    model = self.accelerator.unwrap_model(self.model)
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            out = fill_args_from_dict(self.train_val_forward_fn, data)(model=model,
                                                                                       verbose=False)
                            tracker.log(
                                {'img-pred-gt-vm':
                                     [wandb.Image(o[0, :, :]) for o in out.values()]
                                 })

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()
