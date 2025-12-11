import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import time
# add paths in model/__init__.py for new models
from models import * 

class CombinedModel(pl.LightningModule):
    def __init__(self, specs, point2gs=False):
        super().__init__()
        self.specs = specs
        self.point2gs = point2gs

        self.task = specs['training_task']

        if self.task in ('combined', 'modulation'):
            self.gs_model = GsModel(specs=specs, point2gs=point2gs) 

            feature_dim = specs["GSModelSpecs"]["latent_dim"]
            modulation_dim = feature_dim*3
            latent_std = specs.get("latent_std", 0.25)
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 

    def training_step(self, x, idx):
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        if x is None:
            return None
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.gs_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
                # "monitor": "total"
                # }
        }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def train_modulation(self, x):
        occ_xyz = x['occ_xyz']
        occ = x['occ']
        gt = x['gt_gaussian']
        gs = x['gaussians']
        gaussian_xyz = x['gaussian_xyz']

        if self.point2gs:
            plane_features = self.gs_model.pointnet.get_plane_features(gaussian_xyz)
        else:
            plane_features = self.gs_model.pointnet.get_plane_features(gs)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features)
        reconstructed_plane_feature, latent = out[0], out[-1]

        pred_color, pred_gs = self.gs_model.forward_with_plane_features(reconstructed_plane_feature, gaussian_xyz)
        pred_occ = self.gs_model.forward_with_plane_features_occ(reconstructed_plane_feature, occ_xyz)
        
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        color_loss = F.l1_loss(pred_color.squeeze()[:,:,0:48], gt.squeeze()[:,:,0:48], reduction='none')
        color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

        scale_loss = F.l1_loss(pred_gs.squeeze()[:,:,0:3], gt.squeeze()[:,:,49:52], reduction='none')
        scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()
        rotation_loss = F.l1_loss(pred_gs.squeeze()[:,:,3:7], gt.squeeze()[:,:,52:56], reduction='none')
        rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

        occ_loss = F.l1_loss(pred_occ.squeeze(), occ.squeeze(), reduction='none')
        occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

        loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        loss_dict =  {"loss": loss, "color": color_loss, "vae": vae_loss, "occ": occ_loss, "scale": scale_loss, "rotation": rotation_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss


    def train_diffusion(self, x):

        self.train()

        context = x.get('context', None)
        # context = x['context'] # (B, 1024, 3) or False if unconditional
        latent = x['latent'] # (B, D)

        # unconditional training if cond is None 
        cond = context if self.specs['diffusion_model_specs']['cond'] else None 

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, _, __ = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

