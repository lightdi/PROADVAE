import os
from models.ZDecoder import Decoder
from models.ZEncoder import Encoder
from models.ZDiscriminator import Discriminator
from models.ZGenerator import Generator
import numpy as np
import torch as t
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from typing import List, Optional, Sequence, Union, Any, Callable
from utils.dataReader import get_test_batch
from sklearn.neighbors import KNeighborsClassifier
from torchvision.utils import save_image, make_grid
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as msef
import torch.nn.functional as F

#Visdom
import visdom


class PROGADVAE(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim, 
        w_multi: int = 64,
        noise_dim: int = 50,
        lr: float =0.0002,
        b1: float =0.5,
        b2: float =0.999,
        batch_size: int = 16,
        Nd: int = 50,
        max_epochs: int =5000,
        **kwargs):

        super(PROGADVAE, self).__init__()
        self.save_hyperparameters()

        self.vis = visdom.Visdom()
        
        self.automatic_optimization = False
        self.out_resolution = w_multi

        #networks
        data_shape = (channels, width, height)
        self.channels_num = channels

        self.G = Generator(latent_dim, w_multi,noise_dim).cuda()
        self.C = Discriminator(latent_dim, w_multi, Nd).cuda()
        self.E = Encoder(latent_dim, w_multi).cuda()
        self.D = Decoder(latent_dim, w_multi).cuda()

        self.g_loss = 0.0
        self.d_loss = 0.0

        self.loss_criterion = nn.CrossEntropyLoss()
        self.loss_criterion_gan = nn.BCEWithLogitsLoss()

        self.batch_size = self.hparams.batch_size
        self.Nd = Nd

        self.batch_ones_label = t.ones(self.batch_size).cuda()
        self.batch_zeros_label = t.zeros(self.batch_size).cuda()
        self.soft = t.nn.Softmax(dim = -1).cuda()
        
        #Controls of the progressive growing algorithm
        self.schedule = [[500, 1000, 2000, 3000], [16, 16, 16, 8],[5, 5, 5, 1]]
        #self.schedule = [[1, 2, 3, 4, 5], [16, 16, 16, 8, 4],[5, 5, 5, 1, 1]]
        self.max_epoch = max_epochs

        self.size = 8
        self.old_epoch = 0

    def forward(self, input: t.Tensor) -> t.Tensor:
        return input

    # Utility functions
    def toogle_grad(self,model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    

    def encoder_loss(self, x_real, eps):

        self.toogle_grad(self.E, True)
        self.toogle_grad(self.D, False)
        self.toogle_grad(self.G, False)
        self.toogle_grad(self.C, False)

        z_real_mu, z_real_log_sigma = self.E(x_real)
        z_real = z_real_mu + t.exp(z_real_log_sigma) * eps
        x_real_mu = self.D(z_real)


        #KL divergence
        kl_loss =  t.mean(
            t.sum(
                (z_real_mu**2 + t.exp(2*z_real_log_sigma))/2 - 1/2 - z_real_log_sigma
                , dim=1)
            ,-1)

        #
        ll_loss = t.mean(
            t.sum(
                1/2 * t.square(((x_real+0.0) - x_real_mu) / 1) 
                ,dim=(1,2,3))
            ,-1)

        encoder_loss = 1* kl_loss + ll_loss

        return encoder_loss
    
    def critic_loss(self, x_real, X_id_label,x_var, x_pro, z, eps):
        
        self.toogle_grad(self.E, False)
        self.toogle_grad(self.D, False)
        self.toogle_grad(self.G, False)
        self.toogle_grad(self.C, True)

        z_real_mu, z_real_log_sigma = self.E(x_real)
        c = z_real_mu + t.exp(z_real_log_sigma) * eps

        u1 = 5
        u2 = 0.5

        x_fake = self.G(t.cat([z,c],1))
        d_pro = self.C(x_pro)
        d_fake = self.C(x_fake)
        d_real = self.C(x_real)

        critic_loss = (self.loss_criterion_gan(d_pro[:,self.Nd], self.batch_ones_label) +
                    self.loss_criterion_gan(d_fake[:,self.Nd], self.batch_zeros_label))

        id_loss = self.loss_criterion(d_real[:, :self.Nd], X_id_label )

        var_loss = self.loss_criterion(d_real[:, self.Nd+1], x_var+0.0)

        critic_loss = critic_loss + u1 * id_loss  + u2 * var_loss
         
        return critic_loss
    
    def generator_loss(self, z, x_real, x_real_pro,  x_id_label, x_var, eps):

        self.toogle_grad(self.E, False)
        self.toogle_grad(self.D, True)
        self.toogle_grad(self.G, True)
        self.toogle_grad(self.C, False)

        o1 = 5
        o2 = 0.5
        o3 = 0.1

        z_pro = z.clone() 
        z_pro[:,0] = t.zeros(self.batch_size).cuda()

        z_real_mu, z_real_log_sigma = self.E(x_real)
        c = z_real_mu + t.exp(z_real_log_sigma) * eps

        z_real_pro_mu, z_real_pro_log_sigma = self.E(x_real_pro)

        c_pro = z_real_pro_mu + t.exp(z_real_pro_log_sigma) * eps
        x_pro_real = self.G(t.cat((z_pro,c_pro),1))
       

        x_pro = self.G(t.cat((z_pro,c),1))
        d_pro = self.C(x_pro)
        z_pro_mu, z_pro_log_sigma = self.E(x_pro)
        #gen_loss = -d_pro[:,self.Nd]
        gan_loss = self.loss_criterion_gan(d_pro[:,self.Nd], self.batch_ones_label)


        id_loss = self.loss_criterion(d_pro[:,:self.Nd], x_id_label)

        var_loss = self.loss_criterion(d_pro[:,self.Nd+1], t.zeros(self.batch_size).cuda())

        Index = (x_var == 0).nonzero().squeeze()

        rec_loss = (x_real[Index] - x_real_pro[Index]).pow(2).sum()/self.batch_size

        gen_loss = gan_loss + o1 * id_loss + o2 * var_loss + o3 * rec_loss
        
        #gen_loss = self.loss_criterion_gan(d_pro, self.batch_ones_label)

        w = (self.soft(-2*z_pro_log_sigma[Index])).detach()
        lat_loss = t.sum(
                    1/2* t.square(c[Index] - z_pro_mu[Index])* w) 
        
        g_loss =  t.sum(gen_loss + lat_loss)
        return g_loss, x_pro
    
    def decoder_loss(self, x_real, eps):
        
        self.toogle_grad(self.E, False)
        self.toogle_grad(self.D, True)
        self.toogle_grad(self.G, True)
        self.toogle_grad(self.C, False)

        z_real_mu, z_real_log_sigma = self.E(x_real)
        z_real = z_real_mu + t.exp(z_real_log_sigma) * eps
        x_real_mu = self.D(z_real)
    
        ll_loss = t.mean(t.sum(1/2 * t.square(((x_real+0.0) - x_real_mu) / 1) ,dim=(1,2,3)),-1)
        
        return ll_loss, x_real_mu

    def training_step(self, batch, batch_idx):

        #print (self.current_epoch) 
        #if batch_idx == 80:
            #print("Perto do fim")
        if self.current_epoch -1 in self.schedule[0]:
            #if (2**(self.G.depth+1)< self.out_resolution):
            if ((2*(self.size)) <= self.out_resolution) and (self.old_epoch != self.current_epoch):
                c = self.schedule[0].index(self.current_epoch-1)
                batch_size = self.schedule[1][c]
                growing = self.schedule[2][0]
                tot_iter_num = tot_iter_num  = (len(self.trainer.train_dataloader.dataset)/batch_size)

                self.D.growing_net(growing*tot_iter_num)
                self.G.growing_net(growing*tot_iter_num)
                self.C.growing_net(growing*tot_iter_num)
                self.E.growing_net(growing*tot_iter_num)

                self.size = 2*self.size
                self.old_epoch = self.current_epoch
                #self.size = 2**(self.G.depth+1)
                print("Output Resolution: %d x %d" % (self.size,self.size))

        print("Epoch: %i/%i" % (int(self.current_epoch),int(self.max_epoch)))

        batch_image, batch_id_label, batch_var, batch_pro = batch
        ## Atualizar
        if self.size != self.out_resolution:
            x_real =  F.interpolate(batch_image.cuda(), size=self.size)
            x_pro = F.interpolate(batch_pro.cuda(), size=self.size)
        else:
            x_real =  batch_image.cuda()
            x_pro = batch_pro.cuda()
        x_id_label = batch_id_label.cuda()
        x_var = batch_var.cuda()

        optimizer_E, optimizer_D, optimizer_G, optimizer_C = self.optimizers()   

        eps = t.normal(0., 1., size=(self.hparams.batch_size, self.hparams.latent_dim)).cuda()
        #xi = t.normal(0., 1., size=(self.hparams.batch_size, self.hparams.latent_dim)).cuda()
        z = t.normal(0., 1., size=(self.hparams.batch_size, self.hparams.noise_dim)).cuda()

        z[:,0] = x_var

        optimizer_E.zero_grad()
        e_loss = self.encoder_loss(x_real, eps)   
        self.manual_backward(e_loss)
        optimizer_E.step()  

        optimizer_C.zero_grad()
        c_loss= self.critic_loss(x_real, x_id_label, x_var, x_pro, z, eps)
        self.manual_backward(c_loss)
        optimizer_C.step()

        optimizer_G.zero_grad()
        g_loss, x_fake = self.generator_loss(z, x_real, x_pro,  x_id_label, x_var, eps)
        self.manual_backward(g_loss)
        optimizer_G.step()

        optimizer_D.zero_grad()
        d_loss, x_real_mu = self.decoder_loss(x_real, eps)
        self.manual_backward(d_loss)
        optimizer_D.step()

        self.results = [x_fake, x_real, x_pro, x_real_mu]

        #results = self.results

        #self.vis.images(results[0]/2+0.5,nrow=4,win='generated',
        #            opts={'title':"Generated"})
        #self.vis.images(results[1]/2+0.5,nrow=4,win='original', 
        #            opts={'title':"Original"})
        #self.vis.images(results[2]/2+0.5,nrow=4,win='prototype', 
        #            opts={'title':"Prototyped"})
        #self.vis.images(results[3]/2+0.5,nrow=4,win='Decode', 
        #            opts={'title':"Decode"})

        self.losses = {
                    "e_loss": e_loss,
                    "d_loss": d_loss,
                    "g_loss": g_loss, 
                    "c_loss": c_loss
                    }

        self.log_dict({
                    "e_loss": e_loss,
                    "d_loss": d_loss,
                    "g_loss": g_loss, 
                    "c_loss": c_loss
                    }, prog_bar=True)

    def training_epoch_end(self, outputs) -> None:

        """
        self.toogle_grad(self.E, False)
        self.toogle_grad(self.D, False)
        self.toogle_grad(self.G, False)
        self.toogle_grad(self.C, False)
        """

        results = self.results
        #x = vutils.make_grid(results[0], normalize=True, scale_each=True)
        #self.writer.add_image('Image', x, 1)

        self.vis.images(results[0]/2+0.5,nrow=4,win='generated',
                    opts={'title':"Generated"})
        self.vis.images(results[1]/2+0.5,nrow=4,win='original', 
                    opts={'title':"Original"})
        self.vis.images(results[2]/2+0.5,nrow=4,win='prototype', 
                    opts={'title':"Prototyped"})
        self.vis.images(results[3]/2+0.5,nrow=4,win='Decode', 
                    opts={'title':"Decode"})
        
        if self.current_epoch %10 == 0 :
            files = [
                    'Load_AR_test_50_4.txt'
                    ]
            for file in files:
                # Create training test
                base_dir = '/media/lightdi/CRUCIAL/Datasets/AR-Cropped/'
                test_file = 'dataset_file/' + file
                #test_file = '/media/lightdi/CRUCIAL/Datasets/CAS-PEAL.VDGAN/LoadPEAL_0_100_shuffle.txt'
                test_loader= get_test_batch(base_dir, test_file,1,shuffle=False,drop_last=False)
                self.G.eval()
                X = []
                y = []
                X_gen = []
                y_gen = []
                X_pro = []
                skip = 1
                for i, (x_real, x_id, x_var, x_prototype) in enumerate(test_loader):

                    x_real = x_real.cuda()
                    x_id = x_id.cuda()
                    x_var = x_var.cuda()
                    x_prototype = x_prototype.cuda()

                    ## Atualizar
                    if self.size != self.out_resolution:
                
                        x_real =  F.interpolate(x_real.cuda(), size=self.size)
                       
                        x_prototype = F.interpolate(x_prototype.cuda(), size=self.size)


                    #if x_var.item() == 0 :
                    #    print("Zero")
                    #if x_var.item() == 1 :
                    #   print("1")

                    # Generator
                    eps = t.normal(0., 1., size=(1, self.hparams.latent_dim)).cuda()        
                    z = t.normal(0., 1., size=(1, self.hparams.noise_dim)).cuda()  
                    z[:,0] = 0

                    z_real_mu, z_real_log_sigma = self.E(x_real)
                    c = z_real_mu + t.exp(z_real_log_sigma) * eps
                    
                    z_ = t.cat([z,c], 1)

                    x_synthetic = self.G(z_)

                    eps_pro = t.normal(0., 1., size=(1, self.hparams.latent_dim)).cuda()        
                    z_pro = t.normal(0., 1., size=(1, self.hparams.noise_dim)).cuda()
                    z_pro_mu, z_pro_log_sigma = self.E(x_real)
                    c_pro = z_pro_mu + t.exp(z_pro_log_sigma) * eps_pro
                    
                    z_p_ = t.cat([z_pro,c_pro], 1)

                    x_synthetic_pro = self.G(z_p_)

                    X.append(x_synthetic.detach().cpu().data.numpy())
                    X_gen.append(x_synthetic_pro.detach().cpu().data.numpy())
                    X_pro.append(x_prototype.detach().cpu().data.numpy())
                    y.append(int(x_id.detach().cpu().data.numpy()))
                    y_gen.append(int(x_id.detach().cpu().data.numpy()))

                    


                    #Visdom
                    batch_image = x_real.detach()
                    batch_recon = x_synthetic.detach() #x_synthetic.detach()
                    batch_proto = x_prototype.detach()
                    batch_proge = x_synthetic_pro.detach()
                    
                    #Save Image
                    save_dir_img = "img/file_{}_Epoch_{}".format(file,self.current_epoch)
                    if not os.path.exists(save_dir_img):
                        os.makedirs(save_dir_img)
                        
                        
                    grid = make_grid([batch_image.view(self.channels_num, self.size, self.size)/2+0.5, 
                                    batch_recon.view(self.channels_num, self.size, self.size)/2+0.5, 
                                    batch_proto.view(self.channels_num, self.size, self.size)/2+0.5,
                                    batch_proge.view(self.channels_num, self.size, self.size)/2+0.5], nrow=4, padding=2)
                    save_image(grid,   save_dir_img  + "/epoch_{}_Iteration_{}_Id_{}.bmp".format(self.current_epoch, i, x_id.item()))


                    batch_recon = batch_recon.cpu().data.numpy()/2+0.5
                    batch_image = batch_image.cpu().data.numpy()/2+0.5
                    batch_proto = batch_proto.cpu().data.numpy()/2+0.5
                    batch_proge= batch_proge.cpu().data.numpy()/2+0.5

                    
                    #if self.current_epoch %100 == 0:
                    #    self.vis.images(batch_image,nrow=1,win='original_test', opts={'title':"zOriginal"})
                    #    self.vis.images(batch_recon,nrow=1,win='generated_test', opts={'title':"zGenerated_Orig"})
                    #    self.vis.images(batch_proto,nrow=1,win='prototype_test', opts={'title':"zPrototype"})
                    #    self.vis.images(batch_proge,nrow=1,win='generated__test', opts={'title':"zGenerated_Pro"})

                self.G.train()
                X = np.array(X)
                X_gen = np.array(X_gen)
                X_pro = np.array(X_pro)
                y = np.array(y)
                y_gen = np.array(y_gen)

                X_1 = np.reshape(X, (X.shape[0], -1))

                X_gen_1 = np.reshape(X_gen, (X_gen.shape[0], -1))

                X_pro =   np.reshape(X_pro, (X_pro.shape[0], -1))

                #data_dir = datasets.load_files('./PEAL_pro_test/', shuffle='False')

                #data_gen_dir = datasets.load_files('./PEAL_gen_test/', shuffle='False')

                #X, y = data_dir['data'], data_dir['target_names']

                #X_gen, y_gen = data_gen_dir['data'], data_gen_dir['target_names']

                #X = np.array(X)
                #y = [ int(x) for x in y ]


                #X = [np.asarray(Image.open(io.BytesIO(x))) for x in X]
                
                X_1 = self.normalize_2d(X_1)
                
                X_gen_1 = self.normalize_2d(X_gen_1)

                X_pro = self.normalize_2d(X_pro)


                #for i in range (1):

                neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
                neigh.fit(X_1, y)

                neigh_pro = KNeighborsClassifier(n_neighbors=1, metric='cosine')
                neigh_pro.fit(X_pro, y)

                pred =neigh.predict(X_gen_1) 
                
                result = (pred == y_gen)
                fsupport = precision_recall_fscore_support(y_gen, result, average='micro')
                
                print_file = "File: {} Current Epoch: {}".format(file, self.current_epoch) 
                
                mse = 0 #t.nn.MSEloss(X_pro, X_gen_1).mean()

                for i in range (X_pro.shape[0]):
                    mse = mse + msef(X_pro[0], X_gen_1[0])
                
                mse = mse /X_pro.shape[0]



                for key in self.losses:
                    print_file +=  ",  " +  str(key) + ":" + str(self.losses[key].item())   
                
                print_file += '\n'
                print_file += "accuracy {}% for K={}".format(float((result.sum()/result.shape)*100),1)
                print_file += '\n' + "accuracy {}% for K={}".format(neigh.score(X_gen_1, y_gen),1) 
                print_file += '\n' + "accuracy_Pro {}% for K={}".format(neigh_pro.score(X_gen_1, y_gen)*100,1) 
                print_file += '\n' + "MSE meu {}%  MSE: {}".format(round(mse,8), round(msef(X_pro, X_gen_1),8)) 
                print_file += '\n' + "Precision : {} Recall : {} f1 : {} ".format(round(fsupport[0],4),round(fsupport[1],4),round(fsupport[2],4))

                self.log_dict({"accuracy": (float(result.sum()/result.shape)*100), 
                                "Accuracy_pro": (neigh_pro.score(X_gen_1, y_gen)*100),
                                "MSE": (round(mse,8)),
                                "step": self.current_epoch })

                sourceFile = open('result.txt', 'a')
                print('Resultados ' + print_file +  '\n', file = sourceFile)
                sourceFile.close()
                #tensorboard_logs['step'] = self.current_epoch
        
        return super().training_epoch_end(outputs)
        
    def normalize_2d(self, matrix):
        # Only this is changed to use 2-norm put 2 instead of 1
        norm = np.linalg.norm(matrix, 1)
        # normalized matrix
        matrix = matrix/norm  
        return matrix

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        optimizer_E = t.optim.Adam(self.E.parameters(),
                                        lr=lr, betas=(b1, b2))

        optimizer_D = t.optim.Adam(self.D.parameters(),
                                        lr=lr, betas=(b1, b2))

        optimizer_G = t.optim.Adam(self.G.parameters(),
                                        lr=0.0001, betas=(b1, b2))

        optimizer_C = t.optim.Adam(self.C.parameters(),
                                        lr=0.0003, betas=(b1, b2))
        

        return optimizer_E, optimizer_D, optimizer_G, optimizer_C
