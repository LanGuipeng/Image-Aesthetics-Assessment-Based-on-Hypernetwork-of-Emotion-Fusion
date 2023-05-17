import logging
import time
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import PLCC,SROCC,RMSE,MAE
from common import AverageMeter, Transform
from dataset import AVADataset,EmotionDataset
from emd_loss import EDMLoss
import models
from argparse import ArgumentParser
import os
from visdom import Visdom
import numpy as np
import functools
from scipy import stats
logger = logging.getLogger(__file__)
import torch.nn.functional as F
#指标：
# MSE = np.mean(np.square(y - y_hat))
# RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
# MAE = np.mean(np.abs(y-y_hat))


#path_to_save_csv='/myDockerShare/zys/emotion/csv'
def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')
def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

logging = get_logger('/myDockerShare/zys/aes-emotion/experiment/log_result.txt')

def get_dataloaders(
    path_to_save_csv: Path, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = Transform()

    # train_ds = AVADataset(os.path.join(path_to_save_csv , "train.csv"), '/myDockerShare/zys/ava/aes/train', transform.train_transform)
    # val_ds = AVADataset(os.path.join(path_to_save_csv , "val.csv"), '/myDockerShare/zys/ava/aes/val', transform.val_transform)
    # test_ds = AVADataset(os.path.join(path_to_save_csv , "test.csv"), '/myDockerShare/zys/ava/aes/test', transform.val_transform)

    train_ds = EmotionDataset(os.path.join(path_to_save_csv , "train2.csv"), '/myDockerShare/zys/ava/emotion_data/pic', transform.train_transform)
    val_ds = EmotionDataset(os.path.join(path_to_save_csv , "val2.csv"), '/myDockerShare/zys/ava/emotion_data/pic', transform.val_transform)
    test_ds = EmotionDataset(os.path.join(path_to_save_csv , "test2.csv"), '/myDockerShare/zys/ava/emotion_data/pic', transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader


# def validate_and_test(
#     path_to_save_csv: Path,
#     # path_to_images: Path,
#     batch_size: int,
#     num_workers: int,
#     drop_out: float,
#     path_to_model_state: Path,
# ) -> None:
#     _, val_loader, test_loader = get_dataloaders(
#         path_to_save_csv=path_to_save_csv,  batch_size=batch_size, num_workers=num_workers
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     criterion = EDMLoss().to(device)

#     best_state = torch.load(path_to_model_state)

#     model = create_model(best_state["model_type"], drop_out=drop_out).to(device)
#     model.load_state_dict(best_state["state_dict"])

#     model.eval()
#     validate_losses = AverageMeter()
#     with torch.no_grad():
#         for (x, y) in tqdm(val_loader):
#             x = x.to(device)
#             y = y.to(device)
#             y_pred = model(x)
#             loss = criterion(p_target=y, p_estimate=y_pred)
#             validate_losses.update(loss.item(), x.size(0))

#     test_losses = AverageMeter()
#     sub_t=[]
#     obj_t=[]
#     with torch.no_grad():
#         for (x, y) in tqdm(test_loader):
#             x = x.to(device)
#             y = y.to(device)
#             for i in range(len(y)):
#                 sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
#                 sub_t = sub_t+sub.flatten().cpu().detach().numpy().tolist()
#             y_pred = model(x)
#             for i in range(len(y_pred)):
#                     obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
#                     obj_t = obj_t+obj.flatten().cpu().detach().numpy().tolist()
#             loss = criterion(p_target=y, p_estimate=y_pred)
#             test_losses.update(loss.item(), x.size(0))
#         X1 = pd.Series(sub_t)
#         Y1 = pd.Series(obj_t)
#         plcc_t = X1.corr(Y1, method="pearson")
#         srocc_t = X1.corr(Y1, method="spearman")
#         print('TEST','plcc:',plcc_t,'srocc',srocc_t)

#     logger.info(f"val loss {validate_losses.avg}; test loss {test_losses.avg}")
#     logging("TEST Results - PLCC: {:.4f} SROCC: {:.4f} "
#                     .format(plcc_t, srocc_t ))

# def get_optimizer(optimizer_type: str, model: NIMA, init_lr: float) -> torch.optim.Optimizer:
#     if optimizer_type == "adam":
#         optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.01)
#     elif optimizer_type == "sgd":
#         optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.5, weight_decay=9)
#     else:
#         raise ValueError(f"not such optimizer {optimizer_type}")
#     return optimizer
# viz = Visdom(env='AES')
# x_label, y_label = 0, 0
# win1 = viz.line(X=np.array([x_label]), Y=np.array([y_label]), opts=dict(title='train_loss'))
# win2 = viz.line(X=np.array([x_label]), Y=np.array([y_label]), opts=dict(title='val_loss'))
# win3 = viz.line(X=np.array([x_label]), Y=np.array([y_label]), opts=dict(title='train_acc'))
# win4 = viz.line(X=np.array([x_label]), Y=np.array([y_label]), opts=dict(title='test_acc'))



# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.init_lr * (0.95 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     print('epoch:',epoch,'lr:', lr)

class Trainer:
    def __init__(
        self,
        *,
        path_to_save_csv: Path,
        # path_to_images: Path,
        num_epoch: int,
        # model_type: str,
        num_workers: int,
        batch_size: int,
        init_lr: float,
        lr_ratio: float,
        weight_decay: float,
        experiment_dir: Path,
        # drop_out: float,
        # optimizer_type: str,
    ):

        train_loader, val_loader, test_loader= get_dataloaders(
            path_to_save_csv=path_to_save_csv,
            # path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # predict_model = torch.load('/myDockerShare/zys/aes-emotion/experiment/fusion4.pth')
        self.model_hyper= models.HyperNet(16, 112, 224, 112, 98, 49, 8, 7).cuda()
        # net_dict = self.model_hyper.state_dict()
        # state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        # self.model_hyper.load_state_dict(net_dict)
        self.model_hyper.train(True)
        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = init_lr
        self.lrratio =lr_ratio
        self.weight_decay = weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.optimizer  = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        self.scheduler=torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=5,gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(os.path.join(experiment_dir , "logs")))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100

    # def train_model(self):
    #     # best_loss = float("inf")
    #     # best_srocc=0
    #     # best_state = None
    #     for e in range(1, self.num_epoch + 1):
    #         train_loss, plcc, srocc ,mse= self.train()
    #         print('epoch:',e,'plcc:',plcc,'srocc:',srocc,'mse',mse)
    #         val_loss,plcc_v,srocc_v,mse = self.validate()
    #         print('val-loass:',val_loss,'plcc:',plcc_v,'srocc:',srocc_v,'mse',mse)
    #         logging("VAL Results - Epoch: {} PLCC: {:.4f} SROCC: {:.4f} MSE:{:.4f} "
    #                 .format(e, plcc_v, srocc_v ,mse))
    #         self.scheduler.step(metrics=val_loss)

    #         self.writer.add_scalar("train/loss", train_loss, global_step=e)
    #         self.writer.add_scalar("val/loss", val_loss, global_step=e)
           

    #         # if best_state is None or srocc_v > best_srocc:
    #         #     logger.info(f"updated loss from {best_srocc} to {srocc_v}")
    #         #     best_srocc = srocc_v
    #         #     best_state = {
    #         #         "state_dict": self.model.state_dict(),
    #         #         "model_type": self.model_type,
    #         #         "epoch": e,
    #         #         "best_loss": best_loss,
    #         #     }
    #         #     torch.save(best_state, os.path.join(self.experiment_dir ,"best_state.pth"))

    #         if e % 1 == 0:
    #             plcc_t,srocc_t,mse =self.test()
    #             logging("TEST Results - Epoch: {} PLCC: {:.4f} SROCC: {:.4f} MSE:{:.4f}  "
    #                 .format(e, plcc_t, srocc_t ,mse))

    def train(self):
        # plcc=PLCC()
        # srocc=SROCC()
        # rmse=RMSE()
        
        
        best_acc=0

        train_losses = AverageMeter()
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size
        for t in range(0, self.num_epoch ):
            train_acc=0
            # path_to_model_state=os.path.join(self.experiment_dir ,"best_state.pth")
            # best_state = torch.load('/myDockerShare/zys/aes-emotion/experiment/fusion4.pth')
            # self.model_hyper.load_state_dict(best_state["state_dict1"])
          
            for idx, (x, y) in enumerate(self.train_loader):
                s = time.monotonic()
                # self.model_hyper.load_state_dict(best_state["state_dict1"])
                self.model_hyper.train()
                x = x.to(self.device)
                y = y.to(self.device)
                # print('y:',y)
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models.TargetNet(paras).cuda()
                # model.load_state_dict(best_state["state_dict"])
                model.train()
                # for param in model.parameters():
                #         param.requires_grad = True
                y_pred = model(paras['target_in_vec'])
                # print('pred',y_pred)
                loss =self.criterion(y_pred, y)
                # loss = self.criterion(y_pred.squeeze(), y.float().detach())
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                train_losses.update(loss.item(), x.size(0))
                acc = (y_pred.argmax(dim=1) == y).float().mean()
                train_acc +=(acc /len(self.train_loader)).cpu().detach().numpy() 
     
                
                self.writer.add_scalar("train/current_loss", train_losses.val, self.global_train_step)
                self.writer.add_scalar("train/avg_loss", train_losses.avg, self.global_train_step)
                
                self.global_train_step += 1

                e = time.monotonic()
                if idx % self.print_freq:
                    log_time = self.print_freq * (e - s)
                    eta = ((total_iter - idx) * log_time) / 60.0
                    print(f"iter #[{idx}/{total_iter}] " f"loss = {loss:.3f} " f"time = {log_time:.2f} " f"eta = {eta:.2f}")


            print('Train==>','epoch:',t+1,'train_loss:',train_losses.avg, 'Acc:',train_acc)
            
            # viz.line(X=np.array([t+1]), Y=np.array([train_losses.avg]), win=win1, update='append')
            # viz.line(X=np.array([t+1]), Y=np.array([train_acc]), win=win3, update='append')
  
            validate_losses,val_acc= self.validate()
            # self.scheduler.step(metrics=validate_losses)
            # viz.line(X=np.array([t+1]), Y=np.array([validate_losses]), win=win2, update='append')
            self.scheduler.step()

            logging("VAL Results - Epoch: {} ACC: {:.4f} "
                    .format(t+1, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    "state_dict": model.state_dict(),
                    "state_dict1": self.model_hyper.state_dict(),
                    "epoch": t,
                    "best_loss": validate_losses,
                }
                torch.save(best_state, os.path.join(self.experiment_dir ,"best_state1.pth"))
        
            print('epoch:',t + 1, 'val_acc:',val_acc)
            
            test_acc= self.test()
            logging("TEST Results - Epoch: {} ACC: {:.4f}"
                    .format(t+1, test_acc))

            # viz.line(X=np.array([t+1]), Y=np.array([test_acc]), win=win4, update='append')
                 
                # Update optimizer
            # lr = self.lr / pow(10, (t // 6))
            # if t > 8:
            #     self.lrratio = 1
            # self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
            #                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
            #                 ]
            # self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

          
        return best_acc

    # def validate(self):
    #     """validate"""
    #     self.model_hyper.eval()
    #     sub_v = []
    #     obj_v = []
    #     validate_losses = AverageMeter()
    #     # for idx, (img, label) in enumerate(self.val_loader):
    #     #     # Data.
    #     #     img = img.to(self.device)
    #     #     label = label.to(self.device)

    #     #     paras = self.model_hyper(img)
    #     #     model_target = models.TargetNet(paras).cuda()
    #     #     model_target.train(False)
    #     #     y_pred = model_target(paras['target_in_vec'])
    #     #     # loss = self.criterion(p_target=label, p_estimate=y_pred)
    #     #     loss = self.criterion(y_pred.squeeze(), label.float().detach())
    #     #     validate_losses.update(loss.item(), img.size(0))
    #     #     obj_v = obj_v + y_pred.cpu().tolist()
    #     #     sub_v = sub_v + label.cpu().tolist()
    #     # X1 = pd.Series(sub_v)
    #     # Y1 = pd.Series(obj_v)
    #     # val_plcc = X1.corr(Y1, method="pearson")
    #     # val_srcc = X1.corr(Y1, method="spearman")
    #     # val_mse= np.mean(np.square(np.array(obj_v)-np.array(sub_v)))
    #     for idx, (x, y) in enumerate(self.val_loader):
               
    #         x = x.to(self.device)
    #         y = y.to(self.device)
    #         paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
    #         model = models.TargetNet(paras).cuda()
    #         model.eval()
    #             # for param in model.parameters():
    #             #         param.requires_grad = False
    #         y_pred = model(paras['target_in_vec'])
    #         # obj_v = obj_v + y_pred.cpu().tolist()
    #         # sub_v = sub_v + y.cpu().tolist()
    #         obj_v.append(y_pred)
    #         sub_v.append(y)
            
    #             # loss = self.criterion(p_target=y, p_estimate=y_pred)
    #         loss = self.criterion(y_pred.squeeze(), y.float().detach())
    #         validate_losses.update(loss.item(), x.size(0))
    #     print('obj:',obj_v)
    #     print('sub:',sub_v)    
    #     X1 = pd.Series(sub_v)
    #     Y1 = pd.Series(obj_v)
    #     val_plcc = X1.corr(Y1, method="pearson")
    #     val_srcc = X1.corr(Y1, method="spearman")
    #     val_mse= np.mean(np.square(np.array(obj_v)-np.array(sub_v)))
    #     print('VAL==>','VAL_loss:', validate_losses.avg, 'plcc:',val_plcc,'srocc:',val_srcc,'MSE',val_mse)

    
    #     return validate_losses.avg,val_srcc, val_plcc,val_mse

    # def test(self):
    #     """Testing"""
    #     sub_t = []
    #     obj_t = []
    #     # path_to_model_state=os.path.join(self.experiment_dir ,"best_state.pth")
    #     # best_state = torch.load(path_to_model_state)
    #     # self.model_hyper.load_state_dict(best_state["state_dict1"])
    #     self.model_hyper.eval()
    #     with torch.no_grad():
    #         for idx, (img, label) in enumerate(self.test_loader):
    #             # Data.
    #             img = img.to(self.device)
    #             label = label.to(self.device)

    #             paras = self.model_hyper(img)
    #             model_target = models.TargetNet(paras).cuda()
    #             # model_target.load_state_dict(best_state["state_dict"])
    #             model_target.eval()
    #             y_pred = model_target(paras['target_in_vec'])

    #             obj_t = obj_t + y_pred.cpu().tolist()
    #             sub_t = sub_t + label.cpu().tolist()
    #     X1 = pd.Series(sub_t)
    #     Y1 = pd.Series(obj_t)
    #     test_plcc = X1.corr(Y1, method="pearson")
    #     test_srcc = X1.corr(Y1, method="spearman")
    #     test_mse= np.mean(np.square(np.array(obj_t)-np.array(sub_t)))
    #     print('test==','plcc:', test_plcc,'srcc:',test_srcc,'mse:',test_mse)
    
    #     return test_srcc, test_plcc,test_mse

    def validate(self):
        self.model_hyper.train(False)
        validate_losses = AverageMeter()
    
        val_acc=0
       
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models.TargetNet(paras).cuda()
                model.train(False)
                y_pred = model(paras['target_in_vec'])
           
                loss =self.criterion(y_pred, y)
                validate_losses.update(loss.item(), x.size(0))

                acc = (y_pred.argmax(dim=1) == y).float().mean()
                val_acc +=(acc /len(self.val_loader)) .cpu().detach().numpy() 
        
                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1

        print('val---','acc:',val_acc)
        return validate_losses.avg,val_acc

    def test(self):
        """Testing"""
        
        path_to_model_state=os.path.join(self.experiment_dir ,"best_state1.pth")
        best_state = torch.load(path_to_model_state)
        self.model_hyper.load_state_dict(best_state["state_dict1"])
        self.model_hyper.eval()
        with torch.no_grad():
            test_acc=0
            for idx, (img, label) in enumerate(self.test_loader):
                # Data.
                img = img.to(self.device)
                label = label.to(self.device)

                paras = self.model_hyper(img)
                model_target = models.TargetNet(paras).cuda()
                model_target.load_state_dict(best_state["state_dict"])
                model_target.eval()
                y_pred = model_target(paras['target_in_vec'])

                acc = (y_pred.argmax(dim=1) == label).float().mean()
                test_acc +=(acc /len(self.test_loader)) .cpu().detach().numpy() 
       
        print('test==','acc:', test_acc)
    
        return test_acc

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser=ArgumentParser(description='NIMA')
    parser.add_argument('--path_to_save_csv',type=str,default='/myDockerShare/zys/emotion/csv',help='data-label')
    parser.add_argument('--experiment_dir',type=Path,default='/myDockerShare/zys/aes-emotion/experiment',help='experiment_dir')
    parser.add_argument('--num_epoch',type=int,default=50, help='eopch')
    parser.add_argument('--model_type',type=str,default='resnet50',help=' model_type')
    parser.add_argument('--optimizer_type',type=str,default='adam',help=' optimizer_type')
    parser.add_argument('--num_workers',type=int,default=8, help='num_workers')
    parser.add_argument('--batch_size',type=int,default=32, help='batch_size')
    parser.add_argument('--init_lr',type=float,default=2e-5, help='init_lr')
    parser.add_argument('--lr_ratio',type=float,default=10,help='lr_ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--drop_out',type=float,default=0.75, help='drop_out')
    args=parser.parse_args()
    slover=Trainer(path_to_save_csv=args.path_to_save_csv,num_epoch=args.num_epoch,num_workers=args.num_workers,batch_size=args.batch_size,init_lr=args.init_lr,
                lr_ratio=args.lr_ratio,weight_decay=args.weight_decay,experiment_dir=args.experiment_dir)
    best_acc= slover.train()
    # main(args.path_to_save_csv,args.num_epoch,args.model_type,args.num_workers,args.batch_size,args.init_lr,
    #         args.experiment_dir,args.drop_out,args.optimizer_type)

