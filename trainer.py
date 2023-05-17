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
from dataset import AVADataset
from emd_loss import EDMLoss
import models
from argparse import ArgumentParser
import os
import numpy as np
import functools
logger = logging.getLogger(__file__)

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

    train_ds = AVADataset(os.path.join(path_to_save_csv , "train.csv"), '/myDockerShare/zys/ava/aes/train', transform.train_transform)
    val_ds = AVADataset(os.path.join(path_to_save_csv , "val.csv"), '/myDockerShare/zys/ava/aes/val', transform.val_transform)
    test_ds = AVADataset(os.path.join(path_to_save_csv , "test.csv"), '/myDockerShare/zys/ava/aes/test', transform.val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_ds = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_ds


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

        train_loader, val_loader, _ = get_dataloaders(
            path_to_save_csv=path_to_save_csv,
            # path_to_images=path_to_images,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = _
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
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


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode="min", patience=10)
        self.criterion = EDMLoss().to(self.device)
        

        experiment_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(str(os.path.join(experiment_dir , "logs")))
        self.num_epoch = num_epoch
        self.global_train_step = 0
        self.global_val_step = 0
        self.print_freq = 100

    def train_model(self):
        # best_loss = float("inf")
        # best_srocc=0
        # best_state = None
        for e in range(1, self.num_epoch + 1):
            train_loss, plcc, srocc ,mse= self.train()
            print('epoch:',e,'plcc:',plcc,'srocc:',srocc,'mse',mse)
            val_loss,plcc_v,srocc_v,mse = self.validate()
            print('val-loass:',val_loss,'plcc:',plcc_v,'srocc:',srocc_v,'mse',mse)
            logging("VAL Results - Epoch: {} PLCC: {:.4f} SROCC: {:.4f} MSE:{:.4f} "
                    .format(e, plcc_v, srocc_v ,mse))
            self.scheduler.step(metrics=val_loss)

            self.writer.add_scalar("train/loss", train_loss, global_step=e)
            self.writer.add_scalar("val/loss", val_loss, global_step=e)
           

            # if best_state is None or srocc_v > best_srocc:
            #     logger.info(f"updated loss from {best_srocc} to {srocc_v}")
            #     best_srocc = srocc_v
            #     best_state = {
            #         "state_dict": self.model.state_dict(),
            #         "model_type": self.model_type,
            #         "epoch": e,
            #         "best_loss": best_loss,
            #     }
            #     torch.save(best_state, os.path.join(self.experiment_dir ,"best_state.pth"))

            if e % 1 == 0:
                plcc_t,srocc_t,mse =self.test()
                logging("TEST Results - Epoch: {} PLCC: {:.4f} SROCC: {:.4f} MSE:{:.4f}  "
                    .format(e, plcc_t, srocc_t ,mse))

    def train(self):
        # plcc=PLCC()
        # srocc=SROCC()
        # rmse=RMSE()
        sub1=[]
        obj1=[]
        self.model.train()
        train_losses = AverageMeter()
        total_iter = len(self.train_loader.dataset) // self.train_loader.batch_size

        for idx, (x, y) in enumerate(self.train_loader):
            s = time.monotonic()

            x = x.to(self.device)
            y = y.to(self.device)
            for i in range(len(y)):
                sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
                sub1 = sub1+sub.flatten().cpu().detach().numpy().tolist()
            paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
            model = models.TargetNet(paras).cuda()
            for param in model.parameters():
                    param.requires_grad = False
            y_pred = model(paras['target_in_vec'])
            for i in range(len(y_pred)):
                obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                obj1 = obj1+obj.flatten().cpu().detach().numpy().tolist()
            loss = self.criterion(p_target=y, p_estimate=y_pred)
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            train_losses.update(loss.item(), x.size(0))
         
            
            self.writer.add_scalar("train/current_loss", train_losses.val, self.global_train_step)
            self.writer.add_scalar("train/avg_loss", train_losses.avg, self.global_train_step)
            
            self.global_train_step += 1

            e = time.monotonic()
            if idx % self.print_freq:
                log_time = self.print_freq * (e - s)
                eta = ((total_iter - idx) * log_time) / 60.0
                print(f"iter #[{idx}/{total_iter}] " f"loss = {loss:.3f} " f"time = {log_time:.2f} " f"eta = {eta:.2f}")

        X1 = pd.Series(sub1)
        Y1 = pd.Series(obj1)
        plcc = X1.corr(Y1, method="pearson")
        srocc = X1.corr(Y1, method="spearman")
        MSE = np.mean(np.square(obj1-sub1))
        # print('plcc:',plcc,'srocc:',srocc)

        return train_losses.avg, plcc, srocc,MSE

        

    def validate(self):
        self.model_hyper.train(False)
        validate_losses = AverageMeter()
        sub_v=[]
        obj_v=[]
        best_srocc=0
        best_state = None
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for i in range(len(y)):
                    sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
                    sub_v = sub_v+sub.flatten().cpu().detach().numpy().tolist()
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models.TargetNet(paras).cuda()
                model.eval()
                for param in model.parameters():
                        param.requires_grad = False
                y_pred = model(paras['target_in_vec'])
                for i in range(len(y_pred)):
                    obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                    obj_v = obj_v+obj.flatten().cpu().detach().numpy().tolist()
                loss = self.criterion(p_target=y, p_estimate=y_pred)
                validate_losses.update(loss.item(), x.size(0))

                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1
        X1 = pd.Series(sub_v)
        Y1 = pd.Series(obj_v)
        plcc = X1.corr(Y1, method="pearson")
        srocc = X1.corr(Y1, method="spearman")
        MSE = np.mean(np.square(obj_v-sub_v))
        return validate_losses.avg,plcc,srocc,MSE
    def test(self):
        validate_losses = AverageMeter()
        sub_t=[]
        obj_t=[]
        best_srocc=0
        best_state = None
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                for i in range(len(y)):
                    sub= y[i][0]*1+y[i][1]*2+y[i][2]*3+y[i][3]*4+y[i][4]*5+y[i][5]*6+y[i][6]*7+y[i][7]*8+y[i][8]*9+y[i][9]*10
                    sub_t = sub_t+sub.flatten().cpu().detach().numpy().tolist()
                paras = self.model_hyper(x)  # 'paras' contains the network weights conveyed to target network
                model = models.TargetNet(paras).cuda()
                model.eval()
                model.load_state_dict(best_state["state_dict"])
                for param in model.parameters():
                        param.requires_grad = False
                y_pred = model(paras['target_in_vec'])
                for i in range(len(y_pred)):
                    obj= y_pred[i][0]*1+y_pred[i][1]*2+y_pred[i][2]*3+y_pred[i][3]*4+y_pred[i][4]*5+y_pred[i][5]*6+y_pred[i][6]*7+y_pred[i][7]*8+y_pred[i][8]*9+y_pred[i][9]*10
                    obj_t = obj_t+obj.flatten().cpu().detach().numpy().tolist()
                loss = self.criterion(p_target=y, p_estimate=y_pred)
                validate_losses.update(loss.item(), x.size(0))

                self.writer.add_scalar("val/current_loss", validate_losses.val, self.global_val_step)
                self.writer.add_scalar("val/avg_loss", validate_losses.avg, self.global_val_step)
                self.global_val_step += 1
        X1 = pd.Series(sub_t)
        Y1 = pd.Series(obj_t)
        plcc_t = X1.corr(Y1, method="pearson")
        srocc_t = X1.corr(Y1, method="spearman")
        MSE = np.mean(np.square(obj_t-sub_t))
        return plcc_t,srocc_t,MSE

if __name__=='__main__':
  
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
    train_losses,plcc, srocc = slover.train_model()
    # main(args.path_to_save_csv,args.num_epoch,args.model_type,args.num_workers,args.batch_size,args.init_lr,
    #         args.experiment_dir,args.drop_out,args.optimizer_type)

