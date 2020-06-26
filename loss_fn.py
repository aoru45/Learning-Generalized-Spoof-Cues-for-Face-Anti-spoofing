import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class TripletLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super(TripletLoss,self).__init__()
        self.margin = margin
    def forward(self, f_anchor, f_positive, f_negative): # (-1,c)
        f_anchor, f_positive, f_negative = renorm(f_anchor), renorm(f_positive), renorm(f_negative)
        b = f_anchor.size(0)
        f_anchor = f_anchor.view(b,-1)
        f_positive = f_positive.view(b,-1)
        f_negative = f_negative.view(b, -1)
        with torch.no_grad():
            idx = hard_samples_mining(f_anchor, f_positive, f_negative, self.margin)
        
        d_ap = torch.norm(f_anchor[idx] - f_positive[idx], dim = 1)  # (-1,1)
        d_an = torch.norm(f_anchor[idx] - f_negative[idx], dim = 1)
        return torch.clamp(d_ap - d_an + self.margin,0).mean()
        


def hard_samples_mining(f_anchor,f_positive, f_negative, margin):
    d_ap = torch.norm(f_anchor - f_positive, dim = 1)
    d_an = torch.norm(f_anchor - f_negative, dim = 1)
    idx = (d_ap - d_an) < margin
    return idx 
def renorm(x): # Important for training!
    # renorm in batch axis to make sure every vector is in the range of [0,1]
    # important !
    return x.renorm(2,0,1e-5).mul(1e5)
class TotalLoss(nn.Module):
    def __init__(self,margin = 0.2):
        super(TotalLoss, self).__init__()
        self.margin = margin
        self.trip = TripletLoss(margin)
        self.reg = nn.MSELoss()
        self.cla = nn.CrossEntropyLoss()
        
    def forward(self, regression, classification, feat, labels):
        regression_anchor, regression_positive, regression_negative = regression
        b,c,_,_ = regression_anchor.size()
        classification_anchor, classification_positive, classification_negative = classification
        
        feat_anchor, feat_positive, feat_negative = feat
        reg_loss = self.reg(regression_negative[labels == 1], torch.zeros_like(regression_negative[labels == 1]).cuda(device)) + self.reg(regression_anchor[labels == 0], torch.zeros_like(regression_anchor[labels == 0]).cuda(device)) + self.reg(regression_positive[labels == 0], torch.zeros_like(regression_positive[labels == 0]).cuda(device))
        cla_loss =      self.cla(classification_anchor[labels==0], torch.tensor([1] * classification_anchor[labels==0].size(0), dtype = torch.long).cuda(device)) + \
                        self.cla(classification_anchor[labels==1], torch.tensor([0] * classification_anchor[labels==1].size(0), dtype = torch.long).cuda(device)) +  \
                        self.cla(classification_positive[labels==0], torch.tensor([1] * classification_positive[labels==0].size(0), dtype = torch.long).cuda(device)) + \
                        self.cla(classification_positive[labels==1], torch.tensor([0] * classification_positive[labels==1].size(0), dtype = torch.long).cuda(device)) + \
                        self.cla(classification_negative[labels==0], torch.tensor([0] * classification_negative[labels==0].size(0), dtype = torch.long).cuda(device)) + \
                        self.cla(classification_negative[labels==1], torch.tensor([1] * classification_negative[labels==1].size(0), dtype = torch.long).cuda(device))
        trip_loss = sum([self.trip(a,b,c) for a,b,c in zip(feat_anchor, feat_positive, feat_negative)])
        return reg_loss + cla_loss + trip_loss
if __name__ == "__main__":
    regression = [torch.randn(1,3,24,24), torch.randn(1,3,24,24), torch.randn(1,3,24,24)]
    classification = [torch.randn(1,2), torch.randn(1,2), torch.randn(1,2)]
    feat = [[torch.randn(1,16),torch.randn(1,16)],[torch.randn(1,16),torch.randn(1,16)],[torch.randn(1,16),torch.randn(1,16)]]
    labels = torch.tensor([0],dtype = torch.long)
    loss_fn = TotalLoss()
    res = loss_fn(regression, classification, feat, labels)
