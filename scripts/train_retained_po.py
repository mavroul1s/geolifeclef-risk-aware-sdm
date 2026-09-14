"""v22 matched expert arms, complementary checkpoint selection, PO retention probes."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from geolifeclef.po_expert import POExpert, masked_po_loss
from geolifeclef.losses import asymmetric_loss
from geolifeclef.utils import set_seed
from scripts.ood_po_protocol import sample_f1
from scripts.v22_protocol import SEED, CHECKPOINT_POLICY, mix


class RetainedPOExpert(nn.Module):
    """PA and PO heads share an encoder; inference always uses the PA head."""
    def __init__(self, pretrained):
        super().__init__()
        self.encoder=copy.deepcopy(pretrained.encoder)
        self.classifier=copy.deepcopy(pretrained.classifier)
        self.po_classifier=copy.deepcopy(pretrained.classifier)
        self.num_labels=pretrained.num_labels

    def forward(self,x):
        return self.classifier(self.encoder(x))


@torch.no_grad()
def predict(model, features, device, deadline):
    model.eval()
    out=np.empty((len(features),model.num_labels),dtype=np.float16)
    for start in range(0,len(features),512):
        if time.monotonic() >= deadline: raise TimeoutError('v22 inference deadline')
        x=torch.as_tensor(np.array(features[start:start+512],dtype=np.float32),device=device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            values=model(x).float().sigmoid()
        out[start:start+len(x)]=values.cpu().numpy()
    if not np.isfinite(out).all(): raise ValueError('Invalid v22 probability')
    return out


def tensors(features,labels,indices,device):
    x=torch.as_tensor(np.array(features[indices],dtype=np.float32),device=device)
    y=labels[indices]
    y=y.toarray() if hasattr(y,'toarray') else np.array(y)
    return x,torch.as_tensor(y,dtype=torch.float32,device=device)


def step(loss, model, optimizer, scaler):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite v22 loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
    scaler.step(optimizer); scaler.update()


def po_indices(n):
    permutation=np.random.default_rng(SEED).permutation(n)
    count=min(10000,max(1,n//20))
    return permutation[count:],permutation[:count]


@torch.no_grad()
def po_probe(model,features,labels,indices,device,deadline):
    model.eval(); values=[]
    for begin in range(0,len(indices),256):
        if time.monotonic()>=deadline: raise TimeoutError('PO probe deadline')
        x,y=tensors(features,labels,indices[begin:begin+256],device)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            logits=(model.po_classifier(model.encoder(x)) if isinstance(model,RetainedPOExpert) else model(x))
        values.append((len(x),float(masked_po_loss(logits.float(),y))))
    return sum(n*v for n,v in values)/sum(n for n,v in values)


def pretrain(features,labels,weights,device,output,deadline,epochs=16,draws=300000,width=384):
    set_seed(SEED)
    model=POExpert(labels.shape[1],features.shape[1],width=width).to(device)
    train,probe=po_indices(len(features))
    p=np.asarray(weights,dtype=np.float64)[train]; p/=p.sum()
    rng=np.random.default_rng(SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[]
    for epoch in range(epochs):
        model.train(); start=time.monotonic(); losses=[]
        samples=rng.choice(train,min(draws,len(train)),replace=True,p=p)
        for begin in range(0,len(samples),256):
            if time.monotonic()>deadline-60: raise TimeoutError('PO pretraining deadline')
            x,y=tensors(features,labels,samples[begin:begin+256],device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): logits=model(x)
            loss=masked_po_loss(logits.float(),y)
            step(loss,model,optimizer,scaler); losses.append(float(loss.detach()))
        item={'stage':'v22_po_pretrain','epoch':epoch+1,'loss':float(np.mean(losses)),
              'draws':len(samples),'seconds':time.monotonic()-start}
        history.append(item); print(json.dumps(item),flush=True)
    torch.save({'state_dict':model.state_dict(),'input_dim':features.shape[1],
                'num_labels':labels.shape[1],'width':width},output/'po_pretrained.pt')
    probe_loss=po_probe(model,features,labels,probe,device,deadline)
    return model,{'history':history,'probe_loss_before_adaptation':probe_loss,
                  'probe_groups':len(probe),'probe_used_for_selection':False,'po_train_groups':len(train)}


def fit_arm(pretrained,arm,pa_features,labels,train_ix,selection_ix,base_selection,distances,
            po_features,po_labels,po_weights,device,output,deadline,epochs=48,minimum_epochs=20,patience=10):
    if arm not in ('retained_po','single_head','zero_po'): raise ValueError('Unknown v22 arm')
    set_seed(SEED)
    if arm=='zero_po':
        initial=POExpert(labels.shape[1],pa_features.shape[1],width=pretrained.width).to(device)
    else: initial=pretrained
    model=RetainedPOExpert(initial).to(device)
    teacher=copy.deepcopy(pretrained.encoder).eval() if arm=='retained_po' else None
    if teacher is not None:
        for p in teacher.parameters(): p.requires_grad_(False)
    for p in model.po_classifier.parameters(): p.requires_grad_(arm=='retained_po')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=4e-4,weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=2e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    po_train,po_test=po_indices(len(po_features))
    p=np.asarray(po_weights,dtype=np.float64)[po_train]; p/=p.sum()
    rng=np.random.default_rng(SEED); pa_rng=np.random.default_rng(SEED)
    target=np.array(labels[selection_ix]); history=[]
    before=predict(model,pa_features[selection_ix],device,deadline)
    initial_f1=float(sample_f1(target,before).mean())
    best=-1.; best_epoch=0; path=output/f'{arm}_best.pt'
    for epoch in range(1,epochs+1):
        start=time.monotonic(); model.train(); order=pa_rng.permutation(train_ix); losses=[]
        for begin in range(0,len(order),256):
            if time.monotonic()>deadline-120: raise TimeoutError('PA adaptation deadline')
            x,y=tensors(pa_features,labels,order[begin:begin+256],device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'): logits=model(x)
            pa_loss=asymmetric_loss(logits.float(),y)
            auxiliary=pa_loss.new_zeros(()); retention=pa_loss.new_zeros(())
            if arm=='retained_po':
                sample=rng.choice(po_train,min(256,len(po_train)),replace=True,p=p)
                px,py=tensors(po_features,po_labels,sample,device)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    encoded=model.encoder(px); po_logits=model.po_classifier(encoded)
                    with torch.no_grad(): anchor=teacher(px)
                auxiliary=masked_po_loss(po_logits.float(),py)
                retention=F.mse_loss(F.normalize(encoded.float(),dim=1),F.normalize(anchor.float(),dim=1))
            loss=pa_loss + .001*auxiliary + .01*retention
            step(loss,model,optimizer,scaler)
            losses.append([float(pa_loss.detach()),float(auxiliary.detach()),float(retention.detach())])
        schedule.step()
        values=predict(model,pa_features[selection_ix],device,deadline)
        standalone=float(sample_f1(target,values).mean())
        score=float(sample_f1(target,mix(base_selection,values,distances[selection_ix],CHECKPOINT_POLICY)).mean())
        if score>best:
            best,best_epoch=score,epoch
            torch.save({'state_dict':model.state_dict(),'arm':arm,'epoch':epoch,'selection_mixture_f1':score,
                        'input_dim':pa_features.shape[1],'num_labels':labels.shape[1],'width':pretrained.width},path)
        item={'arm':arm,'epoch':epoch,'selection_mixture_f1':score,'selection_standalone_f1':standalone,
              'loss_components':np.mean(losses,axis=0).tolist(),'seconds':time.monotonic()-start}
        history.append(item); print(json.dumps(item),flush=True)
        (output/f'{arm}_history.json').write_text(json.dumps(history,indent=2))
        if epoch>=minimum_epochs and epoch-best_epoch>=patience: break
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state_dict'])
    probe=po_probe(model,po_features,po_labels,po_test,device,deadline) if arm!='zero_po' else None
    diagnostics={'arm':arm,'best_epoch':best_epoch,'epochs':epoch,'selection_mixture_f1':best,
        'selection_policy':CHECKPOINT_POLICY,'selection_standalone_f1_before_adaptation':initial_f1,
        'po_probe_loss_after_adaptation':probe,'probe_used_for_selection':False,
        'po_auxiliary_weight':.001 if arm=='retained_po' else 0.,
        'encoder_retention_weight':.01 if arm=='retained_po' else 0.,'history':history}
    return model,diagnostics
