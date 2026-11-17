"""Minimal inference-only adapters matching the published SAGE networks."""
import ast
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.layers=nn.Sequential(nn.Linear(width,width),nn.ReLU())
    def forward(self,x):return self.layers(x)


class SimpleMLP(nn.Module):
    def __init__(self,num_classes,hidden_dim=256,num_blocks=4,use_location=False,input_dim=52):
        super().__init__()
        if use_location:raise ValueError('Only the admitted environment-only network is implemented')
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),*[MLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.classifier=nn.Linear(hidden_dim,num_classes)
    def forward(self,x):return self.classifier(self.net(x))


class ResidualBlock(nn.Module):
    def __init__(self,width,inner,dropout1,dropout2):
        super().__init__();self.norm=nn.BatchNorm1d(width);self.fc1=nn.Linear(width,inner)
        self.act=nn.ReLU();self.drop1=nn.Dropout(dropout1);self.fc2=nn.Linear(inner,width);self.drop2=nn.Dropout(dropout2)
    def forward(self,x):return x+self.drop2(self.fc2(self.drop1(self.act(self.fc1(self.norm(x))))))


class ResNet(nn.Module):
    def __init__(self,num_classes,hidden_dim=256,num_blocks=4,d_hidden_multiplier=2.,
                 use_location=False,dropout1=0.,dropout2=0.,input_dim=52):
        super().__init__()
        if use_location:raise ValueError('Only the admitted environment-only network is implemented')
        self.projection=nn.Linear(input_dim,hidden_dim)
        self.blocks=nn.Sequential(*[ResidualBlock(hidden_dim,int(hidden_dim*d_hidden_multiplier),dropout1,dropout2) for _ in range(num_blocks)])
        self.head_norm=nn.BatchNorm1d(hidden_dim);self.head_act=nn.ReLU();self.classifier=nn.Linear(hidden_dim,num_classes)
    def forward(self,x):return self.classifier(self.head_act(self.head_norm(self.blocks(self.projection(x)))))


def normalization_from_source(path,predictors_subset):
    """Read literal constants from publisher code without importing/executing it."""
    tree=ast.parse(Path(path).read_text());statistics={}
    for node in tree.body:
        if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Dict):continue
        names=[target.id for target in node.targets if isinstance(target,ast.Name)]
        for name in names:
            if name not in ['predictors_means','predictors_stds']:continue
            values={}
            for key,value in zip(node.value.keys,node.value.values):
                group=ast.literal_eval(key)
                if group in predictors_subset and group!='location':values[group]=ast.literal_eval(value)
            statistics[name]=values
    arrays=[]
    for name in ['predictors_means','predictors_stds']:
        flat=[statistics[name][group][i] for group,indices in predictors_subset.items() if group!='location' for i in indices]
        arrays.append(np.array(flat,dtype=np.float32))
    if len(arrays[0])!=52 or np.any(arrays[1]<=0):raise ValueError('Published normalization contract mismatch')
    return arrays


def normalize(features,means,stds):
    x=torch.as_tensor(features,dtype=torch.float32)
    mean=torch.as_tensor(means,dtype=torch.float32,device=x.device)
    std=torch.as_tensor(stds,dtype=torch.float32,device=x.device)
    if x.ndim!=2 or x.shape[1]!=len(mean):raise ValueError('Feature count mismatch')
    result=(torch.where(torch.isnan(x),mean,x)-mean)/std
    if not torch.isfinite(result).all():raise ValueError('Nonfinite predictor remains after published normalization')
    return result


def load_network(checkpoint,model_config):
    if not zipfile.is_zipfile(checkpoint):raise ValueError('ZIP-serialized weights-only checkpoint required')
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=loaded.get('state_dict',loaded)
    weights={k.removeprefix('net.'):v for k,v in state.items() if k.startswith('net.')}
    net=model_config['model']['init_args']['net'];kind=net['class_path'].split('.')[-1]
    if kind not in ['SimpleMLP','ResNet']:raise ValueError('Unadmitted model class')
    model=(SimpleMLP if kind=='SimpleMLP' else ResNet)(**net['init_args'])
    model.load_state_dict(weights,strict=True);model.eval()
    return model
