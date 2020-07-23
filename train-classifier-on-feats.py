#%%
import time
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

TRAIN_TXT = os.path.expanduser('~/data/datasets/EPIC_KITCHENS_2018/train.txt')
VAL_TXT = os.path.expanduser('~/data/datasets/EPIC_KITCHENS_2018/val.txt')
LABELS_DIR = os.path.expanduser('~/data/datasets/EPIC_KITCHENS_2018/custom_labels_30')
FRAMES_DIR = os.path.expanduser('~/data/datasets/EPIC_KITCHENS_2018/frames_30')
FEATS_DIR = os.path.expanduser('./feats')
FEATS_EXT = '_base.pt'


#%%
def get_feats_and_labels(video_name):
    f_csv_labels = os.path.join(LABELS_DIR, video_name + '.csv')

    df_labels = pd.read_csv(f_csv_labels)
    df_labels = df_labels.set_index('path', drop=False)

    all_frame_names = df_labels.path.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    feat_files = all_frame_names.apply(lambda x: os.path.join(FEATS_DIR, x + FEATS_EXT))
    feat_files_exist = feat_files.apply(os.path.exists)
    df_feats = pd.DataFrame({'frame_name': all_frame_names, 'frame_path': df_labels.path, 'feat_file': feat_files,
                             'feats_exist': feat_files_exist})
    df_feats = df_feats.set_index('frame_path')
    print(df_feats.feats_exist.sum(), '/', len(df_feats.feats_exist), 'feature files exist.')
    if df_feats.feats_exist.sum() == 0:
        print('features exist in ', FEATS_DIR, '?')

    df_label_feats = pd.concat([df_feats, df_labels], axis=1)
    assert len(df_label_feats) == len(df_feats) == len(df_labels)

    return df_label_feats

#%%
with open(TRAIN_TXT, 'r') as f:
    train_videopaths = [s.strip() for s in f.readlines()]

with open(VAL_TXT, 'r') as f:
    val_videopaths = [s.strip() for s in f.readlines()]

#%%

train_videonames = [os.path.splitext(os.path.basename(pth))[0] for pth in train_videopaths]
val_videonames = [os.path.splitext(os.path.basename(pth))[0] for pth in val_videopaths]

#%%
val_df_label_feats_list = []
for videoname in val_videonames:
    print(videoname)
    val_df_label_feats_list.append(get_feats_and_labels(videoname))


#%%
train_df_label_feats_list = []
for videoname in train_videonames:
    print(videoname)
    train_df_label_feats_list.append(get_feats_and_labels(videoname))

#%%
class FeatsDataset(Dataset):
    def __init__(self, df_feats_labels, feats_in_batch_form=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_labels_feats_files = df_feats_labels
        self.transform = transform
        self.feats_in_batch_form = feats_in_batch_form

    def __len__(self):
        return len(self.df_labels_feats_files)

    def __getitem__(self, idx):
        df_slice = self.df_labels_feats_files.iloc[idx]
        feats_file = df_slice['feat_file']
        feats = torch.load(feats_file)
        if self.feats_in_batch_form:
            assert feats.shape[0] == 1
            feats = feats[0, ...]
        label = df_slice['labels']

        sample = (feats, label)

        if self.transform:
            sample = self.transform(sample)

        return sample
#%%
df_label_feats_train = pd.concat(train_df_label_feats_list, axis=0, ignore_index=True)
df_label_feats_val = pd.concat(val_df_label_feats_list, axis=0, ignore_index=True)
train_dataset = FeatsDataset(df_label_feats_train)
val_dataset = FeatsDataset(df_label_feats_val)

#%%
# Make dataset/dataloader
train_batch_size = 16
val_batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

#%%
# Train model

class CustomSimpleNet(nn.Module):
    def __init__(self, in_channels=1024, global_pooling_type='avg'):
        super(CustomSimpleNet, self).__init__()
        conv1x1_out_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1x1_out_channels, kernel_size=1)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.global_pool_max = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=conv1x1_out_channels, out_features=2)
        self.global_pooling_type = global_pooling_type

    @property
    def global_pool(self):
        if self.global_pooling_type == 'avg':
            global_pool = self.global_pool_avg
        elif self.global_pooling_type == 'max':
            global_pool = self.global_pool_max
        else:
            raise ValueError
        return global_pool

    def forward(self, x):
        # x = F.relu(self.conv1x1(x))
        x = self.conv1x1(x)
        x = self.global_pool(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x

    def initialize_weights(self):
        self.apply(weight_reset)


def weight_reset(m):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose1d)
        or isinstance(m, nn.ConvTranspose2d)
        or isinstance(m, nn.ConvTranspose3d)
        or isinstance(m, nn.BatchNorm1d)
        or isinstance(m, nn.BatchNorm2d)
        or isinstance(m, nn.BatchNorm3d)
        or isinstance(m, nn.GroupNorm)
    ):
        m.reset_parameters()
import torch.optim as optim


class Trainer(object):
    def __init__(self, model: nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader=None, cuda=True):
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model.cuda()
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.3)
        self.loss_fcn = nn.CrossEntropyLoss()

    def train(self, n_epoch):
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.dataloader_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                loss = self.train_step(data)
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def train_step(self, data):
        x, lbl = data
        if self.cuda:
            device = torch.device("cuda")
            x, lbl = x.to(device), lbl.to(device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.model(x)
        loss = self.loss_fcn(outputs, lbl)
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, data):
        x, lbl = data
        if self.cuda:
            device = torch.device("cuda")
            x, lbl = x.to(device), lbl.to(device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.model(x)
        loss = self.loss_fcn(outputs, lbl)
        assert len(outputs.shape) == 2 and outputs.shape[1] == 2
        pred_lbl = torch.argmax(outputs, dim=1)
        return loss, pred_lbl

    def evaluate(self):
        was_training = False
        if self.model.training:
            was_training = True
            self.model.eval()
        confusion_mat = torch.zeros((2,2), dtype=int)
        with torch.no_grad():
            for epoch in range(1):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(self.dataloader_train):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    loss, pred_lbls = self.val_step(data)
                    for lbl, pred_lbl in zip(data[1], pred_lbls):
                        confusion_mat[lbl, pred_lbl] += 1
                    # print statistics
                    running_loss += loss.item()
            if was_training:
                self.model.train()
        return {'running_loss': running_loss,
                'avg_loss': running_loss / len(self.dataloader_train),
                'confusion_mat': confusion_mat}

#%%
device = torch.device("cuda")

trainer = Trainer(CustomSimpleNet(1024, 'avg'), train_dataloader)

#%%
# Confirm we can bring loss to 0 on a single image
example_sample = [x.to(device) for x in next(iter(val_dataloader))]

#%%
# Confirm we can bring loss to 0 on a single image
losses = []
for i in range(200):
    losses.append(trainer.train_step(example_sample).item())

#%%
from bokeh.plotting import figure, output_file, show, output_notebook

output_file("line.html")
p = figure(plot_width=400, plot_height=400)
# add a line renderer
p.line(range(len(losses)), losses, line_width=2)
p.xaxis.axis_label = 'iterations'
p.yaxis.axis_label = 'loss'
show(p)

#%%
trainer.model.apply(weight_reset)
val_dicts = []
start = time.time()
n_epochs = 20
for ep in range(n_epochs):
    val_dicts.append(trainer.evaluate())
    trainer.train(1)
    print(val_dicts[ep])
end = time.time()
print(f"Time elapsed over {n_epochs} epochs of length {len(trainer.dataloader_train)}:", end - start)

#%%
from bokeh.plotting import figure, output_file, show, output_notebook
val_losses = [d['avg_loss'] for d in val_dicts]
output_file("line2.html")
p = figure(plot_width=400, plot_height=400)
# add a line renderer
p.line(range(len(val_losses)), val_losses, line_width=2)
p.xaxis.axis_label = 'epochs'
p.yaxis.axis_label = 'average training loss'
show(p)

n_neg = sum(val_dicts[0]['confusion_mat'][0, :]).item()
n_pos = sum(val_dicts[0]['confusion_mat'][1, :]).item()
true_neg = [d['confusion_mat'][0, 0].item() for d in val_dicts]
true_pos = [d['confusion_mat'][1, 1].item() for d in val_dicts]
output_file("line3.html")
p = figure(plot_width=400, plot_height=400)
# add a line renderer
p.line(range(len(true_neg)), [x / n_neg for x in true_neg], line_width=2, color='blue', legend_label=f"TN (max={n_neg}")
p.line(range(len(true_pos)), [x / n_pos for x in true_pos], line_width=2, color='green', legend_label=f"TP (max={n_pos})")
p.xaxis.axis_label = 'epochs'
p.legend.location = "top_left"
show(p)

#%%
x = 2
