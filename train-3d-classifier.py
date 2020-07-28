# %%
import argparse
import os
import shlex
import subprocess
import time

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import datetime, pytz
import yaml.representer
from tensorboardX import SummaryWriter


def get_framepaths_and_labels(video_name, labels_dir, feats_dir, feats_ext):
    f_csv_labels = os.path.join(labels_dir, video_name + '.csv')
    assert os.path.exists(f_csv_labels)

    df_labels = pd.read_csv(f_csv_labels)
    df_labels = df_labels.set_index('path', drop=False)

    all_frame_names = df_labels.path.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    feat_files = all_frame_names.apply(lambda x: os.path.join(feats_dir, x + feats_ext))
    feat_files_exist = feat_files.apply(os.path.exists)
    df_feats = pd.DataFrame({'frame_name': all_frame_names, 'frame_path': df_labels.path, 'feat_file': feat_files,
                             'feats_exist': feat_files_exist})
    df_feats = df_feats.set_index('frame_path')
    print(df_feats.feats_exist.sum(), '/', len(df_feats.feats_exist), 'feature files exist.')
    if df_feats.feats_exist.sum() == 0:
        print('Do frames exist in ', feats_dir, '?')

    df_label_feats = pd.concat([df_feats, df_labels], axis=1)
    assert len(df_label_feats) == len(df_feats) == len(df_labels)

    return df_label_feats


class ImagesDataset(Dataset):
    def __init__(self, df_frames_labels, feats_in_batch_form=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.full_df_labels_feats_files = df_feats_labels
        self.df_labels_feats_files = df_feats_labels[df_feats_labels['feats_exist'] != 0].reset_index()
        self.transform = transform
        self.feats_in_batch_form = feats_in_batch_form

    def __len__(self):
        return len(self.df_labels_feats_files)

    def __getitem__(self, idx):
        df_slice = self.df_labels_feats_files.iloc[idx]
        feats_file = df_slice['feat_file']
        if not os.path.exists(feats_file):
            print(feats_file, 'does not exist.')
            raise Exception
        feats = torch.load(feats_file)
        if self.feats_in_batch_form:
            assert feats.shape[0] == 1
            feats = feats[0, ...]
        label = df_slice['labels']

        sample = (feats, label)

        if self.transform:
            sample = self.transform(sample)

        return sample


class CustomSimpleNet(nn.Module):
    def __init__(self, in_channels=1024, global_pooling_type='avg'):
        super(CustomSimpleNet, self).__init__()
        conv1x1_out_channels = in_channels
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1x1_out_channels, kernel_size=1)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.global_pool_max = nn.AdaptiveMaxPool2d(output_size=(1, 1))
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


class Trainer(object):
    def __init__(self, model: nn.Module, dataloader_train: DataLoader, dataloader_val: DataLoader = None, cuda=True,
                 tensorboard_writer: SummaryWriter=None, start_iteration=0, lr=0.001, momentum=0.2):
        assert start_iteration == 0
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model.cuda()
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.loss_fcn = nn.CrossEntropyLoss()
        if tensorboard_writer is None:
            print(Warning('No tensorboard writer set.'))
        self.tensorboard_writer = tensorboard_writer
        self.iteration = start_iteration
        self.iteration_within_epoch = 0

    def train(self, n_epoch):
        n_itr_in_epoch = len(self.dataloader_train)
        for epoch in range(n_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.dataloader_train):
                self.iteration += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                loss = self.train_step(data)
                # print statistics
                running_loss += loss.item()
                self.tensorboard_writer.add_scalar('training_loss', loss.item(), global_step=self.iteration)
                if i % 2000 == 1999:  # print every 2000 mini-batches
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

    def evaluate(self, split='val'):
        was_training = False
        if self.model.training:
            was_training = True
            self.model.eval()
        confusion_mat = torch.zeros((2, 2), dtype=torch.int)
        dataloaders = {'val': self.dataloader_val, 'train': self.dataloader_train}
        with torch.no_grad():
            for epoch in range(1):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(dataloaders[split]):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    loss, pred_lbls = self.val_step(data)
                    for lbl, pred_lbl in zip(data[1], pred_lbls):
                        confusion_mat[lbl, pred_lbl] += 1
                    # print statistics
                    running_loss += loss.item()
            if was_training:
                self.model.train()
        self.tensorboard_writer.add_scalar(f'validation_{split}/avg_loss',  running_loss / len(dataloaders[split]),
                                           global_step=self.iteration)
        self.tensorboard_writer.add_scalar(f'validation_{split}/TN',  confusion_mat[0, 0].item(),
                                           global_step=self.iteration)
        self.tensorboard_writer.add_scalar(f'validation_{split}/TP',  confusion_mat[1, 1].item(),
                                           global_step=self.iteration)
        self.tensorboard_writer.add_scalar(f'validation_{split}/tot_neg',  confusion_mat[0, :].sum(),
                                           global_step=self.iteration)
        self.tensorboard_writer.add_scalar(f'validation_{split}/tot_pos',  confusion_mat[1, :].sum(),
                                           global_step=self.iteration)
        return {'running_loss': running_loss,
                'avg_loss': running_loss / len(dataloaders[split]),
                'confusion_mat': confusion_mat}


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    if type(hash) is not str:
        hash = hash.decode('UTF-8')
    return hash


def get_log_dir(basename, config_dictionary=None, additional_tag='', set_up_dir=True):
    basedir = os.path.dirname(basename)
    basename = os.path.basename(basename)
    name = basename
    if len(name) > 0 and name is not None:
        name += '_'

    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    name += '%s' % now.strftime('%Y-%m-%d-%H%M%S')
    name += '_VCS-%s' % git_hash()

    # load config
    if config_dictionary is not None:
        for k, v in config_dictionary.items():
            v = str(v)
            if '/' in v:
                continue
            if len(name) > 0 and name[-1] != '_':
                name += '_'
            name += '%s-%s' % (k.upper(), v)
    name += additional_tag

    # create out
    log_dir = os.path.join(basedir, name)
    if set_up_dir:
        logdir_setup(config_dictionary, log_dir)
    return log_dir


def logdir_setup(config_dictionary, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if config_dictionary is not None:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            try:
                yaml.safe_dump(config_dictionary, f, default_flow_style=False)
            except yaml.representer.RepresenterError:
                yaml.safe_dump(dict(config_dictionary), f, default_flow_style=False)
                print(Warning('converted config dictionary to non-ordereddict when saving'))


def parse_args():
    default_dataset_root = os.path.expanduser('~/data/datasets/EPIC_KITCHENS_2018')
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', action='store_true', default=False)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--dataset_root', type=str, default=default_dataset_root)
    p.add_argument('--train_txt', type=str, default='train.txt', help='relative to dataset_root')
    p.add_argument('--val_txt', type=str, default='val.txt', help='relative to dataset_root')
    p.add_argument('--labels_dir', default='custom_labels_30', help='relative to dataset_root')
    p.add_argument('--frames_dir', default='frames_30', help='relative to dataset_root')
    p.add_argument('--feats_dir', default='./feats', help='absolute path')
    p.add_argument('--feats_ext', default='_base.pt')
    p.add_argument('--global_pooling_type', default='max', choices=['avg', 'max'])
    return p.parse_args()


def main(args):
    cfg = args.__dict__
    cfg_pprint = cfg.copy()
    keys_list = list(cfg_pprint.keys())
    for k in keys_list:
        if k.endswith('txt') or k.endswith('_dir') or k.endswith('root'):
            cfg_pprint.pop(k)
        if k.endswith('ext'):
            cfg_pprint[k] = os.path.splitext(cfg_pprint[k])[0]
        if k.endswith('global_pooling_type'):
            k.replace('global_pooling_type', 'pool')
    logdir = get_log_dir(basename=os.path.join('output', 'train'), config_dictionary=cfg_pprint)
    model_save_dir = os.path.join(logdir, 'models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    with open(os.path.join(args.dataset_root, args.train_txt), 'r') as f:
        train_videopaths = [s.strip() for s in f.readlines()]

    with open(os.path.join(args.dataset_root, args.val_txt), 'r') as f:
        val_videopaths = [s.strip() for s in f.readlines()]

    train_videonames = [os.path.splitext(os.path.basename(pth))[0] for pth in train_videopaths]
    val_videonames = [os.path.splitext(os.path.basename(pth))[0] for pth in val_videopaths]

    # val
    val_df_label_feats_list = []
    for videoname in val_videonames:
        val_df_label_feats_list.append(get_feats_and_labels(videoname, os.path.join(args.dataset_root, args.labels_dir),
                                                            args.feats_dir, args.feats_ext))

    # train
    train_df_label_feats_list = []
    for videoname in train_videonames:
        train_df_label_feats_list.append(
            get_feats_and_labels(videoname, os.path.join(args.dataset_root, args.labels_dir),
                                 args.feats_dir, args.feats_ext))

    df_label_feats_train = pd.concat(train_df_label_feats_list, axis=0, ignore_index=True)
    df_label_feats_train = df_label_feats_train[df_label_feats_train.feats_exist == True].reset_index()
    df_label_feats_val = pd.concat(val_df_label_feats_list, axis=0, ignore_index=True)
    df_label_feats_val = df_label_feats_val[df_label_feats_val.feats_exist == True].reset_index()
    val_dataset = FeatsDataset(df_label_feats_val)
    train_dataset = FeatsDataset(df_label_feats_train)

    # Make dataset/dataloader
    train_batch_size = 16
    val_batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    print('len(train_dataloader)', len(train_dataloader))
    print('len(val_dataloader)', len(val_dataloader))

    # Train model
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda")

    my_model = CustomSimpleNet(1024, global_pooling_type=args.global_pooling_type)
    my_writer = SummaryWriter(logdir)
    trainer = Trainer(my_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                      tensorboard_writer=my_writer, cuda=not args.cpu)

    # Confirm we can bring loss to 0 on a single image

    itrlod = iter(val_dataloader)
    example_sample = [x for x in next(itrlod)]

    trainer.model.apply(weight_reset)
    val_dicts = []
    start = time.time()
    n_epochs = args.n_epochs
    losses_file = 'losses.txt'
    _ = trainer.evaluate('train')
    val_dict = trainer.evaluate()
    val_dicts.append(val_dict)
    with open(losses_file, 'w') as f:
        f.write(','.join(['ep'] + list(val_dict.keys())))
        f.write(','.join([str(-1)] + [as_str(val_dict[k]) for k in val_dict.keys()]))

    save_every = 1 if n_epochs < 20 else int(20 * n_epochs)
    for ep in range(n_epochs):
        # train_val_dict = trainer.evaluate('train')
        _ = trainer.evaluate('train')
        val_dict = trainer.evaluate()
        val_dicts.append(val_dict)
        with open(losses_file, 'a+') as f:
            f.write(','.join([str(ep)] + [as_str(val_dict[k]) for k in val_dict.keys()]))
        trainer.train(1)
        if ep % save_every == 0 or ep == n_epochs - 1:
            torch.save(trainer.model.state_dict(), os.path.join(model_save_dir, f'saved_model_ep_{ep}.pt'))
        print(val_dicts[ep])

    end = time.time()
    print(f"Time elapsed over {n_epochs} epochs of length {len(trainer.dataloader_train)}:", end - start)


def as_str(x):
    if torch.is_tensor(x):
        if torch.numel(x) == 1:
            return str(x.item())
        else:
            return f"{x}"
    return f"{x}"


if __name__ == '__main__':
    main(parse_args())
