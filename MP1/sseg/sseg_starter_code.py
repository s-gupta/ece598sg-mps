import glob
import os
import numpy as np
from tqdm import tqdm
from sseg_eval import segmentation_eval 
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils import data
from torch import nn, optim
from torchvision.transforms import ToTensor

DATASET_PATH = 'data/sbd/'

class SegmentationDataset(data.Dataset):
    """
    Data loader for the Segmentation Dataset. If data loading is a bottleneck,
    you may want to optimize this in for faster training. Possibilities include
    pre-loading all images and annotations into memory before training, so as
    to limit delays due to disk reads.
    """
    def __init__(self, split="train", data_dir=DATASET_PATH):
        assert(split in ["train", "val", "test"])
        self.img_dir = os.path.join(data_dir, split)
        self.classes = []
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
          for l in f:
            self.classes.append(l.rstrip())
        self.n_classes = len(self.classes)
        self.split = split
        self.data = glob.glob(self.img_dir + '/*.jpg')
        self.data = [os.path.splitext(l)[0] for l in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index] + '.jpg')
        gt = Image.open(self.data[index] + '.png')

        img = ToTensor()(img)
        gt = np.asarray(gt)
        gt = torch.from_numpy(np.array(gt)).long().unsqueeze(0)
        return img, gt

# We are providing a very simple network that does a single 1x1 convolution to
# prdict the class label. As part of the assignment you will develop a more
# expressive model to improve performance.
class SimpleNet(nn.Module):
    def __init__(self, n_classes):
        super(SimpleNet, self).__init__()
        self.mu = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).reshape(1,-1,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).reshape(1,-1,1,1)
        self.layers = nn.Sequential(
            nn.Conv2d(3, n_classes, 1, padding=0),
        )

    def to(self, device):
        self.mu = self.mu.to(device)
        self.std = self.std.to(device)
        return super(SimpleNet, self).to(device)


    def forward(self, img):
        # Normalizing the image
        x = img
        x = x - self.mu
        x = x / self.std

        x = self.layers(x)
        return x

# We have provided a basic training loop below with val and train functions.
# Feel free to modify the training loop, to suit your needs.
def val(model, imset, device, filename):
    val_dataset = SegmentationDataset(split=imset)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, 
                                     shuffle=False, num_workers=0, 
                                     drop_last=False)
    preds, gts = [], []
    
    # Put model in evaluation mode.
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    total_loss = []
    for i, batch in enumerate(val_dataloader):
        img, gt = batch
        img = img.to(device)
        gt = gt.to(device).long()
        pred = model(img)
        loss = criterion(pred, gt.squeeze(1))
        pred = torch.softmax(pred, 1)
        preds.append(pred.detach().cpu().numpy())
        gts.append(gt.detach().cpu().numpy())
        total_loss.append(loss.detach().cpu().numpy())
    gts = np.concatenate(gts, 0)
    preds = np.concatenate(preds, 0)
    aps, ious = segmentation_eval(gts, preds, val_dataset.classes, filename)
    print('Val Loss: {:5.4f}'.format(np.mean(total_loss)))
    
    # Put model back in training mode
    model.train()
    return aps, ious

def simple_train(device):
    train_dataset = SegmentationDataset(split='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, 
                                       shuffle=True, num_workers=4, 
                                       drop_last=True)
    

    # Initializing a simple model.
    model = SimpleNet(len(train_dataset.classes))

    # Moving the model to the device (eg GPU) used for training.
    model = model.to(device)

    # Setting the model in training mode.
    model.train()

    # Loss criterion that will be used to train the model
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizer that we will be using, along with the learning rate. Feel free
    # to experiment with a different learning rate, optimizer.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    iteration = 0
    
    # Number of epochs to train for. You mat need to train for much longer.
    num_epochs = 60
    for j in tqdm(range(num_epochs)):
        training_loss = []
        for i, batch in enumerate(train_dataloader):
            # Zero out gradient blobs in the optimizer
            optimizer.zero_grad()
            img, gt = batch

            # Move data to device for training
            img = img.to(device)
            gt = gt.to(device).long()

            # Get model predictions
            pred = model(img)
            
            # Compute loss against gt
            loss = criterion(pred, gt.squeeze(1))

            # Compute gradients with respect to loss
            loss.backward()

            # Take a step to update network parameters.
            optimizer.step()
            iteration += 1

            l = loss.detach().cpu().numpy()
            training_loss += [l]
        
        # Every few epochs print the loss.
        if np.mod(j+1, 20) == 0:
            print('Loss [{:8d}]: {:5.4f}'.format(iteration, np.mean(training_loss)))
        
        # Every few epochs get metrics on the validation set.
        if np.mod(j+1, 20) == 0:
            filename = os.path.join('SimpleNet', '{:06d}.pdf'.format(iteration))
            val(model, 'val', device, filename)
    
    # We are simply returning the model after 20 epocs, you may want to train
    # for longer, and possibly pick the model that leads to the best metrics on
    # the validation set.
    return model

def main():
    device = torch.device('cuda:0')
    model = simple_train(device)

    # Next, we evaluate the model. Here we are evaluating on the validation
    # set. Evaluation code produces confusion matrices, and category wise and
    # mean IoU and Average Precision. During development, you should evaluate
    # on the validation set, and identify which variant works the best.
    # Document your observations (variant tried, and performance obtained on
    # the validation set) in your PDF report. Once you are happy with the
    # performance of your model, you should test on the test set, and report
    # the final performance.
    filename = os.path.join('SimpleNet', 'final.pdf')
    aps, ious = val(model, 'val', device, filename)

if __name__ == '__main__':
    main()
