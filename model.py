import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import math

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        if np.random.choice(100) < 50:
            image = sample['image']
            keypoints = sample['keypoints']

            image_flip = cv2.flip(image, 1)
            center = [image.shape[1]/2, image.shape[0]/2]

            new_points = np.array([])
            for point in keypoints:
                new_x = center[0] - (point[0] - center[0])
                new_points = np.append(new_points, [new_x, point[1]])

            new_points = new_points.reshape(keypoints.shape)

            return {'image': image_flip, 'keypoints': new_points}
        else:
            return sample
class Rescale_own(object):
    
    def __init__(self, output_size):
        assert type(output_size) == int or type(output_size) == tuple
        if type(output_size) == int:
            self.output_size = (output_size, output_size)
        if type(output_size) == tuple:
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']
        image = cv2.resize(image, self.output_size)
        h, w = image.shape[:2]
        keypoints = keypoints * [self.output_size[1] / w, self.output_size[0] / h] 
        return {'image' : image, 'keypoints' : keypoints}

class Normalize(object):

    def __call__(self, sample):

        image = sample['image']
        keypoints = sample['keypoints']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image/ 255
        keypoints = (keypoints - 100) / 50
        return {'image': image, 'keypoints' : keypoints.astype(np.float64)}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}



class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, center=None):
        if type(degrees) == int:
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
    
    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    @staticmethod
    def rotate_point(points, center, degree):
        new_points = np.array([])
        for point in points:
            diff = point - center
            diff[1] = -diff[1]
            distance = np.sqrt(diff[0]**2 + diff[1]**2)
            if diff[0] == 0:
                if diff[1] >= 0:
                    old_degree = 90
                if diff[1] < 0:
                    old_degree = 270
            else:
                old_degree = np.arctan(diff[1]/diff[0])
                if diff[0] < 0:
                    old_degree = old_degree + math.pi
            new_degree = old_degree + degree/57.3
        
            new_points = np.append(new_points, [distance * np.cos(new_degree) + center[0], center[1] - distance * np.sin(new_degree) ])
        return new_points.reshape(points.shape[0], points.shape[1])

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):

        angle = self.get_params(self.degrees)
        
        image, keypoints = sample['image'], sample['keypoints']

        image_rotate = self.rotate(image, angle)
        
        center = [image.shape[1]/2, image.shape[0]/2]

        new_points = self.rotate_point(keypoints, center, angle)

        return {'image': image_rotate, 'keypoints': new_points}

class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']

        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(keypoints)}

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        self.keypoints_frame = np.array(pd.read_csv(csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.keypoints_frame[idx][0])
        image = mpimg.imread(image_name)

        if(image.shape[2] == 4):
            image = image[:,:,0:3]

        keypoints = self.keypoints_frame[idx][1:]
        keypoints = keypoints.astype('float').reshape(-1, 2)

        sample = {'image' : image, 'keypoints' : keypoints}
        if self.transform:
            sample = self.transform(sample)

        return sample

import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.4)

        self.fn1 = nn.Linear(256 * 5 * 5, 1000)
        self.fn2 = nn.Linear(1000, 1000)
        self.fn3 = nn.Linear(1000, 136)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fn1(x))
        x = self.drop2(x)
        x = F.relu(self.fn2(x))
        x = self.drop3(x)
        x = self.fn3(x)

        return x

import torch.optim as optim

if __name__ == '__main__':
    transform = transforms.Compose([Rescale((108, 108)),RandomCrop(96),RandomRotation(30), Normalize(), ToTensor()])

    training_dataset = ImageDataset('data/training_frames_keypoints.csv', 'data/training/', transform)

    training_dataloader = torch.utils.data.DataLoader(training_dataset,batch_size = 10, shuffle = True, num_workers = 0)
    print(len(training_dataset))
    test_dataset = ImageDataset('data/test_frames_keypoints.csv', 'data/test/', transform)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True, num_workers = 0)

    net = Net()

    print(net)

    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)
    
    criterion = nn.MSELoss()

    device = torch.device("cuda:0")

    net.to(device)

    def test_net():
        avg_loss = []
        for sample in test_dataloader:
            key_pts = sample['keypoints']
            image = sample['image']
            image = image.type(torch.FloatTensor)
            key_pts, image = key_pts.to(device), image.to(device) 
            predicted = net(image)
            predicted = predicted.view(predicted.size()[0], 68, 2).cpu().data.numpy()
            key_pts = key_pts.cpu().data.numpy()
    
            batch_loss = np.zeros(10)
    
            for i in range(predicted.shape[0]):
                p = predicted[i]
                k = key_pts[i]
                difference = p - k
                squared = difference ** 2
                squared = np.sum(squared, -1) ** 0.5
                squared = squared.mean()
                batch_loss = np.append(batch_loss, squared)
    
            m = batch_loss.mean()
    
            avg_loss.append(m)
            #print('Average Loss for Batch' + str(i) + ' : ' + str(m))

        avg = np.array(avg_loss).mean()
        print('Total Average Loss : ' + str(avg))

    def train_net(n_epochs):
        
        net.train()

        for epoch in range(n_epochs):
            running_loss = 0.0    

            for (batch_idx, data) in enumerate(training_dataloader):

                image = data['image']
                keypoints = data['keypoints']
                image = image.type(torch.FloatTensor)
                keypoints = keypoints.type(torch.FloatTensor)
                keypoints = keypoints.view(keypoints.size()[0], -1)
                
                image, keypoints = image.to(device), keypoints.to(device)

                output_pts = net(image)
                loss = criterion(output_pts, keypoints)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 10 == 9:
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_idx+1, running_loss/10))
                running_loss = 0.0

            with torch.no_grad():
                test_net()

        print('Finish Training')

    n_epochs = 120
    train_net(n_epochs)

    model_dir = 'saved_models/'
    model_name = 'keypoints_model_1.pt'

    torch.save(net.state_dict(), model_dir+model_name)
