import math
from flask import Flask, json
import io
import os

import boto3
import PIL.Image
import torch
import torch.nn as nn
import torchvision


import numpy as np

app = Flask(__name__)

#My S3 bucket
bucket = os.environ['S3_BUCKET']

#AWS S3 client
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

"""
    Transformations for our input image:

    Resize and Centercrop gives us a square image that matches the trained resolution of our ResNet50 model
    ToTensor takes the RGB values (0, 255) and normalizes them from (0, 1)
    Normalize performs normalization operations that match our original imageNet pre-trained model
"""

valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=305, interpolation=PIL.Image.ANTIALIAS),
    torchvision.transforms.CenterCrop(299),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

#Averaging and MaxPooling layer that adapts to various resolution/architecture
#size
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

#Flattens an inpput tensor into a [1, X] length tensor
class Flatten(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x): return x.view(x.size(0), -1)


#From torchvision/models/resnet.py - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

"""
    From torchvision/models/resnet.py - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Based on the pytorch ResNet50 model implementation but with added layers to support a
    variable number of classes

"""
class ResNet50(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.num_classes = num_classes
        super(ResNet50, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax(1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


"""
    Initializes our Pytorch model and loads our pre-trained weights from S3
"""
class SetupModel(object):
    net = ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=7)

    def __init__(self, f):
        self.f = f
        print("Loading Model...")
        s3.meta.client.download_file('coffeelambda', 'model/coffee_299_res50.h5', '/tmp/coffee_299_res50.h5')
        self.net.model.load_state_dict(
            torch.load('/tmp/coffee_299_res50.h5',
                       map_location=lambda storage, loc: storage))
        print("Successfully loaded Model!")
        self.net.eval()

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

"""
    Takes in the image upload to S3 from our app and applies the proper transforms
    before running it through our model
"""
@SetupModel
def predict(r):
    input_batch = []
    with PIL.Image.open(io.BytesIO(r)) as im:
        im.convert("RGB")
        input_batch.append(valid_transform(im))

    input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
    return SetupModel.net(input_batch_var)

"""
    Debug use only
"""
@app.route('/test')
def test():
    return str('I\'m Alive!')


"""
    Primary handler function.
    Args:
        img_req: Path to valid S3 file by Key
    Returns:
        result: Json encoded dictionary containing our actual Prediction,
        an array of the classes in our model, the index of actual Prediction
        within the previous array, and our raw model output run through a
        Softmax activation
"""
@app.route('/app/<path:img_req>')
def lambda_handler(img_req):

    print(img_req)
    img_req = str(img_req)
    output = predict(s3_client.get_object(Bucket=bucket, Key=img_req)['Body'].read())
    classes = s3_client.get_object(Bucket=bucket, Key='model/classes.txt')['Body'].read()
    classes = str(classes[:-1], 'utf-8').split(' ')
    preds = np.around(output.data.numpy(), decimals=2)

    pred_idx = int(np.argmax(preds))
    np.set_printoptions(precision=2)
    preds = str(preds)

    result = dict(preds=preds, prediction=classes[pred_idx], classes=classes, pred_idx=pred_idx)
    result = json.dumps(result)

    return result, 200

if __name__ == '__main__':
    app.run()
