import torch
from torch import nn

#model based on paper 6 and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()


        self.conv1 = nn.Conv2d(num_classes, 64, (4*4), 2, bias=False)
        self.conv2 = nn.Conv2d(64, 128, (4*4), 2, bias=False)
        self.conv3 = nn.Conv2d(128, 256, (4 * 4), 2, bias=False)
        self.conv4 = nn.Conv2d(256, 512, (4*4), 2, bias=False)
        self.conv5 = nn.Conv2d(512, 1, (4*4), 2, bias=False)
        self.up = nn.Upsample(scale_factor=64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 64)
        self.drop = nn.Dropout(p=0.25)


    def forward(self, input):
        #print("input =   " + str(input.shape))
        x = self.relu(self.conv1(input))
        #print("1 =   " + str(x.shape))
        x = self.relu(self.conv2(x))
        #print("2 =   " + str(x.shape))
        x = self.relu(self.conv3(x))
        #print("3 =   " + str(x.shape))
        x = self.relu(self.conv4(x))
        #print("4 =   " + str(x.shape))
        x = self.conv5(x)
        #print("5 =   " + str(x.shape))
        x = self.flatten(x)
        #print("flatten =   " + str(x.shape))
        x = self.drop(self.fc(x)) #only here
        #print("fully =   " + str(x.shape))
        #x = torch.reshape(x,(4,1,4,16))
        #x = self.up(x)
        #print("up =   " + str(x.shape))
        return x
        #return self.main(input)