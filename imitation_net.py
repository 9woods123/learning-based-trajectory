import torch
import torch.nn as nn

class ImageModule(nn.Module):




    def __init__(self):
        super(ImageModule, self).__init__()
        self.image_size=128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.relu8 = nn.ReLU()
        
        self.fc1 = nn.Linear(256, 256)
        self.relu_fc1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 256)
        self.relu_fc2 = nn.ReLU()
        
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.max_pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = self.max_pool4(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.max_pool4(x)

        x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = self.relu_fc1(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)

        return x


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.5):
        super(FullyConnectedModule, self).__init__()
        
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        return x


class imitationModel(nn.Module):
    def __init__(self):
        super(imitationModel, self).__init__()
        
        self.image_module = ImageModule()
        self.fc_module1 = FullyConnectedModule(512, 1024, dropout=0)
        self.fc_module2 = FullyConnectedModule(1024, 512, dropout=0)
        self.fc_module3 = FullyConnectedModule(512, 40,dropout=0)

        self.fc_module_goal_1 = FullyConnectedModule(4, 256,dropout=0)
        self.fc_module_goal_2 = FullyConnectedModule(256, 256,dropout=0)

        
    def forward(self, img, goal, current_pose):
        
        img_tensor = self.image_module(img)
        
        goal_and_pose = torch.cat((goal, current_pose), dim=1)

        goal_and_pose=self.fc_module_goal_1(goal_and_pose)
        goal_and_pose=self.fc_module_goal_2(goal_and_pose)


        x = torch.cat((img_tensor, goal_and_pose), dim=1)

        x = x.to(torch.float)
        x = self.fc_module1(x)
        x = self.fc_module2(x)
        x= self.fc_module3(x)

        return x

