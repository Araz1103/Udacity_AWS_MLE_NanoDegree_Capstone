# Importing Required Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
# We were facing Truncated Images error, so to avoid that using this

ImageFile.LOAD_TRUNCATED_IMAGES = True

#For Logging
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Test Function 

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss=0
    running_corrects=0
    total_data_len = 0
    pred = []
    label = []
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device) #FOR GPU
        labels=labels.to(device) #FOR GPU  
        outputs=model(inputs)
        logger.info(f"Outputs: {outputs}")
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        logger.info(f"Prediction is:{preds}")
        logger.info(f"Label is: {labels.data}")
        new_pred = preds.tolist()
        new_label= labels.data.tolist()
        logger.info(f"Prediction List:{new_pred}")
        logger.info(f"Label List: {new_label}")
        pred.extend(new_pred)
        label.extend(new_label)
        logger.info(f"Final Prediction List Updated:{pred}")
        logger.info(f"Final Label List Updated: {label}")
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_data_len+= len(labels.data)
        logger.info(f"Correct are: {torch.sum(preds == labels.data)}")
        logger.info(f"Running Corrects are: {running_corrects}")

    metrics = {0:{"tp":0, "fp":0}, 1:{"tp":0, "fp":0}, 2:{"tp":0, "fp":0}, 3:{"tp":0, "fp":0}, 4:{"tp":0, "fp":0}}
    label_count = {0:0, 1:0, 2:0, 3:0, 4:0}
    for l, p in zip(label, pred):
        label_count[l]+=1
        if(p==l):
            metrics[l]["tp"]+=1
        else:
            metrics[p]["fp"]+=1
            
    logger.info(f"Metrics Computed: {metrics}")
    logger.info(f"Label Count Computed: {label_count}")
    Precision = {0:0, 1:0, 2:0, 3:0, 4:0}
    Recall = {0:0, 1:0, 2:0, 3:0, 4:0}
    F1 = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for c in Precision:
        denom = metrics[c]["tp"] + metrics[c]["fp"]
        if(denom==0):
            Precision[c]==0
        else:
            num = metrics[c]["tp"]
            Precision[c] = num/denom
    
    
    for c in Recall:
        denom = label_count[c]
        if(denom==0):
            Recall[c]==0
        else:
            num = metrics[c]["tp"]
            Recall[c] = num/denom
            
    for c in F1:
        if(Precision[c]==0 and Recall[c]==0):
            F1[c]=0
        else:
            num = 2*Precision[c]*Recall[c]
            denom = Precision[c] + Recall[c]
            F1[c] = num/denom
            
    
    logger.info(f"Precision Computed: {Precision}")
    logger.info(f"Recall Computed: {Recall}")
    logger.info(f"F1 Computed: {F1}")
    
    
    logger.info(f"Test Len: {total_data_len}")
    logger.info(f"{running_corrects.double()}")
    total_loss = running_loss / len(test_loader)
    new_acc = float(running_corrects)/float(total_data_len)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {new_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    
    epochs=5 
    image_dataset={'train':train_loader, 'valid':validation_loader}
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        total_data_len = 0
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
               
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device) #FOR GPU
                labels=labels.to(device) #FOR GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_data_len+= len(labels.data)

            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = float(running_corrects) / float(total_data_len)
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if(epoch==(epochs-1) and phase=="valid"): #Last epoch's validation is objective metric
                logger.info('Final Validation Loss: {:.4f}, acc: {:.4f}'.format(epoch_loss,epoch_acc))
        
    return model
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #num channels, num of kernels, kernel size
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2) #Kernel Size, stride
        self.fc1 = nn.Linear(14*14*256, 2048) #Fully Connected takes flattened o/ps of Conv
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 5)
        # TODO: Define the layers you need in your model

    def forward(self, x):
        #TODO: Define your model execution
        x = F.relu(self.conv1(x))  
        x = self.pool(F.relu(self.conv2(x))) # Conv -> Relu -> Pool
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x))) 
        x = self.pool(F.relu(self.conv5(x)))
        # print(x.shape) to see final conv shape
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train_data')
    test_data_path = os.path.join(data, 'test_data')
    validation_data_path=os.path.join(data, 'valid_data')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')

    # FOR GPU 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    model=Model()
    model=model.to(device) #FOR GPU
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer, device)
    
    logger.info("Testing Model")
    test(model, test_loader, criterion, device)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    #Parsing Arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
