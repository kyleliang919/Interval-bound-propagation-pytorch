import torch
import torchvision
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

def get_mnist(batch_size = 100):
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False) 
    return trainloader,testloader

def get_cifar(batch_size = 200):
    transform = transforms.Compose(
        [
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),  
         transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader
     
        

def print_accuracy(net, trainloader, testloader, device, test=True, eps = 0):
    loader = 0
    loadertype = ''
    if test:
        loader = testloader
        loadertype = 'test'
    else:
        loader = trainloader
        loadertype = 'train'
    correct = 0
    total = 0
    with torch.no_grad():
        for ii, data in enumerate(loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            x_ub = images + eps
            x_lb = images - eps
            
            outputs = net(torch.cat([x_ub,x_lb], 0))
            z_hb = outputs[:outputs.shape[0]//2]
            z_lb = outputs[outputs.shape[0]//2:]
            lb_mask = torch.eye(10).cuda()[labels]
            hb_mask = 1 - lb_mask
            outputs = z_lb * lb_mask + z_hb * hb_mask
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    correct = correct / total
    print('Accuracy of the network on the', total, loadertype, 'images: ',correct)
    return correct


def verify(net, image, label, device, eps):
    x_ub = image + eps
    x_lb = image - eps
    outputs = net.forward_g(torch.cat([x_ub, x_lb], 0))
    v_u = outputs[:outputs.shape[0]//2]
    v_l = outputs[outputs.shape[0]//2:]
    weight = net.score_function.weight
    bias = net.score_function.bias
    label = label.item()
    for i in range(weight.shape[0]):
        new_w = weight[i:i+1] - weight[label:label+1]
        u = (v_u + v_l)/2
        r = (v_u - v_l)/2
        if (torch.dot(new_w[0],u[0]) + torch.dot(torch.abs(new_w[0]),r[0]) + bias[i] - bias[label]).item() > 0:
            return False
    return True

def verify_robustness(net, dataloader, device, eps = 0.1):
    net.train()
    total = 0
    correct = 0
    for ii, data in enumerate(dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        for idx in range(labels.size(0)):
            image = images[idx:idx + 1]
            label = labels[idx:idx + 1]
            if verify(net, image, label, device, eps):
                correct+=1
    correct = correct / total
    return correct

from torch.autograd import Variable
def pgd(net, image, label, device, step = 0.01, iterations = 40, eps = 0.1):
    criterion = nn.CrossEntropyLoss()
    inputs = image
    for _ in range(iterations):
        net.zero_grad()
        inputs = Variable(inputs,requires_grad = True)
        outputs = net(torch.cat([inputs,inputs], 0))
        z_hb = outputs[:outputs.shape[0]//2]
        z_lb = outputs[outputs.shape[0]//2:]
        lb_mask = torch.eye(10).cuda()[label]
        hb_mask = 1 - lb_mask
        outputs = z_lb * lb_mask + z_hb * hb_mask
        loss =  criterion(outputs, label)
        loss.backward()
        inputs = torch.clamp(inputs + step * inputs.grad - image, -eps, eps) + image
    return torch.clamp(inputs.data, 0, 1)

def verify_robustness_pgd(net, dataloader, device, eps = 0.1, iterations = 10):
    net.train()
    total = 0
    correct = 0
    for ii, data in enumerate(dataloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        for idx in range(labels.size(0)):
            image = images[idx:idx + 1]
            label = labels[idx:idx + 1]
            image = pgd(net, image, label, device, step = 0.01, iterations = iterations, eps = eps)
            if verify(net, image, label, device, eps = 0):
                correct+=1
        print(correct/total)
    correct = correct / total
    return correct