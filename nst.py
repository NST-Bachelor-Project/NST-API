import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from io import BytesIO

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 256 if torch.cuda.is_available() else 128

print(f"device ------------ {device}")

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def load_image(img, maxsize=None, shape=None):
    image = Image.open(BytesIO(img))
    image = image.convert('RGB')
    if maxsize:
        loader = transforms.Compose([
            transforms.Resize(maxsize),
            transforms.ToTensor()])
    elif shape:
        loader = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def get_style_model_and_losses(cnn, mean, std, style_img, content_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = copy.deepcopy(cnn)

    normalization = NormalizationLayer(mean, std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    conv_ind = 0
    for name, layer in  cnn._modules.items():
        if isinstance(layer, nn.Conv2d):
            conv_ind += 1
            name = f'conv_{conv_ind}'
        elif isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{conv_ind}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{conv_ind}", style_loss)
            style_losses.append(style_loss)
    
    model = model[:18]

    return model, style_losses, content_losses

def run_style_transfer(cnn, mean, std,content_img, style_img, input_img):
    print('Training...')
    style_weight = 1000000
    content_weight = 1
    num_steps = 300
    model, style_losses, content_losses = get_style_model_and_losses(cnn, mean, std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    iter = 0
    while iter <= num_steps:
        def closure():
            nonlocal iter
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            iter += 1
            if iter % 50 == 0:
                print(f"Iter {iter}:")
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()


            return style_score + content_score

        optimizer.step(closure)
        
    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def NST(style_img, content_img):
    # return {"image":"success"}
    content_img = load_image(content_img, maxsize=imsize)
    style_img = load_image(style_img,shape=(content_img.size(2), content_img.size(3)))
    
    print(style_img.size(), content_img.size())
    assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"
    input_img = content_img.clone()
    image = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = image.cpu().clone()
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
        
    result = {}
    result["image"] = 'data:image/jpeg;base64,' + img_str.decode('utf-8')
    return result