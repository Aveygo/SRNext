import torch
import torch.nn.functional as F

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        
        self.features = []
        self.hook()

    def hook(self):
        def hook_function(module, input, output):
            self.features.append(output)

        i = 0
        for name, module in self.model.named_modules():
            i += 1
            if isinstance(module, torch.nn.Conv2d) and i % 2 ==0:
                
                print(f"Hooked {name}")
                module.register_forward_hook(hook_function)

    def forward(self, x):
        self.features = []
        self.model(x)
        return self.features

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model, loss=torch.nn.functional.l1_loss):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = FeatureExtractor(model)
        self.loss = loss

    def forward(self, x, gt):
        # Get the features from the feature extractor
        x_features = self.feature_extractor(x)
        gt_features = self.feature_extractor(gt)

        loss = 0
        
        for i, (x, y) in enumerate(zip(x_features, gt_features)):
            #loss += self.loss(x, y)
            act_x = x.reshape(x.shape[0], x.shape[1], -1)
            act_y = y.reshape(y.shape[0], y.shape[1], -1)
            gram_x = act_x @ act_x.permute(0, 2, 1)
            gram_y = act_y @ act_y.permute(0, 2, 1)
            loss += self.loss(gram_x, gram_y)

        return loss