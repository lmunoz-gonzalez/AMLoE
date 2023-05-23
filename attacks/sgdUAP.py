import torch
import torch.nn as nn


'''
Code borrowed and adapted from Kenneth Co's Github: 
    https://github.com/kenny-co/sgd-uap-torch
    
Basic version of untargeted stochastic gradient descent UAP adapted from:
[AAAI 2020] Universal Adversarial Training
- https://ojs.aaai.org//index.php/AAAI/article/view/6017
Layer maximization attack from:
Universal Adversarial Perturbations to Understand Robustness of Texture vs. Shape-biased Training
- https://arxiv.org/abs/1911.10364

Parameters:
    + model       model
    + loader      dataloader
    + nb_epoch    number of optimization epochs
    + eps         maximum perturbation value (L-infinity) norm
    + device      CPU/GPU
    + starts      number of random starts for crafting the UAP
    + beta        clamping value
    + y_target    target class label for Targeted UAP variation
    + loss_fn     custom loss function (default is CrossEntropyLoss)
    + layer_name  target layer name for layer maximization attack
    + uap_init    custom perturbation to start from (default is random 
                         vector with pixel values {-eps, eps})

Output:
    + best_delta  adversarial perturbation
'''
def sgdUAP(model, loader, nb_epoch, eps, device, starts=1, beta = 12,
            y_target = None, loss_fn = None, layer_name = None, 
            uap_init = None):

    _, (x_val, y_val) = next(enumerate(loader))
    batch_size = len(x_val)
    
    for st in range(starts):
        eps_decay = eps 
        print("INITIALIZATION NUMBER: ", st+1)
        
        #Initialize UAP or use existing initialization
        if uap_init is None:
            batch_delta = torch.zeros_like(x_val) # initialize as zero vector
            delta = batch_delta[0].to(device)
            #delta = torch.empty_like(x_val[0]).uniform_(-eps*(x_val.max() - x_val.min())
            #                                               , eps*(x_val.max() - x_val.min())).to(device)
            
        else:
            batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
            delta = batch_delta[0].to(device)
        
        # loss function
        if layer_name is None:
            if loss_fn is None: 
                loss_fn = nn.CrossEntropyLoss(reduction = 'none')
            beta = torch.FloatTensor([beta]).to(device)
            #Function to limit the effect of possible dominating examples 
            #(see paper "Universal Adversarial Training" for a detail description)
            def clamped_loss(output, target):
                loss = torch.mean(torch.min(loss_fn(output, target), beta))
                return loss
           
        # layer maximization attack
        else:
            def get_norm(self, forward_input, forward_output):
                global main_value
                main_value = torch.norm(forward_output, p = 'fro')
            for name, layer in model.named_modules():
                if name == layer_name:
                    handle = layer.register_forward_hook(get_norm)
       
        best_loss = float('-inf')
        best_delta = torch.zeros_like(delta).to(device)
    
    
        for epoch in range(nb_epoch):
            #print('EPOCH %i/%i' % (epoch + 1, nb_epoch))
            losses = []
            # perturbation step size with decay
            #eps_step = eps * step_decay
            eps_decay = eps_decay*0.8
            
            for i, (x_val, y_val) in enumerate(loader):
                
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                            
                batch_delta = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])
                batch_delta.requires_grad_()
                            
                # for targeted UAP, switch output labels to y_target
                if y_target is not None: 
                    y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype)*y_target
                
                perturbed = torch.clamp((x_val + batch_delta).to(device), 0, 1)
                #perturbed.requires_grad = True
                outputs = model(perturbed)
                
                # loss function value
                if layer_name is None: 
                    loss = clamped_loss(outputs, y_val.to(device))
                else: 
                    loss = main_value
                
                if y_target is not None: 
                    loss = -loss # minimize loss for targeted UAP
                
                
                # Compute gradient
                grad = torch.autograd.grad(loss, batch_delta,
                                           retain_graph=False,
                                           create_graph=False)[0]
                # batch update
                grad_sign = grad.data.mean(dim = 0).sign().to(device)            
                #eps_step = eps*(x_val.max() - x_val.min()) * step_decay
                #delta = delta + grad_sign * eps_step
                delta = delta + grad_sign * eps_decay
                delta = torch.clamp(delta, -eps*(x_val.max() - x_val.min()), 
                                                 eps*(x_val.max() - x_val.min()))
                #batch_delta.grad.data.zero_()
                losses.append(loss.item())
            loss_epoch = sum(losses)/len(losses)
            #print(f"Loss: {loss_epoch:.4f}")
        if (loss_epoch > best_loss):
            best_delta = delta.data
            best_loss = loss_epoch
            
    if layer_name is not None: handle.remove() # release hook
        
    return best_delta



''' 
Function to craft a UAP attack and test it against the same set of points used
to craft the UAP attack. 

Parameters:
    + model       model
    + device      CPU/GPU
    + test_loader dataloader
    + epochs      number of optimization epochs
    + epsilon     maximum perturbation value (L-infinity) norm 
    + starts      number of restarts for crafting the UAP
    + beta        clamping value
    + y_target    target class label for Targeted UAP variation
    + loss_fn     custom loss function (default is CrossEntropyLoss)
    + layer_name  target layer name for layer maximization attack
    + uap_init    custom perturbation to start from (default is random 
                         vector with pixel values {-eps, eps})
    
Output: 
    + accuracy   robust accuracy against the UAP attack on the data in test_loader

'''
def test(model, device, test_loader, epochs, epsilon, starts=1, beta = 12, 
         y_target = None, loss_fn = None, layer_name = None, uap_init = None):

    # Accuracy counter
    correct = 0
    samples_target_class = 0
    adv_examples = []

    perturbation = sgdUAP(model, test_loader, epochs, epsilon, device, 
                             starts, beta, y_target, loss_fn, 
                             layer_name, uap_init = None)
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        x_adv = x + perturbation.unsqueeze(0).repeat([x.shape[0], 1, 1, 1])
        x_adv.to(device)
        
        # Classify adversarial examples
        output = model(x_adv)

        # Compute adversarial accuracy
        _, predicted_labels = torch.max(output, 1)
        if (y_target == None):
            correct += (predicted_labels == y).sum()
        elif (y_target is not None):
            correct += (predicted_labels == y_target).sum() 
            samples_target_class += (y == y_target).sum()
        else:
            correct = 0.0

        # Append adversarial examples for visualization
        adv_examples.append(x_adv.squeeze().detach().cpu().numpy())
    # Accuracy of the model
    numerator = float(correct - samples_target_class)
    denominator = float(len(test_loader.dataset) - samples_target_class)
    accuracy =  numerator / denominator
    return accuracy, adv_examples, perturbation



''' 
Function to craft a UAP attack and test it against the same set of points used
to craft the UAP attack. 

Parameters:
    + S_model     source model used to craft the UAP
    + models      set of target models
    + device      CPU/GPU
    + test_loader dataloader
    + epochs      number of optimization epochs
    + epsilon     maximum perturbation value (L-infinity) norm 
    + starts      number of restarts for crafting the UAP
    + beta        clamping value
    + y_target    target class label for Targeted UAP variation
    + loss_fn     custom loss function (default is CrossEntropyLoss)
    + layer_name  target layer name for layer maximization attack
    + uap_init    custom perturbation to start from (default is random 
                         vector with pixel values {-eps, eps})
    
Output: 
    + accuracy   robust accuracies against the UAP transfer attack for the
                 different models (including the source model in the 1st position)
'''

def testTransferAttack(S_model, models, device, test_loader, epochs, 
                       epsilon, starts=1, beta = 12, y_target = None, 
                       loss_fn = None, layer_name = None, uap_init = None):
   
    #Number of models
    nmodels = len(models) + 1
    
    # Accuracy counter
    correct = [0] * nmodels
    samples_target_class = [0] * nmodels
    accuracy = [0] * nmodels

    perturbation = sgdUAP(S_model, test_loader, epochs, epsilon, device, 
                             starts, beta, y_target, loss_fn, 
                             layer_name, uap_init = None)
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        x_adv = x + perturbation.unsqueeze(0).repeat([x.shape[0], 1, 1, 1])
        x_adv.to(device)
        
        # Classify adversarial examples
        output = S_model(x_adv)

        i = 0
        # Classify adversarial examples on the source model
        output = S_model(x_adv)
        # Compute adversarial accuracy
        _, predicted_labels = torch.max(output, 1)
        correct[i] += (predicted_labels == y).sum()

        i += 1
        # Classify adversarial examples on the target models
        for model in models:
            output = model(x_adv)
            # Compute adversarial accuracy
            _, predicted_labels = torch.max(output, 1)
            correct[i] += (predicted_labels == y).sum()
            i += 1

    for i in range(nmodels):
        # Accuracy of the model
        numerator = float(correct[i] - samples_target_class[i])
        denominator = float(len(test_loader.dataset)- samples_target_class[i])
        accuracy[i] =  numerator / denominator
    
    return accuracy

