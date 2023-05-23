import torch
import torch.nn as nn


'''
BASIC ITERATIVE METHOD (BIM) ATTACK

Paper: “Adversarial examples in the physical world”. Kurakin et al. 2017: 
    https://arxiv.org/abs/1607.02533

Code Adapted from GitHub user Harry24k: 
    https://github.com/Harry24k/adversarial-attacks-pytorch 
'''


'''
Function to craft the BIM attack on a set of labeled data points (x, y)

Parameters: 
    + x:        inputs
    + y:        labels
    + model:    model used to craft the adversarial examples
    + epsilon:  maximumm value of the perturbation according to some l_p norm
                (specificed below)
    + alpha:    learning rate for crafting the adversarial examples
    + steps:    number of steps in the optimization of the adv. examples
    + device:   CPU/GPU
    + attack_type: targeted / untargeted
    + lp_norm:  l_p norm used to constrain the adversarial perturbations
    
Outputs:
    + perturbed_x: resulting adversarial perturbations
    
Note: for targeted attacks, y should define the set of target labels for each
    data point x. 
    
'''

def bim_attack(x, y, model, epsilon, alpha, steps, device, 
               attack_type = 'untargeted', lp_norm = 'linf'):
    
    tol = 1e-10
    #Define loss
    loss = nn.CrossEntropyLoss()
    
    #Rescale epsilon (to account for data normalization)
    epsilon = epsilon*(x.max() - x.min())
    
    perturbed_x = x.clone().detach()

    for st in range(steps):
        perturbed_x.requires_grad = True
        # Forward pass the data through the model
        output = model(perturbed_x)
        
        # Compute loss
        cost = loss(output, y)
        
        #Compute loss depending on the attack type
        if (attack_type == 'untargeted'):
            cost = loss(output, y)
        elif (attack_type == 'targeted'):
            cost = -loss(output, y)
        else:
            cost = loss(output, y)
        
        
        # Compute gradient
        grad = torch.autograd.grad(cost, perturbed_x,
                                   retain_graph=False,
                                   create_graph=False)[0]

        #Update adversarial example
        perturbed_x = perturbed_x.detach() + alpha*grad.sign()
        delta = torch.clamp(perturbed_x - x, min= -epsilon, max = epsilon)
        if (lp_norm == 'linf'):
            delta = torch.clamp(perturbed_x - x, min= -epsilon, max = epsilon)
            perturbed_x = torch.clamp(x + delta, min=x.min(), max=x.max()).detach()
        elif (lp_norm == 'l2'):
            delta = perturbed_x - x  
            delta_norm = torch.norm(delta.view(x.size(0), -1), p=2, dim=1) + tol
            factor = epsilon / delta_norm
            factor = torch.min(factor, torch.ones_like(delta_norm))
            delta = delta * factor.view(-1, 1, 1, 1)
            perturbed_x = torch.clamp(x + delta, min=x.min(), max=x.max()).detach()
        else:
            print("Non supported norm. Applying linf instead")
            delta = torch.clamp(perturbed_x - x, min= -epsilon, max = epsilon)
            perturbed_x = torch.clamp(x + delta, min=x.min(), max=x.max()).detach()
        
    return perturbed_x
    


'''
Function to test the effectiveness of a BIM attack on a set of test data points

Parameters: 
    + model:    model used to craft the adversarial examples
    + device:   CPU/GPU
    + test_loader: data loader with the set of test points
    + epsilon:  maximumm value of the perturbation according to some l_p norm
                (specificed below)
    + alpha:    learning rate for crafting the adversarial examples
    + steps:    number of steps in the optimization of the adv. examples
    + attack_type: targeted / untargeted
    + lp_norm:  l_p norm used to constrain the adversarial perturbations
    + target_class: for targeted attacks, the target label for the adv. examples
    
Outputs:
    + accuracy: robust accuracy of the model evaluated on the resulting adv. 
                examples.
    + x_adv: resulting adversarial perturbations
        
'''

def test(model, device, test_loader, epsilon, alpha, steps = 5,
         attack_type = 'untargeted', lp_norm = 'linf', target_class = 0):

    # Accuracy counter
    correct = 0
    samples_target_class = 0
    adv_examples = []


    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        x.requires_grad = True
        model.zero_grad()
        
        
        if (attack_type == 'untargeted'):
            x_adv = bim_attack(x, y, model, epsilon, alpha, steps, device, 
                               attack_type, lp_norm)
        elif (attack_type == 'targeted'):
            y2 = y.clone().detach().to(device)
            y2[:] = target_class
            x_adv = bim_attack(x, y2, model, epsilon, alpha, steps, device, 
                               attack_type, lp_norm)
        else:
            x_adv = bim_attack(x, y, model, epsilon, alpha, steps, device, 
                               attack_type, lp_norm)
        
        # Classify adversarial examples
        output = model(x_adv)
        
        # Compute adversarial accuracy
        _, predicted_labels = torch.max(output, 1)
        if (attack_type == 'untargeted'):
            correct += (predicted_labels == y).sum()
        elif (attack_type == 'targeted'):
            correct += (predicted_labels == target_class).sum() 
            samples_target_class += (y == target_class).sum()
        else:
            correct = 0.0
            
        del output
        del predicted_labels
        torch.cuda.empty_cache()

        # Append adversarial examples for visualization
        adv_examples.append(x_adv.squeeze().detach().cpu().numpy())
    # Accuracy of the model
    numerator = float(correct - samples_target_class)
    denominator = float(len(test_loader.dataset)- samples_target_class)
    accuracy =  numerator / denominator
    return accuracy, adv_examples


'''
Function to test the effectiveness BIM transfer attacks

Parameters: 
    + S_model:  source model used to craft the adv. examples
    + model:    set of target models
    + device:   CPU/GPU
    + test_loader: data loader with the set of test points
    + epsilon:  maximumm value of the perturbation according to some l_p norm
                (specificed below)
    + alpha:    learning rate for crafting the adversarial examples
    + steps:    number of steps in the optimization of the adv. examples
    + attack_type: targeted / untargeted
    + lp_norm:  l_p norm used to constrain the adversarial perturbations
    + target_class: for targeted attacks, the target label for the adv. examples
    
Outputs:
    + accuracy: robust accuracies for the source and target models (the accuracy
                of the source model is given in the first position)       
'''

def testTransferAttack(S_model, models, device, test_loader, epsilon, alpha, steps = 5, 
         attack_type = 'untargeted', lp_norm = 'linf', target_class = 0):
   
    #Number of models
    nmodels = len(models) + 1
    
    # Accuracy counter
    correct = [0] * nmodels
    samples_target_class = [0] * nmodels
    accuracy = [0] * nmodels

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        
        x.requires_grad = True
        S_model.zero_grad()
        if (attack_type == 'untargeted'):
            x_adv = bim_attack(x, y, S_model, epsilon, alpha, 
                               steps, device, attack_type, lp_norm)
        elif (attack_type == 'targeted'):
            y2 = y.clone().detach().to(device)
            y2[:] = target_class
            x_adv = bim_attack(x, y2, S_model, epsilon, alpha, 
                               steps, device, attack_type, lp_norm)
        else:
             x_adv = bim_attack(x, y, S_model, epsilon, alpha, 
                               steps, device, attack_type, lp_norm)
        

        i = 0
        # Classify adversarial examples on the source model
        output = S_model(x_adv)
        # Compute adversarial accuracy
        _, predicted_labels = torch.max(output, 1)
        if (attack_type == 'untargeted'):
            correct[i] += (predicted_labels == y).sum()
        elif (attack_type == 'targeted'):
            correct[i] += (predicted_labels == target_class).sum() 
            samples_target_class[i] += (y == target_class).sum()
        else:
            correct[i] = 0.0
            
        i += 1
        # Classify adversarial examples on the target models
        for model in models:
            output = model(x_adv)
            # Compute adversarial accuracy
            _, predicted_labels = torch.max(output, 1)
            if (attack_type == 'untargeted'):
                correct[i] += (predicted_labels == y).sum()
            elif (attack_type == 'targeted'):
                correct[i] += (predicted_labels == target_class).sum() 
                samples_target_class[i] += (y == target_class).sum()
            else:
                correct[i] = 0.0
            i += 1

    for i in range(nmodels):
        # Accuracy of the model
        numerator = float(correct[i] - samples_target_class[i])
        denominator = float(len(test_loader.dataset)- samples_target_class[i])
        accuracy[i] =  numerator / denominator
    
    return accuracy