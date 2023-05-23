import torch
import torch.nn as nn

'''
BASIC ITERATIVE METHOD (BIM) ATTACK

Paper: “Explaining and Harnessing Adversarial Examples”. Goodfellow et al. 2014: 
    https://arxiv.org/abs/1412.6572

Code Adapted from GitHub user Harry24k: 
    https://github.com/Harry24k/adversarial-attacks-pytorch 
'''


'''
Function to craft the FGSM attack on a set of labeled data points (x, y)

Parameters: 
    + x:        inputs
    + y:        labels
    + model:    model used to craft the adversarial examples
    + epsilon:  perturbation
    + device:   CPU/GPU
    + attack_type: targeted / untargeted
    
Outputs:
    + perturbed_x: resulting adversarial perturbations
    
Note: for targeted attacks, y should define the set of target labels for each
    data point x. 
    
'''

def fgsm_attack(x, y, model, epsilon, device, attack_type='untargeted'):
    # Define loss depending on the attack type
    loss = nn.CrossEntropyLoss()

    #Rescale epsilon (to account for data normalization)
    epsilon = epsilon*(x.max() - x.min())

    perturbed_x = x.clone().detach()

    perturbed_x.requires_grad = True
    # Forward pass the data through the model
    output = model(perturbed_x)
    # Compute loss depending on the attack type
    if (attack_type == 'untargeted'):
        cost = loss(output, y)
    elif(attack_type == 'targeted'):
        cost = -loss(output, y)
    else: 
        cost = loss(output, y)
        
    # Compute gradient
    grad = torch.autograd.grad(cost, perturbed_x,
                               retain_graph=False,
                               create_graph=False)[0]

    # Compute adversarial example
    perturbed_x = perturbed_x.detach() + epsilon*grad.sign()
    perturbed_x = torch.clamp(perturbed_x, min=x.min(), max=x.max()).detach().to(device)

    return perturbed_x



'''
Function to test the effectiveness of a FGSM attack on a set of test data points

Parameters: 
    + model:    model used to craft the adversarial examples
    + device:   CPU/GPU
    + test_loader: data loader with the set of test points
    + epsilon:  perturbation
    + device:   CPU/GPU
    + attack_type: targeted / untargeted
    
Outputs:
    + accuracy: robust accuracy of the model evaluated on the resulting adv. 
                examples.
    + x_adv: resulting adversarial perturbations
        
'''

def test(model, device, test_loader, epsilon,
         attack_type='untargeted', target_class=0):

    # Accuracy counter
    correct = 0
    samples_target_class = 0
    adv_examples = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        x.requires_grad = True
        model.zero_grad()

        if (attack_type == 'untargeted'):
            x_adv = fgsm_attack(x, y, model, epsilon, device, attack_type)
        elif(attack_type == 'targeted'):
            y2 = y.clone().detach().to(device)
            y2[:] = target_class
            x_adv = fgsm_attack(x, y2, model, epsilon, device, attack_type)
        else:
            #Perform an untargeted attack by default
            x_adv = fgsm_attack(x, y, model, epsilon, device, attack_type)

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

        # Append adversarial examples for visualization
        adv_examples.append(x_adv.squeeze().detach().cpu().numpy())
    # Accuracy of the model
    numerator = float(correct - samples_target_class)
    denominator = float(len(test_loader.dataset)- samples_target_class)
    accuracy =  numerator / denominator
    return accuracy, adv_examples



'''
Function to test the effectiveness FGSM transfer attacks

Parameters: 
    + S_model:  source model used to craft the adv. examples
    + model:    set of target models
    + device:   CPU/GPU
    + test_loader: data loader with the set of test points
    + epsilon:  perturbation
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
            x_adv = fgsm_attack(x, y, S_model, epsilon, device, attack_type) 
            
        elif (attack_type == 'targeted'):
            y2 = y.clone().detach().to(device)
            y2[:] = target_class
            x_adv = fgsm_attack(x, y2, S_model, epsilon, device, attack_type) 

        else:
             x_adv = fgsm_attack(x, y, S_model, epsilon, device, attack_type) 
        

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

