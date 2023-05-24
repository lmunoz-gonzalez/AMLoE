from attacks import pgd
import torch

"""
Function for training a single epoch (iteration) with adversarial training and
PGD attack
#Parameters:
    + model
    + optimizer
    + loss
    + x: training inputs
    + y: training labels
    + epsilon: magnitude of the perturbation for the adversarial examples
    + alpha: learning rate for crafting the adversarial examples
    + steps: number of steps in PGD attack
    + starts: number of random starts to initialize the adv perturbations
    + device (CUDA/CPU)    
    + lp_norm: lp_norm constraint for the adv perturbations (linf or l2)
    
#Outputs: 
    + err: loss evaluated on the training datapoints
    + pred: predictions on the inputs
"""

def advTrainingIterPGD(model, optimizer, loss, x, y, epsilon, alpha, steps, 
                       starts,device, lp_norm = 'linf'):   
    model.to(device)
    x.to(device)
    y.to(device)
    #Reset gradients
    optimizer.zero_grad()
    #Perturbed examples
    attack_type = 'untargeted'
    x_adv = pgd.pgd_attack(x, y, model, epsilon, alpha, steps, 
                       starts, device, attack_type, lp_norm)
    x_adv.to(device)
    pred = model(x_adv)
    pred.to(device)
    err = loss(pred, y).to(device)
    err.backward()
    #Update optimizer
    optimizer.step()
    return err, pred



"""
Function for adversarially training the model with PGD attack
Parameters:
    + model             
    + loss             
    + optimizer         
    + train _loader
    + test_loader
    + epochs
    + epsilon: magnitude of the perturbation for the adversarial examples
    + alpha: learning rate for crafting the adversarial examples
    + steps: number of steps in PGD attack
    + starts: number of random starts to initialize the adv perturbations
    + lp_norm constraint for the adv perturbations (linf or l2)
    + device: CUDA/CPU
    + epoch_evaluation: evaluate the training/test accuracy every
        "epoch_evaluation" number of epochs (0 for no evaluation during
                                             the training)
    + epoch_save_checkpoint: save the parameters of the model and the 
        training state every "epoch_save_checkpoint" number of epochs
        (0 for not saving the model during training)
    + PATH: path for saving/loading the model 
    + resumeTraining: True for resume training of a  model stored in PATH.
        False for starting training from scratch. 
    
Output:
    + model: resulting model after adversarial training
    + optimizer
"""

def AdvTrainPGD(model, loss, optimizer, train_loader, test_loader, 
          epochs, epsilon, alpha, steps, starts, lp_norm, device, 
          epoch_evaluation = 0, epoch_save_checkpoint = 0, PATH = ".dummy.pt",
          resumeTraining = False):
    
    # set objects for storing metrics
    train_losses = []
 
    # If we are resuming training we load parameters from the previously saved
    # model
    if (resumeTraining): 
        print("Loading model...")
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        initial_epoch = 1
 
    
    # Train model
    for epoch in range(initial_epoch, epochs + 1):
        epoch_losses = []
        for x_tr, y_tr in train_loader:
            error, _ = advTrainingIterPGD(model, optimizer, loss, 
                             x_tr.to(device), y_tr.to(device), epsilon, 
                             alpha, steps, starts, device, lp_norm)
            epoch_losses.append(error.cpu().data.numpy())
        #print(f'Training loss: {error:.4f}')
        tr_error = sum(epoch_losses)/len(train_loader.dataset)
        print("Training epoch: ", epoch)
        print(f'Training loss: {tr_error:.4f}')
        train_losses.append(tr_error)
    
        # Evaluate the model every "epoch_evaluation" number of epochs
        # If epoch_evaluation = 0: we never evaluate the model during training
        if ((epoch_evaluation != 0) and (epoch % epoch_evaluation == 0)):
            training_acc = get_accuracy(model, train_loader, device)
            test_acc = get_accuracy(model, test_loader, device)
            # We show accuracy on the clean dataset
            print(f'TRAINING ACCURACY: {100*training_acc:.4f}%')
            print(f'TEST ACCURACY: {100*test_acc:.4f}%')
            
        # Save the model and its parameters every "epoch_save_checkpoint" 
        # number of epochs.
        # If epoch_save_checkpoint = 0, we never save the model
        if ((epoch_save_checkpoint != 0) and (epoch % epoch_save_checkpoint == 0)):
            print("\n-->Saving model as ", PATH, "\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
        #Print progress bar
        progress(epoch, epochs, 40)
        #progress(100*epoch/epochs, width=40)
    
    #Save at the end of training
    if ((epoch_save_checkpoint != 0) and (epochs % epoch_save_checkpoint != 0)):
            print("\n-->Saving model as ", PATH, "\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
    return model, optimizer


"""
Auxiliary function to compute the model's accuracy over the entire
data_loader
Parameters:
    + model
    + data_loader
    + device: (CUDA/CPU)
    
Output: 
    + accuracy
"""
def get_accuracy(model, data_loader, device):
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


# Function to show progress bar during training
def progress(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar +
          '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='\n')

