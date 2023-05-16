import torch

"""
Function for training a single epoch (iteration)
#Parameters:
    + model
    + optimizer
    + loss
    + x: training inputs
    + y: training labels
    + device (CUDA/CPU)    
    
#Outputs: 
    + err: loss evaluated on the training datapoints
    + pred: predictions on the inputs
"""
def train_iter(model, optimizer, loss, x, y, device):
    #Send model and data to device
    model.to(device)
    x.to(device)
    y.to(device)
    #Reset gradients
    optimizer.zero_grad()
    #Predictions
    pred = model(x)
    pred.to(device)
    #Compute loss
    err = loss(pred, y).to(device)
    err.backward()
    #Update optimizer
    optimizer.step()
    return err, pred



"""
Function for evaluating the model's performance on a validation/test dataset
Parameters:
    + model
    + Xtest: test/validation inputs
    + device (CUDA/CPU)

Outputs:
    + Ypred: soft labels / class probabilities
    + label: prediction (class with the highest probability)

"""
def predict(model, Xtest, device):
    model.to(device)
    Xtest.to(device)
    with torch.no_grad():
        Ypred = model.forward(Xtest).to(device)
        label = torch.max(Ypred, 1)
    return Ypred, label


"""
Function for training the model.
Parameters:
    + model             
    + loss             
    + optimizer         
    + train _loader
    + test_loader
    + epochs
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
    + model: resulting model after training
    + optimizer
"""

def train(model, loss, optimizer, train_loader, test_loader, epochs, 
          device, epoch_evaluation = 0, epoch_save_checkpoint = 0, 
          PATH = ".dummy.pt", resumeTraining = False):
    
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
        model.train()
        for x_tr, y_tr in train_loader:
            error, _ = train_iter(model, optimizer, loss, 
                             x_tr.to(device), y_tr.to(device), device)
            epoch_losses.append(error.cpu().data.numpy())
        tr_error = sum(epoch_losses)/len(train_loader.dataset)
        print("Training epoch: ", epoch)
        print(f'Training loss: {tr_error:.4f}')
        train_losses.append(tr_error)
        
        # Evaluate the model every "epoch_evaluation" number of epochs
        # If epoch_evaluation = 0: we never evaluate the model during training
        if ((epoch_evaluation != 0) and (epoch % epoch_evaluation == 0)):
            model.eval()
            training_acc = get_accuracy(model, train_loader, device)
            test_acc = get_accuracy(model, test_loader, device)
            print(f'TRAINING ACCURACY: {100*training_acc:.4f}%')
            print(f'TEST ACCURACY: {100*test_acc:.4f}%')
            
        # Save the model and its parameters every "epoch_save_checkpoint" 
        # number of epochs.
        # If epoch_save_checkpoint = 0, we never save the model
        if ((epoch_save_checkpoint != 0) and (epoch % epoch_save_checkpoint == 0)):
            print("-->Saving model as ", PATH, "\n")
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


"""
Auxiliary function to compute the model's top-K accuracy over the entire
data_loader
Parameters:
    + model
    + data_loader
    + device: (CUDA/CPU)
    + k
    
Output: 
    + Top-K accuracy
"""
def get_topK_accuracy(model, data_loader, device, k):
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.topk(y_prob, k, 1)
            n += y_true.size(0)            
            for i in range(k):
                correct_pred += (predicted_labels[:,i] == y_true).sum()

    return correct_pred.float() / n


# Function to show progress bar during training
def progress(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar +
          '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='\n')




