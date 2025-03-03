import torch
from torcheval.metrics.functional import multiclass_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loss_function, optimizer, train_dataset):
    '''
    Function that trains the given CNN model for 1 epoch
    and returns the average training loss for that epoch.
    '''
    model.train()
    epoch_loss = 0
    for minibatch_images, minibatch_labels in train_dataset:
        minibatch_images, minibatch_labels = minibatch_images.to(device), minibatch_labels.to(device)
        optimizer.zero_grad()
        output = model(minibatch_images)

        loss = loss_function(output, minibatch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()*minibatch_images.shape[0]
    return epoch_loss/len(train_dataset.dataset)


def validate(model, loss_function, validation_dataset):
    '''
    Function that validates model by calculating the loss on the validation dataset
    and returns the average validation loss.
    '''
    model.eval()
    validation_loss = 0
    output_list = []
    true_labels = []
    for minibatch_images, minibatch_labels in validation_dataset:
        minibatch_images, minibatch_labels = minibatch_images.to(device), minibatch_labels.to(device)
        with torch.no_grad():
            output = model(minibatch_images)
            loss = loss_function(output, minibatch_labels)
            validation_loss += loss.item()*minibatch_images.shape[0]
            output_list.append(output.detach().cpu())
            output_tensor = torch.cat(output_list)
            true_labels.append(minibatch_labels.detach().cpu())
            true_labels_tensor = torch.cat(true_labels)
    return validation_loss/len(validation_dataset.dataset), output_tensor, true_labels_tensor


def test(model, test_dataset):
    '''
    Function that uses model to make predictions on dataset and returns a tensor
    containing the model's predictions for dataset.
    '''
    model.eval()
    output_list  = []
    for minibatch_images, minibatch_labels in test_dataset:
        minibatch_images, minibatch_labels = minibatch_images.to(device), minibatch_labels.to(device)
        with torch.no_grad():
            output = model(minibatch_images)
            output_list.append(output.detach().cpu())
            output_tensor = torch.cat(output_list)
    return output_tensor




class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        '''
        initiate EarlyStopping object that will cause model to stop when the number of epochs without improvement
        exceeds the patience.
        '''
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.epochs_without_improvement = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.epochs_without_improvement = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)




def train(model, loss_function, optimizer,
          train_dataset, dev_dataset, num_epochs=10, 
          record_weights=None, early_stopping=None):
    '''
    Function that trains the model over num_epochs with the train_epoch function
    and that returns a dictionary with the training history
    '''
    param_history = []
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
    for epoch in range(num_epochs):
        if epoch == 0 and record_weights:
            param_history.append(record_weights(model))

        current_train_loss = train_epoch(model=model, loss_function=loss_function, optimizer=optimizer, 
                                         train_dataset=train_dataset)
        current_validation_loss, predicted, true_labels = validate(model=model, loss_function=loss_function, validation_dataset=dev_dataset)
        current_validation_accuracy = multiclass_accuracy(predicted, true_labels)

        # Record train_loss and validation_loss in the history dictionary
        history["train_loss"].append(current_train_loss)
        history["validation_loss"].append(current_validation_loss)
        history["validation_accuracy"].append(current_validation_accuracy)

        print("epoch {}".format(epoch + 1) + \
                        " train loss: {:.4f} ".format(current_train_loss) + \
                        " validation loss: {:.4f} ".format(current_validation_loss) + \
                        " validation accuracy: {:.4f} ".format(current_validation_accuracy))

        if record_weights:
            param_history.append(record_weights(model))
        history["params"] = param_history


        if early_stopping is not None: 
            early_stopping(current_validation_loss, model)
            if early_stopping.early_stop:
                print("EARLY STOPPING ")
                break

    early_stopping.load_best_model(model)
    return history