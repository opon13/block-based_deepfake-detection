import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
import wandb
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
from IPython.display import clear_output

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def training(model, loaders, epochs, optimizer, loss_fn, starting_epoch=0, scheduler=None, mode_logs:str='online', model_name='', save_best_model=False, saving_path='', initial_val_to_beat=10):
  if save_best_model: assert saving_path.endswith('.pt')
  wandb.init(
    project='overfitting-approach',
    name=model_name,
    mode=mode_logs)
  model.to(dev)
  train_loader, valid_loader = loaders['train'], loaders['valid']
  losses={'train' : [], 'val' : []}
  accuracies={'train' : [], 'val' : []}
  best_val_loss = initial_val_to_beat
  staring_time = time.time()
  for epoch in tqdm(range(starting_epoch, epochs), desc='epochs'):
    model.train()
    running_train_loss = 0.0
    running_train_accuracy = 0.0
    for batch, data in enumerate(train_loader):
      images, labels, _ = data
      images = images.to(dev)
      labels = labels.squeeze_().to(dev)
      output = model(images)
      loss = loss_fn(output, labels)
      #backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #calculate performance
      preds = torch.argmax(output, 1)
      batch_accuracy = (preds == labels).sum().item()/images.size(0)
      running_train_loss += loss.item()
      running_train_accuracy += batch_accuracy
    losses['train'].append(running_train_loss / len(train_loader))
    accuracies['train'].append(running_train_accuracy / len(train_loader))
    model.eval()
    running_val_loss = 0.0
    running_val_accuracy = 0.0
    with torch.no_grad():
      for batch, data in enumerate(valid_loader):
        images, labels, _ = data
        images = images.to(dev)
        labels = labels.squeeze_().to(dev)
        output = model(images)
        preds = torch.argmax(output, 1)
        loss = loss_fn(output, labels)
        #calculate performance
        batch_accuracy = (preds == labels).sum().item()/images.size(0)
        running_val_loss += loss.item()
        running_val_accuracy += batch_accuracy
    losses['val'].append(running_val_loss/len(valid_loader))
    accuracies['val'].append(running_val_accuracy/len(valid_loader))
    wandb.log({'train':{'accuracy':accuracies['train'][-1],'loss':losses['train'][-1]}, 'valid':{'accuracy':accuracies['val'][-1], 'loss':losses['val'][-1]}})
    clear_output(wait=True)
    print(f'Epoch {epoch+1}/{epochs}: ',
          f'train loss {losses["train"][-1]:.4f}, val loss: {losses["val"][-1]:.4f}, ',
          f'train acc {accuracies["train"][-1]*100:.2f}%, val acc: {accuracies["val"][-1]*100:.2f}%')
    if scheduler != None:
      scheduler.step()
    if save_best_model and losses['val'][-1]<best_val_loss:
      best_val_loss=losses['val'][-1]
      best_epoch = epoch
      torch.save(model.state_dict(), saving_path)
  ending_time = time.time()
  esecution_time = ending_time - staring_time
  if save_best_model: print(f'\nmodel saved at epoch: {best_epoch+1}')
  print(f'total training time: {(esecution_time/60):2f} minutes.\n')
  wandb.finish()

def testing(model, test_loader, loss_fn, plot_cm=False, save_cm=False, classes=None, average='binary', saving_dir='', model_name='', convert_to_binary=False):
  assert not saving_dir.endswith('/')
  model.eval()
  model.to(dev)
  total_loss = 0
  total_correct = 0
  total_samples = 0
  y_true = []
  y_pred = []
  for batch, data in enumerate(test_loader):
    inputs, labels, _ = data
    inputs = inputs.to(dev)
    labels = labels.to(dev)
    with torch.no_grad():
        outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    total_loss += loss.item() * inputs.size(0)
    _, predicted = torch.max(outputs.data, 1)
    if convert_to_binary and predicted != 0: predicted -=1
    total_correct += (predicted == labels).sum().item()
    total_samples += labels.size(0)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())
  average_loss = total_loss / total_samples
  accuracy = total_correct / total_samples
#  print(f"Test Loss: {average_loss:.4f}")
  print(f"{accuracy * 100:.2f}%, {recall_score(y_true, y_pred, average=average)*100:.2f}%, {precision_score(y_true, y_pred, average=average)*100:.2f}%, {f1_score(y_true, y_pred, average=average)*100:.2f}%")
  if plot_cm:
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    if save_cm:
      cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)
      cm_display.plot()
      plt.savefig(os.path.join(saving_dir, f'testing_{model_name}-cm.png'))