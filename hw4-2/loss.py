"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(L1DepthLoss, self).__init__()
    self.args = args
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
      normalized_loss_per_sent = loss_per_sent / length_batch.float()
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents

class RankDepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self, args):
    super(RankDepthLoss, self).__init__()
    self.args = args
    self.word_pair_dims = 1

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      seq_len = predictions_masked.size(1)
      rank_loss = predictions_masked[:,0]*0
      for m in range(seq_len-1):
        #print(predictions_masked[:,m].repeat(1,seq_len-m).size())
        sign = LBSign.apply
        pred_diff = predictions_masked[:,m].unsqueeze(1).repeat(1,seq_len-m-1) -predictions_masked[:,m+1:]
        #pred_diff = 2*(pred_diff>0).float()-1
        label_diff = labels_masked[:,m].unsqueeze(1).repeat(1,seq_len-m-1) -labels_masked[:,m+1:]
        #label_diff = 2*(label_diff>0).float()-1
        
        #print((1 - sign((pred_diff)*(label_diff))))
        #s = (torch.sign(label_diff)**2).detach()
        #diff_mul = (pred_diff)*(label_diff)
        diff_mul = torch.sign(label_diff)
        #sign_diff = (diff_mul/(torch.abs(diff_mul)+1e-6) )
        
        
        rank_loss += torch.sum(F.relu(1 - diff_mul*pred_diff)*labels_1s[:,m+1:],dim=self.word_pair_dims)#*labels_1s[m]
        
      #print(predictions_masked.size())
      loss_per_sent = rank_loss
      #print(loss_per_sent)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
      #print(batch_loss)
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents

class LBSign(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input):
    return torch.sign(input)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clamp_(-1, 1)


