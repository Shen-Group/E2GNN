import os
import pickle
import torch

class RemoveMean(object):
    """Remove mean for a Tensor and restore it later. """

    def __init__(self, mean=None, tensor=None):
        """tensor is taken as a sample to calculate the"""
        if mean:
            self.mean = mean
        else:
            assert tensor is not None
            self.mean = torch.mean(tensor)

    def norm(self, tensor):
        return tensor - self.mean
    
    def denorm(self, normed_tensor):
        return normed_tensor + self.mean
    
    def state_dict(self):
        return {'mean': self.mean}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, mean=None, std=None, tensor=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if mean is None or std is None:
            assert tensor is not None
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        else:
            self.mean = mean
            self.std = std

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg