from torch.utils.data import random_split

def train_test_split_dataset(dataset, train_size=0.8):
    train_size = int(len(dataset) * train_size)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])
