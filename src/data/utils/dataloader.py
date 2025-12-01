import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_collated_dataloader(dataset, shuffle=True, batch_size=32):
    def collate_pad(batch):
        xs, ys = zip(*batch)
        lengths = torch.tensor([x.size(0) for x in xs])
        x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return x_padded, lengths, y_tensor

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, collate_fn=collate_pad)