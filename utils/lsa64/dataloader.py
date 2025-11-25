import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_collated_dataloader(dataset, idx, shuffle=True):
    def collate_pad(batch):
        xs, ys = zip(*batch)
        hand_labels = [f[idx] for f in ys]
        lengths = torch.tensor([x.size(0) for x in xs])
        x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
        y_tensor = torch.tensor(hand_labels, dtype=torch.float32).unsqueeze(1)
        return x_padded, lengths, y_tensor

    return DataLoader(dataset, batch_size=32, shuffle=shuffle, num_workers=6, collate_fn=collate_pad)