import argparse
import collections
import os

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transformers

from model import Transformer

# Initialize global variables
PAD_INDEX = None  # Will be set after tokenizer is created


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(data_name, tokenizer_name):
    if os.path.exists(data_name):
        data = datasets.load_from_disk(data_name)
    else:
        data = datasets.load_dataset(data_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    global PAD_INDEX
    PAD_INDEX = tokenizer.pad_token_id

    def tokenize_and_numericalize_example(example, tokenizer):
        ids = tokenizer(example["text"], truncation=True)["input_ids"]
        return {"ids": ids}

    data = {split: data[split].map(tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}) for split in
            ['train', 'test']}
    return data, tokenizer


def prepare_data_loaders(data, batch_size):
    data_loaders = {}
    if not 'valid' in data:
        test_size = 0.25
        train_valid_data = data['train'].train_test_split(test_size=test_size)
        data["train"] = train_valid_data["train"]
        data["valid"] = train_valid_data["test"]
    for split, dataset in data.items():
        dataset = dataset.with_format(type="torch", columns=["ids", "label"])
        shuffle = split == 'train'
        data_loader = get_data_loader(dataset, batch_size, PAD_INDEX, shuffle=shuffle)
        data_loaders[split] = data_loader
    return data_loaders


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
        batch_label = torch.tensor([i["label"] for i in batch])
        return {"ids": batch_ids, "label": batch_label}

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


def get_accuracy(prediction, label):
    correct_predictions = prediction.argmax(dim=-1).eq(label).sum()
    accuracy = correct_predictions.float() / label.size(0)
    return accuracy


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in tqdm.tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        ids, labels = batch['ids'].to(device), batch['label'].to(device)
        predictions = model(ids)
        loss = criterion(predictions, labels)
        acc = get_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            ids, labels = batch['ids'].to(device), batch['label'].to(device)
            predictions = model(ids)
            loss = criterion(predictions, labels)
            acc = get_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer model for text classification.')
    parser.add_argument('--data_name', type=str, default='imdb', help='The directory where the data is stored.')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='The name of the tokenizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-5, help='The learning rate.')
    parser.add_argument('--seed', type=int, default=1234, help='The seed for random number generation.')
    args = parser.parse_args()

    set_seeds(args.seed)
    data, tokenizer = load_data(args.data_name, args.tokenizer_name)
    data_loaders = prepare_data_loaders(data, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(args.tokenizer_name, 2, False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    best_valid_loss = float("inf")
    metrics = collections.defaultdict(list)
    os.makedirs("model", exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            data_loaders['train'], model, criterion, optimizer, device
        )
        valid_loss, valid_acc = evaluate(data_loaders['train'], model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "model/transformer.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    model.load_state_dict(torch.load("model/transformer.pt"))
    test_loss, test_acc = evaluate(data_loaders['test'], model, criterion, device)

    print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
