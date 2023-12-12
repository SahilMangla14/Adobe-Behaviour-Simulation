import torch.optim as optim

def train_model(model, train_dataloader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        count = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            print(f"Batch : {count} done for epoch {epoch} and loss : {loss}")
            count += 1
        print(f'Epoch {epoch + 1}/{num_epochs}')