import torch.optim as optim

def train(model, train_dataloader, criterion, optimizer, device, num_epochs=30):
    model.train()
    count = 0
    for batch in train_dataloader:
        date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, media_input_ids, media_attention_mask, company_input_ids, company_attention_mask ,labels = batch

        date_input_ids, date_attention_mask, labels = date_input_ids.to(device), date_attention_mask.to(device), labels.to(device)
        content_input_ids, content_attention_mask = content_input_ids.to(device), content_attention_mask.to(device)
        username_input_ids, username_attention_mask = username_input_ids.to(device), username_attention_mask.to(device)
        media_input_ids, media_attention_mask = media_input_ids.to(device), media_attention_mask.to(device)
        company_input_ids, company_attention_mask = company_input_ids.to(device), company_attention_mask.to(device)

        optimizer.zero_grad()

        # Extract embeddings for each feature
        outputs = model(date_input_ids, date_attention_mask, content_input_ids, content_attention_mask, username_input_ids, username_attention_mask, media_input_ids, media_attention_mask, company_input_ids, company_attention_mask)

        # Calculate loss
        loss = criterion(outputs, labels.unsqueeze(1).float())
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Batch : {count} done for epoch {epoch} and loss : {loss}")
        count += 1

    print(f'Epoch {epoch + 1}/{num_epochs}')