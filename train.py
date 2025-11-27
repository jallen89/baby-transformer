from src import utils, model
import logging
from torch.utils.data import DataLoader, random_split
import torch 
import torch.nn as nn
import torch.optim as optim
from src.model import Transformer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load dataset 

dataset = utils.FrenchEnglishDataset('data/fra.txt')
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size


# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

logging.info(f"Training set size: {len(train_dataset)}")
logging.info(f"Validation set size: {len(val_dataset)}")
logging.info(f"Test set size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Do Training

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")


### Create 
NUM_EPOCHS = 1
criterion = nn.CrossEntropyLoss(ignore_index=0)


french_word_count = dataset.french_tokenizer.vocab_size
english_word_count = dataset.english_tokenizer.vocab_size
model = Transformer(english_word_count, french_word_count)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
)



for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0

    for idx, batch in enumerate(train_loader):
        english, french = batch['english'], batch['french']

        english = english.to(device)
        french = french.to(device)


        decoder_input = french[:, :-1]
        # No need for .to(device) again, it's already on the device from the `french` tensor

        optimizer.zero_grad()
        logits = model(english, decoder_input)
        labels = french[:, 1:]

        logits = logits.reshape(-1, french_word_count)
        labels = labels.reshape(-1)

        loss = criterion(logits, labels)
        if idx % 1000 == 0:
            logging.info(f"Batch {idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)
    logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")
