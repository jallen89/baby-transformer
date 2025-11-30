import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src import utils
from src.model import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataloaders(config):
    """
    Loads the dataset, splits it into train, validation, and test sets,
    and returns the corresponding DataLoaders.
    """
    dataset = utils.FrenchEnglishDataset(config['data_path'])
    dataset.intialize_dataset()
    total_size = len(dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    logging.info(f"Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset

def train_model(config, model, train_loader, criterion, optimizer, scheduler, device):
    """
    Main training loop.
    """
    french_word_count = train_loader.dataset.dataset.french_tokenizer.vocab_size
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0

        for idx, batch in enumerate(train_loader):
            english, french = batch['english'].to(device), batch['french'].to(device)

            decoder_input = french[:, :-1]

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
        logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.4f}")

def main():
    """
    Main function to run the training script.
    """
    config = {
        'data_path': 'data/fra.txt',
        'train_split': 0.8,
        'val_split': 0.1,
        'batch_size': 64,
        'num_epochs': 10,
        'lr': 0.001,
        'model_output': 'weights/english-french.pt',
        'src_output_tokenizers': 'weights/english-tokenizer.json',
        'tgt_output_tokenizers': 'weights/french-tokenizer.json',
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    train_loader, _, _, dataset = get_dataloaders(config)

    english_word_count = dataset.english_tokenizer.vocab_size
    french_word_count = dataset.french_tokenizer.vocab_size
    
    model = Transformer(english_word_count, french_word_count).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
    )

    train_model(config, model, train_loader, criterion, optimizer, scheduler, device)

    # save model and tokenizers to disk
    model_path = config['model_output']
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    dataset.english_tokenizer.save_vocab(config['src_output_tokenizers'])
    dataset.french_tokenizer.save_vocab(config['tgt_output_tokenizers'])

    test_model(model, dataset.english_tokenizer, dataset.french_tokenizer)


def test_model(model, english_tokenizer, french_tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create encoder input.
    sentence = "Don't resist us."
    encoded_sentence = english_tokenizer.encode_sentence(sentence)
    encoder_input = encoded_sentence.unsqueeze(0) # Shape: 1 x EncSeqLength
    encoder_input = encoder_input.to(device)

    # Create decoder input.
    decoder_sentence = french_tokenizer.encode_sentence("<sos>", add_sos_eos=False)
    decoder_input = decoder_sentence.unsqueeze(0) # Shape: 1 x 1
    decoder_input = decoder_input.to(device)


    # Start decoding loop
    with torch.no_grad():
        for _ in range(20): # Using _ since i is not needed
            # 1. Forward pass
            # logits shape: 1 x CurrentSeqLength x VocabSize
            logits = model(encoder_input, decoder_input)
            
            # 2. Get the logits for the *last* token position (the one we are predicting)
            # next_token_logits shape: 1 x VocabSize
            next_token_logits = logits[:, -1, :]
            
            # 3. Greedy step: get the token ID with the highest probability
            # next_token_id shape: 1 (or 1x1)
            next_token_id = next_token_logits.argmax(dim=-1)
            
            # 4. Check for end-of-sequence
            if next_token_id.item() == french_tokenizer.word_to_id['<eos>']:
                print("Translation finished with <eos>")
                break
                
            # 5. Append the new token ID to the decoder input for the next step
            # Using torch.cat and dim=1 (the sequence dimension)
            decoder_input = torch.cat((decoder_input, next_token_id.unsqueeze(0)), dim=1)
            
            # For debugging/tracking
            print(f"Current Translation: {french_tokenizer.decode(decoder_input.squeeze(0))}")
            
        print("\nFinal Decoder Input Tensor:")
        print(decoder_input)

if __name__ == '__main__':
    main()
