import os
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup


# Configuration
class Config:
    model_name = "gpt2"
    train_data = "data/Indian_Tax_Law_train.txt"
    val_data = "data/Indian_Tax_Law_val.txt"
    batch_size = 80
    epochs = 1000
    lr = 3e-5
    log_interval = 10
    checkpoint_interval = 100
    max_seq_len = 256
    checkpoint_dir = "checkpoints"
    accumulation_steps = 1
    
# Dataset Class
class TaxLawDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            self.texts = [text.strip() for text in f.readlines()]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=Config.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze()
        }

# Initialize Training
tokenizer = GPT2Tokenizer.from_pretrained(Config.model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(Config.model_name)

train_dataset = TaxLawDataset(tokenizer, Config.train_data)
val_dataset = TaxLawDataset(tokenizer, Config.val_data)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

optimizer = AdamW(model.parameters(), lr=Config.lr)
writer = SummaryWriter()

# Training Loop
def train():
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)
    total_steps = len(train_loader) * Config.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warm-up
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    for epoch in range(1, Config.epochs + 1):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss
            loss = loss / Config.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % Config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation
        if epoch % Config.log_interval == 0:
            val_loss = validate(model, val_loader, device)
            writer.add_scalar('Loss/val', val_loss, epoch)
            logging.info(f'Epoch {epoch} | Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f}')
        
        # Save Checkpoint
        if epoch % Config.checkpoint_interval == 0:
            save_checkpoint(epoch, avg_train_loss, val_loss)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)

def save_checkpoint(epoch, train_loss, val_loss):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    path = f"{Config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()