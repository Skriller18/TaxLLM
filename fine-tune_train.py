import os
import logging
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Configuration
class Config:
    model_name = "meta-llama/Llama-2-7b"  # Change to your preferred LLaMA model
    train_data = "data/tax_law_train.txt"
    val_data = "data/tax_law_val.txt"
    batch_size = 16  # Reduced batch size as LLaMA is larger
    epochs = 1000
    lr = 1e-5  # Adjusted learning rate for LLaMA
    log_interval = 10
    checkpoint_interval = 100
    max_seq_len = 512  # LLaMA supports longer sequences
    checkpoint_dir = "checkpoints"
    accumulation_steps = 4  # Increased for memory management
    dropout_rate = 0.1  # Standard dropout for LLaMA
    weight_decay = 0.01  # L2 regularization
    patience = 5  # Early stopping patience
    use_8bit = True  # Optional: for lower memory usage

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
tokenizer = LlamaTokenizer.from_pretrained(Config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model - LLaMA has different parameter naming
if Config.use_8bit:
    model = LlamaForCausalLM.from_pretrained(
        Config.model_name,
        load_in_8bit=True,
        device_map="auto"
    )
else:
    model = LlamaForCausalLM.from_pretrained(Config.model_name)

# Prepare datasets
train_dataset = TaxLawDataset(tokenizer, Config.train_data)
val_dataset = TaxLawDataset(tokenizer, Config.val_data)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)

# For parameter-efficient fine-tuning (optional but recommended for LLaMA)
from peft import get_peft_model, LoraConfig, TaskType

# PEFT configuration - using LoRA for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

if not Config.use_8bit:
    model = get_peft_model(model, peft_config)
    
# Only optimize trainable parameters if using PEFT
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=Config.lr, weight_decay=Config.weight_decay)

# Learning rate scheduler
total_steps = len(train_loader) * Config.epochs
warmup_steps = int(0.1 * total_steps)  # 10% warm-up
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# TensorBoard logging
writer = SummaryWriter(log_dir="runs/tax_law_training")

# Early stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training Loop
def train():
    global best_val_loss, epochs_without_improvement
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(1, Config.epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
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
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(epoch, avg_train_loss, val_loss, is_best=True)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= Config.patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Save checkpoint periodically
        if epoch % Config.checkpoint_interval == 0:
            save_checkpoint(epoch, avg_train_loss, val_loss)

# Validation function
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

# Checkpoint saving
def save_checkpoint(epoch, train_loss, val_loss, is_best=False):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    if is_best:
        path = f"{Config.checkpoint_dir}/best_checkpoint.pt"
    else:
        path = f"{Config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()