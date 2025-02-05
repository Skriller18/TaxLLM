import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TaxLawAssistant:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        return model
    
    def generate(self, prompt, max_length=300, temperature=0.7):
        inputs = self.tokenizer(
            f"TAX QUESTION: {prompt}\nANSWER:",
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("ANSWER:")[-1].strip()

if __name__ == "__main__":
    assistant = TaxLawAssistant("checkpoints/checkpoint_epoch_1000.pt")
    while True:
        question = input("Ask a tax law question: ")
        print(assistant.generate(question))