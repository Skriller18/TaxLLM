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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        return model
    
    def generate(self, prompt, max_length=300, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(
            f"TAX QUESTION: {prompt}\nANSWER:",
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True  # Ensure padding is handled
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Pass attention mask
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,  # Enable sampling for temperature/top_p
            pad_token_id=self.tokenizer.eos_token_id,  # Explicitly set pad token
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("ANSWER:")[-1].strip()

if __name__ == "__main__":
    assistant = TaxLawAssistant("checkpoints/checkpoint_epoch_100.pt")
    while True:
        try:
            question = input("Ask a tax law question (or 'exit' to quit): ")
            if question.lower() == "exit":
                break
                
            print("\nGenerating response...")
            response = assistant.generate(question)
            print(f"\nResponse: {response}\n")
            
        except KeyboardInterrupt:
            break

    print("\nExiting tax law assistant...")