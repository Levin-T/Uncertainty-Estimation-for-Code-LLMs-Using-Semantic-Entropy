from embedding import Embedding, calculate_similarity_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



class OpenSourceLLM_Embedding:
    def __init__(self, model_name="bigcode/starcoder2-7b"):
        # Use mixed precision to reduce memory usage
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
             torch_dtype=torch.bfloat16  # Use mixed precision
        )
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the appropriate device
        self.model.eval()  # Set model to evaluation mode

    def get_embedding(self, code):
        # Tokenize input with truncation and max length to prevent excessive memory usage
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():  # Disable gradients to save memory
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use the last hidden layer
        # Compute the mean of the hidden states as the embedding
        embedding = hidden_states.mean(dim=1).to(dtype=torch.float16).squeeze().cpu().numpy()
        return embedding

    def calculate_similarity(self, code1, code2):
        # Get embeddings for both code snippets
        embedding1 = self.get_embedding(code1)
        embedding2 = self.get_embedding(code2)
        # Calculate similarity using the provided similarity function
        return calculate_similarity_template(embedding1, embedding2)

if __name__ == '__main__':
    # Clear GPU memory before running the script
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Initialize the embedding class with the open-source model
    open_source_model = OpenSourceLLM_Embedding("bigcode/starcoder2-7b")

    # Example code snippets
    code1 = "def abc(a, b): return a + b"
    code2 = "def ab(a, b): return a - b"