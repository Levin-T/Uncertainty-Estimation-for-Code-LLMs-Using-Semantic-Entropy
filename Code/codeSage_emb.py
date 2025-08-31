import torch
from transformers import AutoModel, AutoTokenizer
from embedding import Embedding, calculate_similarity_template


class CodeSage_Connector(Embedding):
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def get_embedding(self, code):
        # Tokenize and encode the code snippet using the specified method
        inputs = self.tokenizer.encode(code, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Forward pass to get the embedding
            embedding = self.model(inputs)[0].mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def calculate_similarity(self, code1, code2):
        emb1 = self.get_embedding(code1)
        emb2 = self.get_embedding(code2)
        similarity = calculate_similarity_template(emb1, emb2)
        return similarity


if __name__ == '__main__':
    code1 = "def add(a, b): return a + b"
    code2 = "def add(a, b): return a + b"
    model = CodeSage_Connector("codesage/codesage-large-v2")
    print("Similarity:", model.calculate_similarity(code1, code2))
