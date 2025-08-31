from embedding import Embedding, calculate_similarity_template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class OpenSourceEmb(Embedding):

    def __init__(self, model_name, max_length=32768, load_model=True):
        if(load_model):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda")
            self.max_length = max_length

    def get_embedding(self, code):
        instruction = "Find the semantic embedding of the code snipped."
        with torch.no_grad():
            embeddings = self.model.encode_queries(
                [code], instruction=instruction, max_length=self.max_length
            ).to("cuda")
        normalized_embedding = F.normalize(embeddings, p=2, dim=1)
        return normalized_embedding[0].cpu().detach().numpy()

    def calculate_similarity(self, emb_code1, emb_code2):
        return calculate_similarity_template(emb_code1, emb_code2)


if __name__ == '__main__':
    open_source_model = OpenSourceEmb('Salesforce/SFR-Embedding-Code-2B_R')

    # Example code snippets
    code1 = "def abc(a,b) = return a + b"
    code2 = "def abc(a,b) = return a - b"

    # Calculate similarity between the code snippets
    similarity_score = open_source_model.calculate_similarity(code1, code2)
    print(f"Similarity Score: {similarity_score}")