from openai import OpenAI
from helpers import get_api_key
from embedding import Embedding, calculate_similarity_template


class OpenAI_emb(Embedding):

    # Initialize the OpenAI client with your API key
    def __init__(self, model):
        self.client = OpenAI(api_key=get_api_key())
        self.model = model

    def get_embedding(self, code):
        response = self.client.embeddings.create(
            input=code,
            model=self.model
        )
        return response.data[0].embedding

    def calculate_similarity(self, code1, code2):
        return calculate_similarity_template(self.get_embedding(code1), self.get_embedding(code2))


if __name__ == '__main__':
    openAI = OpenAI_emb("text-embedding-3-large")
    code1 = "def(a,b) = return a + b"
    code2 = "def(a,b) = return a - b"

    print(openAI.calculate_similarity(code1, code2))
