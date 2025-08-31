from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_template(emb1, emb2):
    embedding1 = np.array(emb1).reshape(1, -1)
    embedding2 = np.array(emb2).reshape(1, -1)

    if embedding1.shape[0] != embedding2.shape[0]:
        raise ValueError()
    return cosine_similarity(embedding1, embedding2)[0][0]


class Embedding(ABC):

    @abstractmethod
    def get_embedding(self, code):
        pass

    @abstractmethod
    def calculate_similarity(self, code1, code2):
        pass
