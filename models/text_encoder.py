import torch
import numpy as np

class TextEncoder:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=None):
        """
        Multilingual Text Encoder using SentenceTransformers.
        Supports both Russian and English prompts.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        
    def _lazy_init(self):
        if self.model is None:
            # We import here so we don't slow down imports if sentence_transformers isn't used
            from sentence_transformers import SentenceTransformer
            print(f"Loading Text Encoder model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Text Encoder loaded successfully.")

    def encode(self, texts):
        """
        Encodes a text or list of texts into embeddings.
        Args:
            texts (str or list of str): Text(s) to encode.
        Returns:
            torch.Tensor: Embedding tensor of shape (embedding_dim,) or (batch_size, embedding_dim).
        """
        self._lazy_init()
        
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            
        # Get embeddings as numpy arrays
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        
        # Convert to torch.Tensor
        embeddings_tensor = torch.from_numpy(embeddings).float()
        
        if is_single:
            return embeddings_tensor[0]
        return embeddings_tensor

    @property
    def embedding_dim(self):
        """Returns the dimension of the embedding vector (usually 384 for paraphrase-multilingual-MiniLM-L12-v2)"""
        if self.model_name == 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
            return 384
        elif 'LaBSE' in self.model_name:
            return 768
        else:
            self._lazy_init()
            return self.model.get_sentence_embedding_dimension()
