
from joblib import dump, load
from loguru import logger

from umda import smi_vec


class EmbeddingModel(object):
    def __init__(self, w2vec_obj, transform=None, radius: int = 1) -> None:
        self._model = w2vec_obj
        self._transform = transform
        self.radius = radius

    @property
    def model(self):
        return self._model

    @property
    def transform(self):
        return self._transform

    def pipeline(self, smi: str):
        vector = smi_vec.smi_to_vector(smi, self.model, self.radius)
        if self.transform is not None:
            new_vector = self.transform.transform(vector)
        else:
            new_vector = vector
        return new_vector

    def __call__(self, smi: str):
        return self.pipeline(smi)

    @classmethod
    def from_pkl(cls, w2vec_path, transform_path=None, **kwargs):
        w2vec_obj = smi_vec.load_model(w2vec_path)
        if transform_path:
            transform_obj = load(transform_path)
        else:
            transform_obj = None
        return cls(w2vec_obj, transform_obj, **kwargs)

    def save(self, path: str):
        dump(self, path)
        logger.info(f"Saved model to {path}")