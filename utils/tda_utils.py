"""
TDA utilities for building persistence diagrams from sentence-level word vectors.

This module provides a TDAProcessor class with a fit/transform API so that
components (FastText, PCA, StandardScaler, Landscape transforms) are fit
on training data only and then used to transform test data (no leakage).

Functions / classes:
- TDAProcessor: .fit(texts_train), .transform(texts)
- simple helper sentence_diagram (used internally)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gudhi.representations import DiagramSelector, DiagramScaler, Clamping, Landscape
import gudhi
from gensim.models import FastText
import nltk
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist

# Ensure NLTK tokenizer is available
nltk.download("punkt_tab", quiet=True)


def sentence_diagram(sentence, model, vector_size=100, quantile=0.9):
    """
    Compute a persistence diagram for a single sentence using the provided FastText model.
    Returns an array of (birth, death) pairs or an empty array if none.
    """
    words = sentence.split()
    embeddings = [model.wv[w] for w in words if w in model.wv]

    if len(embeddings) == 1:
        embeddings.append(np.zeros(vector_size))
    if not embeddings:
        return np.zeros((0, 2))

    embeddings = np.array(embeddings)
    if embeddings.shape[0] > 1:
        dists = pdist(embeddings)
        max_edge_length = np.quantile(dists, quantile) if dists.size > 0 else 0.1
    else:
        max_edge_length = 0.1

    rips = gudhi.RipsComplex(points=embeddings, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence(homology_coeff_field=2, min_persistence=-1)
    return np.array([p[1] for p in diag])


class TDAProcessor:
    """
    Fit-transform TDA pipeline:
    - Train FastText on training texts
    - Build persistence diagrams for texts
    - Convert diagrams to landscapes
    - Fit PCA and StandardScaler on landscapes (train), then transform (train/test)
    """

    def __init__(self, fasttext_dim=100, pca_dim=100, resolution=200, workers=4):
        self.fasttext_dim = fasttext_dim
        self.pca_dim = pca_dim
        self.resolution = resolution
        self.workers = workers

        # placeholders for fitted objects
        self.ft_model = None
        self.pca = None
        self.scaler = None
        self.LS = Landscape(resolution=self.resolution)

        # Gudhi processors (stateless)
        self.proc1 = DiagramSelector(use=True, point_type="finite")
        # we will apply DiagramScaler transformations in-line as needed
        self.clamp = Clamping(maximum=0.9)

    def _diagrams_from_texts(self, texts):
        # compute diagrams for each sentence
        diagrams = [sentence_diagram(s, self.ft_model, vector_size=self.fasttext_dim) for s in texts]
        return diagrams

    def _landscapes_from_diagrams(self, diagrams):
        landscapes = []
        for diag in diagrams:
            # Apply selection and scaling/clamping using Gudhi pipeline
            try:
                D1 = self.proc1(diag)
                # apply clamping to death coordinate (index 1)
                D1 = DiagramScaler(use=True, scalers=[([1], self.clamp)])(D1)
                L = self.LS(D1)
            except Exception:
                # fallback to zero-vector if something goes wrong
                L = np.zeros(self.resolution)
            landscapes.append(L)
        return np.array(landscapes)

    def fit(self, texts_train):
        """
        Fit the processor on the training texts.
        """
        # Tokenize and train FastText
        tokenized = [word_tokenize(t) for t in texts_train]
        self.ft_model = FastText(sentences=tokenized,
                                 vector_size=self.fasttext_dim,
                                 window=5,
                                 min_count=1,
                                 workers=self.workers)

        # Get diagrams and landscapes for train
        diagrams_train = self._diagrams_from_texts(texts_train)
        landscapes_train = self._landscapes_from_diagrams(diagrams_train)

        # PCA: choose n_components <= n_features
        n_components = min(self.pca_dim, landscapes_train.shape[1])
        self.pca = PCA(n_components=n_components)
        tda_pca = self.pca.fit_transform(landscapes_train)

        # Standard scaling on PCA outputs
        self.scaler = StandardScaler()
        tda_scaled = self.scaler.fit_transform(tda_pca)

        return tda_scaled  # returns train features

    def transform(self, texts):
        """
        Transform given texts using the already-fitted pipeline.
        """
        if self.ft_model is None or self.pca is None or self.scaler is None:
            raise RuntimeError("TDAProcessor not fitted. Call .fit(train_texts) first.")

        diagrams = self._diagrams_from_texts(texts)
        landscapes = self._landscapes_from_diagrams(diagrams)

        # Handle degenerate case (single sample)
        if landscapes.ndim == 1:
            landscapes = landscapes.reshape(1, -1)

        # If PCA was fitted with fewer features than current landscapes, ensure shape is compatible
        n_components = self.pca.n_components_
        # If landscapes have fewer columns than expected, pad zeros (rare)
        if landscapes.shape[1] < self.pca.components_.shape[1]:
            pad_width = self.pca.components_.shape[1] - landscapes.shape[1]
            landscapes = np.pad(landscapes, ((0, 0), (0, pad_width)), mode="constant")

        tda_pca = self.pca.transform(landscapes)
        tda_scaled = self.scaler.transform(tda_pca)
        return tda_scaled
