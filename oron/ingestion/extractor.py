import spacy
from typing import List, Tuple


class KGExtractor:
    """
    Knowledge Graph extractor using spaCy for NER and relation extraction.
    Pulls structured facts (subject, relation, object) from text.
    Lazy-loads the spaCy model on first use — safe to instantiate without the model present.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise ImportError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Run: python -m spacy download {self.model_name}"
                )
        return self._nlp

    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        doc = self.nlp(text)
        facts = []

        # Simple rule-based extraction for demonstration
        # In a more advanced version, use dependency parsing
        for sent in doc.sents:
            # Look for SVO patterns
            subj = ""
            verb = ""
            obj = ""

            for token in sent:
                if "subj" in token.dep_:
                    subj = token.text
                if token.pos_ == "VERB":
                    verb = token.lemma_
                if "obj" in token.dep_:
                    obj = token.text

            if subj and verb and obj:
                facts.append((subj, verb, obj))

        # Also extract entities as potential nodes
        for ent in doc.ents:
            # We could do something with entities here
            pass

        return facts
