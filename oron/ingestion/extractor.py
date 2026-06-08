import spacy
from typing import List, Tuple, Dict, Any

class KGExtractor:
    """
    Knowledge Graph extractor using spaCy for NER and relation extraction.
    Pulls structured facts (subject, relation, object) from text.
    """
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Fallback/Instruction if model not found
            # In a real app, we might want to auto-download or raise error
            raise ImportError(f"spaCy model '{model_name}' not found. Run 'python -m spacy download {model_name}'")

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
