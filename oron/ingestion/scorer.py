import re

class ImportanceScorer:
    """
    Stricter importance scorer to filter out noise and gibberish.
    """
    def __init__(self):
        self.noise_patterns = [
            r"^(ok|thanks|thank you|cool|great|yes|no|maybe|yep|nope|yo|hi|hello|ahoy|hola|ciao|bih|sexo)$",
            r"^[^\w\s]+$", # only punctuation
            r"^\d+$",     # only numbers
        ]
        self.important_keywords = [
            "prefer", "dislike", "hate", "love", "work", "live", "using", 
            "project", "stack", "goal", "should", "must", "always", "never",
            "name", "called", "born", "favorite", "allergic", "setup", "bank"
        ]

    def _is_gibberish(self, text: str) -> bool:
        # Very naive gibberish check: low vowel count or extremely short words
        words = text.split()
        if not words:
            return True
            
        for word in words:
            if len(word) > 15: # suspiciously long
                return True
            
            # Count vowels
            vowels = len(re.findall(r'[aeiouy]', word.lower()))
            if len(word) > 3 and vowels == 0:
                return True
                
        return False

    def score(self, text: str) -> float:
        text = text.lower().strip()
        
        if not text:
            return 0.0
            
        # Check for noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, text):
                return 0.1
        
        if self._is_gibberish(text):
            return 0.1
            
        score = 0.2 # low default score
        
        # Length heuristic - meaningful content usually has some length
        if len(text) > 15:
            score += 0.2
        if len(text) > 40:
            score += 0.2
        
        # Keyword heuristic
        for kw in self.important_keywords:
            if kw in text:
                score += 0.4 # significant boost for keywords
                break 
        
        # Named entity signals (crude check for Capitalized words in original text)
        # This would be better with spaCy, but let's keep it lightweight here.
        
        return min(score, 1.0)

    def is_important(self, text: str, threshold: float = 0.3) -> bool:
        return self.score(text) >= threshold
