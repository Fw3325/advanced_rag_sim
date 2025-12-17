"""
Entity extraction from text chunks.
Supports multiple methods: spaCy NER, LLM-based extraction, and simple regex fallback.
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from langchain_core.documents import Document

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class EntityExtractor:
    """
    Extracts entities and relationships from text.
    """
    
    def __init__(self, method: str = "spacy", llm=None):
        """
        Args:
            method: "spacy", "llm", or "simple"
            llm: LangChain LLM instance (required for "llm" method)
        """
        self.method = method
        self.llm = llm
        self.nlp = None
        
        if method == "spacy":
            if SPACY_AVAILABLE:
                try:
                    # Try to load English model
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                    print("   Falling back to simple extraction method.")
                    self.method = "simple"
            else:
                print("⚠️  spaCy not installed. Install with: pip install spacy")
                print("   Falling back to simple extraction method.")
                self.method = "simple"
    
    def extract_entities(self, text: str, chunk_key: str = None) -> Set[str]:
        """
        Extract entity names from text.
        Returns a set of unique entity names.
        """
        if self.method == "spacy" and self.nlp:
            return self._extract_spacy_entities(text)
        elif self.method == "llm" and self.llm:
            return self._extract_llm_entities(text)
        else:
            return self._extract_simple_entities(text)
    
    def extract_relations(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities.
        Returns list of (head, relation, tail) triples.
        """
        if self.method == "llm" and self.llm:
            return self._extract_llm_relations(text, entities)
        else:
            # Simple co-occurrence based relations
            return self._extract_simple_relations(text, entities)
    
    def _extract_spacy_entities(self, text: str) -> Set[str]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = set()
        
        # Extract named entities (PERSON, ORG, GPE, etc.)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.add(ent.text.strip())
        
        # Also extract noun phrases that might be important concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short noun phrases
                text_lower = chunk.text.lower().strip()
                # Filter out common stop words
                if text_lower not in ["the", "a", "an", "this", "that", "these", "those"]:
                    entities.add(chunk.text.strip())
        
        return entities
    
    def _extract_llm_entities(self, text: str) -> Set[str]:
        """Extract entities using LLM."""
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        
        system = SystemMessage(content="""
        Extract important entities (people, organizations, concepts, products, nutrients, supplements, exercises, etc.) 
        from the following text. Return ONLY a JSON list of entity names as strings.
        Example: ["Vitamin D", "protein", "muscle growth", "creatine"]
        """)
        
        user_msg = HumanMessage(content=f"Text: {text[:2000]}")  # Limit text length
        
        try:
            response = self.llm.invoke([system, user_msg]).content.strip()
            cleaned = response.replace("```json", "").replace("```", "").strip()
            entities = json.loads(cleaned)
            return set(entities) if isinstance(entities, list) else {entities}
        except Exception as e:
            print(f"⚠️  LLM entity extraction failed: {e}. Falling back to simple method.")
            return self._extract_simple_entities(text)
    
    def _extract_simple_entities(self, text: str) -> Set[str]:
        """Simple regex-based entity extraction (fallback)."""
        entities = set()
        
        # Extract capitalized phrases (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, text)
        entities.update(matches)
        
        # Extract common health/nutrition terms (case-insensitive)
        health_terms = [
            r'\b(?:vitamin|vitamins)\s+[A-Z]?\w+\b',
            r'\b(?:protein|proteins|carbohydrate|carbohydrates|fat|fats)\b',
            r'\b(?:supplement|supplements|creatine|whey|casein)\b',
            r'\b(?:exercise|exercises|workout|training)\b',
            r'\b(?:muscle|muscles|strength|endurance)\b',
        ]
        
        for pattern in health_terms:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update([m.strip() for m in matches])
        
        # Filter out very short or common words
        filtered = {e for e in entities if len(e) > 2 and e.lower() not in ["the", "and", "for", "are", "but"]}
        
        return filtered
    
    def _extract_llm_relations(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships using LLM."""
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        
        if len(entities) < 2:
            return []
        
        entities_list = list(entities)[:10]  # Limit to avoid token overflow
        
        system = SystemMessage(content="""
        Extract relationships between entities in the form of (head, relation, tail) triples.
        Return ONLY a JSON list of lists: [["head", "relation", "tail"], ...]
        Example: [["Vitamin D", "helps_with", "bone health"], ["protein", "builds", "muscle"]]
        """)
        
        user_msg = HumanMessage(content=f"Entities: {entities_list}\n\nText: {text[:1500]}")
        
        try:
            response = self.llm.invoke([system, user_msg]).content.strip()
            cleaned = response.replace("```json", "").replace("```", "").strip()
            relations = json.loads(cleaned)
            
            if isinstance(relations, list):
                return [tuple(r) for r in relations if len(r) == 3]
            return []
        except Exception as e:
            print(f"⚠️  LLM relation extraction failed: {e}")
            return []
    
    def _extract_simple_relations(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Simple co-occurrence based relations."""
        relations = []
        entity_list = list(entities)
        
        # If entities appear in the same sentence, create a generic relation
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            found_entities = [e for e in entity_list if e.lower() in sentence.lower()]
            if len(found_entities) >= 2:
                # Create pairwise relations
                for i in range(len(found_entities)):
                    for j in range(i + 1, len(found_entities)):
                        relations.append((found_entities[i], "related_to", found_entities[j]))
        
        return relations[:20]  # Limit relations

