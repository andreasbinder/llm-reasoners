import json
import argparse
from abc import ABC, abstractmethod
# Import specific libraries for KeyBERT, spaCy, etc.

class KeywordExtractor(ABC):
    @abstractmethod
    def extract_keywords(self, text):
        pass

class KeyBERTExtractor(KeywordExtractor):

    def __init__(self) -> None:
        super().__init__()
        from keybert import KeyBERT
        self.kw_model = KeyBERT()

    def extract_keywords(self, text):
        # Implementation for KeyBERT
        # doc = data[key]["prediction"]
        # keywords_b = self.kw_model.extract_keywords(doc, 
        #                                     keyphrase_ngram_range=(1, 2), 
        #                                     stop_words='english', 
        #                                     top_n=4,
        #                                     #    use_mmr=True, 
        # #                                     #    diversity=0.7
        #             )
        keywords = []
        return keywords

class SpacyExtractor(KeywordExtractor):
    def __init__(self) -> None:
        super().__init__()
        import spacy

        # Load the SpaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, text):
        # Implementation for spaCy
        doc =self. nlp(text)

        # Extracting keywords (nouns and proper nouns in this case)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

        return keywords

class KeywordFiller:
    def __init__(self, extractor: KeywordExtractor):
        self.extractor = extractor

    def process_file(self, file_path):
        # Load JSON
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Process each entry
        for key in data:
            keywords = self.extractor.extract_keywords(data[key]['A'][0])
            data[key]['Keywords_A'] = keywords

        # Save updated JSON
        with open(file_path.replace(".json", "_{}.json".format('Keywords_A')), 'w') as file:
            json.dump(data, file, indent=4)

# Usage
# extractor = KeyBERTExtractor()  # Or SpacyExtractor(), etc.
# keyword_filler = KeywordFiller(extractor)
# keyword_filler.process_file('your_file.json')

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process a JSON file to fill in keywords.")
    parser.add_argument("--file", type=str, help="Path to the JSON file")
    parser.add_argument("--extractor", choices=['keybert', 'spacy'], help="Keyword extraction method", default='spacy')

    # Parse arguments
    args = parser.parse_args()

    # Select the keyword extractor
    if args.extractor == "keybert":
        extractor = KeyBERTExtractor()
    elif args.extractor == "spacy":
        extractor = SpacyExtractor()
    else:
        raise ValueError("Unsupported extractor method.")

    # Process the file
    keyword_filler = KeywordFiller(extractor)
    keyword_filler.process_file(args.file)

if __name__ == "__main__":
    main()
