import nltk
from nltk import word_tokenize, pos_tag
import sys

def test_nltk_setup():
    try:
        # Test sentence
        text = "Testing my text to speech setup. This is a simple test."
        
        # Try tokenizing and POS tagging
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        print("NLTK Test Results:")
        print("-----------------")
        print("Tokenization test:")
        print(tokens[:5])
        print("\nPOS tagging test:")
        print(tagged[:5])
        
        print("\nSuccess! NLTK is properly set up.")
        
    except LookupError as e:
        print(f"\nError: Missing NLTK resource. Try running these commands:")
        print("nltk.download('punkt')")
        print("nltk.download('averaged_perceptron_tagger')")
        print(f"\nFull error: {str(e)}")
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print(f"Python version: {sys.version}")

if __name__ == "__main__":
    print("Starting NLTK test...")
    
    # First make sure we have required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptual_tagger')
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
    
    # Run the test
    test_nltk_setup()