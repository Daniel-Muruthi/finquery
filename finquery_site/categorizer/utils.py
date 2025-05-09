import os
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, 'bert_intent_classifier.pth')
TOKENIZER_NAME = 'bert-base-uncased'

# Cache for model, tokenizer, and id2label mapping
_model = None
_tokenizer = None
_id2label = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _load_model():
    """
    Load the BERT model, tokenizer, and label mapping.
    Use cached versions if already loaded.
    """
    global _model, _tokenizer, _id2label, _device
    
    # Only load if not already in memory
    if _model is None:
        # Load tokenizer
        _tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
        
        # Load the saved model and mappings
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        
        # Get the label mappings
        _id2label = checkpoint['id2label']
        label2id = checkpoint['label2id']
        
        # Initialize model with the correct number of labels
        _model = BertForSequenceClassification.from_pretrained(
            TOKENIZER_NAME,
            num_labels=len(_id2label),
            id2label=_id2label,
            label2id=label2id
        )
        
        # Load the fine-tuned weights
        _model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to appropriate device (GPU if available)
        _model.to(_device)
        
        # Set model to evaluation mode
        _model.eval()

def categorize_intent(text: str) -> str:
    """
    1. Clean text (lower + collapse whitespace)
    2. Tokenize with BERT tokenizer
    3. Run inference with BERT model
    4. Return predicted category
    """
    # Ensure model is loaded
    if _model is None:
        _load_model()
    
    # Clean text (same as before)
    cleaned = re.sub(r'\s+', ' ', text.lower()).strip()
    
    # Tokenize and prepare for model
    inputs = _tokenizer(
        cleaned,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    ).to(_device)
    
    # Run model inference
    with torch.no_grad():
        outputs = _model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    
    # Map numeric prediction to category label
    predicted_category = _id2label[predicted_class_id]
    
    return predicted_category