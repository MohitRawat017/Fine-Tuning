"""
Inference script for Tsuzi Intent Classification
"""

import torch
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel

# Label mapping
ID_TO_LABEL = {
    0: "casual",
    1: "productivity", 
    2: "system",
    3: "research",
    4: "communication"
}

LABEL_TO_ACTIONS = {
    "casual": [],
    "productivity": ["set_timer", "set_alarm", "create_calendar_event", "add_task", "get_tasks"],
    "system": ["open_app", "run_command", "get_system_info"],
    "research": ["web_search", "search_stackoverflow", "search_arxiv"],
    "communication": ["send_email", "read_emails"]
}


class MiniLMClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super(MiniLMClassifier, self).__init__()
        self.minilm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(384, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.minilm(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class IntentClassifier:
    def __init__(self, model_path="tsuzi_intent_model.pt", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load model
        self.model = MiniLMClassifier(num_classes=5)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, return_confidence=True):
        """Predict intent for a single text"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
        
        label = ID_TO_LABEL[pred_id]
        actions = LABEL_TO_ACTIONS[label]
        
        result = {
            "text": text,
            "intent": label,
            "label_id": pred_id,
            "available_actions": actions
        }
        
        if return_confidence:
            result["confidence"] = round(confidence, 4)
            # Add all probabilities
            result["all_probs"] = {
                ID_TO_LABEL[i]: round(probs[0][i].item(), 4) 
                for i in range(5)
            }
        
        return result


def main():
    # Initialize classifier
    print("Loading model...")
    classifier = IntentClassifier()
    
    # Test examples
    test_queries = [
        "hey um can you set an alarm for 7am",
        "open vscode please",
        "search stackoverflow for python error",
        "check my emails",
        "thanks a lot",
        "hey what's up",
        "start a timer for 10 minutes",
        "run pip install requests",
        "find papers on machine learning",
        "send email to john"
    ]
    
    print("\n" + "=" * 60)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 60)
    
    for query in test_queries:
        result = classifier.predict(query)
        print(f"\nQuery: \"{query}\"")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']})")
        print(f"  Available actions: {result['available_actions']}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        result = classifier.predict(query)
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']})")
        print(f"  Available actions: {result['available_actions']}")


if __name__ == "__main__":
    main()
