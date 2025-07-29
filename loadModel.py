# Reload model
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained("my_fine_tuned_model")
tokenizer = BertTokenizer.from_pretrained("my_fine_tuned_model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This movie was amazing!")
print(result)