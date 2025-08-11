import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, predictions),
        'f1': f1_score(p.label_ids, predictions),
    }

data=pd.read_csv("D:\Code\Sentiment Analysis\Data\IMDB Dataset.csv")

def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    return text

data['review'] = data['review'].apply(clean_text)

label_mapping={'negative':0,'positive':1}
data['label']=data['sentiment'].map(label_mapping)

from transformers import RobertaTokenizer

tokenizer=RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(text):
    return tokenizer(text,padding='max_length',truncation=True,max_length=256)

data['tokenized']=data['review'].apply(tokenize_function)

import torch

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item
    
from sklearn.model_selection import train_test_split

train_texts, temp_texts, train_labels, temp_labels = train_test_split(data['tokenized'].to_list(), data['label'].to_list(), test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

def convert_to_dicts(tokenized_texts):
    input_ids = [d['input_ids'] for d in tokenized_texts]
    attention_masks = [d['attention_mask'] for d in tokenized_texts]
    return {'input_ids': input_ids, 'attention_mask': attention_masks}

train_encodings = convert_to_dicts(train_texts)
val_encodings = convert_to_dicts(val_texts)
test_encodings = convert_to_dicts(test_texts)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

from transformers import RobertaForSequenceClassification,Trainer,TrainingArguments

model=RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=2)

training_args = TrainingArguments(
    output_dir='./results', # All files generated during training will be stored here
    num_train_epochs=3, # The model will be trained for 3 full epochs unless the step limit (max_steps) is reached first
    per_device_train_batch_size=10, # Training batch size per device (GPU or CPU).
    per_device_eval_batch_size=10, # Evaluation batch size per device (GPU or CPU).
    warmup_steps=10, # Number of warm-up steps during which the learning rate gradually increases to its initial value
    weight_decay=0.01, # Weight decay rate: this technique helps to avoid overfitting, penalizing large weights in the neural network
    logging_dir='./logs', # Directory where training logs will be stored
    save_steps=2000,  # Range of steps after which the model will be saved
    logging_steps=500,  # Range of steps after which log information will be recorded
)

trainer = Trainer(
    model=model, # The pre-trained model that you want to fine-tune or train
    args=training_args, # The training arguments that specify the configurations for the training process
    train_dataset=train_dataset, # The dataset used for training the model
    eval_dataset=val_dataset, # The dataset used for evaluating the model during training
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Evaluate the Model
results = trainer.evaluate(test_dataset)

print("Evaluation Results:")
print(f"  - Loss: {results['eval_loss']:.4f}")
print(f"  - Runtime: {results['eval_runtime']:.2f} seconds")
print(f"  - Samples per Second: {results['eval_samples_per_second']:.2f}")
print(f"  - Steps per Second: {results['eval_steps_per_second']:.2f}")
print(f"  - Epoch: {results['epoch']:.4f}")
print(f" - Accuracy: {results['eval_accuracy']:.4f}")
print(f" - F1 Score: {results['eval_f1']:.4f}")

# Save the model and tokenizer in the current folder
model_save_path = "./"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)