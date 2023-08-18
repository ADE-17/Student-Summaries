import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel, BertConfig
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
import torch.nn as nn
import torch

# Load datasets
prompts_test = pd.read_csv("/home/woody/iwso/iwso092h/student_summaries/commonlit-evaluate-student-summaries/prompts_test.csv")
prompts_train = pd.read_csv("/home/woody/iwso/iwso092h/student_summaries/commonlit-evaluate-student-summaries/prompts_train.csv")
summaries_test = pd.read_csv("/home/woody/iwso/iwso092h/student_summaries/commonlit-evaluate-student-summaries/summaries_test.csv")
summaries_train = pd.read_csv("/home/woody/iwso/iwso092h/student_summaries/commonlit-evaluate-student-summaries/summaries_train.csv")

# Drop student_id column from summaries_train and summaries_test
summaries_train = summaries_train.drop(columns=['student_id'])
summaries_test = summaries_test.drop(columns=['student_id'])
summaries_train = summaries_train

id_mapping = {id_val: idx for idx, id_val in enumerate(prompts_train['prompt_id'].unique())}

summaries_train['prompt_id'] = summaries_train['prompt_id'].replace(id_mapping)
summaries_test['prompt_id'] = summaries_test['prompt_id'].replace(id_mapping)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the 'text' column
texts = summaries_train['text'].tolist()
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]



class CustomBERTModel(nn.Module):
    def __init__(self, config, num_prompt_classes, hidden_size=256):
        super(CustomBERTModel, self).__init__()

        self.bert = BertModel(config)

        # Classification head for prompts
        self.prompt_classifier_1 = nn.Linear(config.hidden_size, hidden_size)
        self.prompt_classifier_2 = nn.Linear(hidden_size, num_prompt_classes)

        # Regression head for wording
        self.wording_regressor_1 = nn.Linear(config.hidden_size, hidden_size)
        self.wording_regressor_2 = nn.Linear(hidden_size, 1)

        # Regression head for content
        self.content_regressor_1 = nn.Linear(config.hidden_size, hidden_size)
        self.content_regressor_2 = nn.Linear(hidden_size, 1)

        # Regression head for combined wording & content
        self.combined_regressor_1 = nn.Linear(config.hidden_size, hidden_size)
        self.combined_regressor_2 = nn.Linear(hidden_size, 2)

        # Activation and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # you can adjust the dropout rate if needed

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] representation
        
        # Classification head for prompts
        prompt_output = self.prompt_classifier_1(pooled_output)
        prompt_output = self.relu(prompt_output)
        prompt_output = self.dropout(prompt_output)
        prompt_output = self.prompt_classifier_2(prompt_output)
        
        # Regression head for wording
        wording_output = self.wording_regressor_1(pooled_output)
        wording_output = self.relu(wording_output)
        wording_output = self.dropout(wording_output)
        wording_output = self.wording_regressor_2(wording_output)
        
        # Regression head for content
        content_output = self.content_regressor_1(pooled_output)
        content_output = self.relu(content_output)
        content_output = self.dropout(content_output)
        content_output = self.content_regressor_2(content_output)
        
        # Regression head for combined wording & content
        combined_output = self.combined_regressor_1(pooled_output)
        combined_output = self.relu(combined_output)
        combined_output = self.dropout(combined_output)
        combined_output = self.combined_regressor_2(combined_output)
        
        avg_wording = wording_output + combined_output[0]
        avg_content = content_output + combined_output[1]
        return avg_wording, avg_content


# Now, update your training function
def train_model(model, input_ids, attention_mask, prompt_id_labels, wording_labels, content_labels, batch_size=32, epochs=3):
    # Define the loss functions
    classification_criterion = CrossEntropyLoss()
    regression_criterion = MSELoss()

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()

    for epoch in range(epochs):
        for batch in range(0, len(input_ids), batch_size):  # assume batch_size is the size of your batch
            optimizer.zero_grad()
            
            # Forward pass
            wording, content = model(input_ids[batch:batch+batch_size], attention_mask=attention_mask[batch:batch+batch_size])

            # Compute loss

            loss1 = regression_criterion(wording, wording_labels[batch:batch+batch_size])
            loss2 = regression_criterion(content, content_labels[batch:batch+batch_size])

            # Total loss
            loss = (loss1 + loss2) / 2
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Batch {batch//batch_size + 1}, Loss: {loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), "/home/woody/iwso/iwso092h/student_summaries/model_weights.pth")

prompt_id_labels = torch.tensor(summaries_train['prompt_id'].values)
wording_labels = torch.tensor(summaries_train['wording'].values).float().unsqueeze(1)
content_labels = torch.tensor(summaries_train['content'].values).float().unsqueeze(1)

import os

# Set the hyperparameters
batch_size = 32  # or whatever you choose
num_prompt_classes = 4  # replace with the actual number of classes for prompt classification

# Instantiate model with BERT's configuration
config = BertConfig.from_pretrained("bert-base-uncased")
model = CustomBERTModel(config, num_prompt_classes)

# Path to the saved model weights
model_weights_path = "./saved_model_directory/model_weights.pth"

# Instantiate model with BERT's configuration
config = BertConfig.from_pretrained("bert-base-uncased")
model = CustomBERTModel(config, num_prompt_classes)

# Check if the model weights exist and load them
if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path))
    print("Loaded saved model weights!")

# If no saved model is found, train from scratch
train_model(
    model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    prompt_id_labels=prompt_id_labels,
    wording_labels=wording_labels,
    content_labels=content_labels,
    batch_size=batch_size,
    epochs=1
)