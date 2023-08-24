import numpy as np
import pandas as pd
data = pd.read_csv('../commonlit-evaluate-student-summaries/summaries_train.csv')
import pandas as pd
import torch
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import BertTokenizer, BertModel
# from transformers import ElectraTokenizer, ElectraModel
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for the progress bar
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

max_seq_length = 512

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True)

data['text_tokens'] = data['text'].apply(tokenize_text)

roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

def generate_roberta_embeddings(text_tokens):
    embeddings_list = []
    with tqdm(total=len(text_tokens)) as pbar:
        with torch.no_grad():
            for tokens in text_tokens:
                outputs = roberta_model(tokens.unsqueeze(0))  # Unsqueeze to add batch dimension
                embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings for [CLS] token
                embeddings_list.append(embeddings)
                pbar.update(1)
        embeddings_tensor = torch.cat(embeddings_list, dim=0)
        return embeddings_tensor
    
X = data['text_tokens'].tolist()
X = torch.tensor(X).to(device)

X_embeddings = generate_roberta_embeddings(X)

embeddings_np = X_embeddings.cpu().numpy()

np.save("/home/woody/iwso/iwso092h/student_summaries/saved_data/roberta_embeddings.npy", embeddings_np)