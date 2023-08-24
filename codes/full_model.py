import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for the progress bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV data
data = pd.read_csv('/home/woody/iwso/iwso092h/student_summaries/commonlit-evaluate-student-summaries/summaries_train.csv')

# Preprocess the text data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

max_seq_length = 512

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True)

data['text_tokens'] = data['text'].apply(tokenize_text)

# Split the data into features and targets
X = data['text_tokens'].tolist()
y = data[['wording', 'content']].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train).to(device)
X_test = torch.tensor(X_test).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

base_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# base_model = ElectraModel.from_pretrained('google/electra-base-discriminator').to(device)

def generate_bert_embeddings(text_tokens):
    embeddings_list = []
    with tqdm(total=len(text_tokens)) as pbar:
        with torch.no_grad():
            for tokens in text_tokens:
                outputs = base_model(tokens.unsqueeze(0))  # Unsqueeze to add batch dimension
                embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings for [CLS] token
                embeddings_list.append(embeddings)
                pbar.update(1)
        embeddings_tensor = torch.cat(embeddings_list, dim=0)
        return embeddings_tensor

X_train_embeddings = generate_bert_embeddings(X_train)
X_test_embeddings = generate_bert_embeddings(X_test)

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def mean_columnwise_rmse(y_pred, y_true):
    columnwise_rmse = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=0))
    return torch.mean(columnwise_rmse)

# Define the model
input_dim = base_model.config.hidden_size
hidden_dim = 256  # Adjust the hidden layer dimension as needed
output_dim = 2  # Number of target variables
model = RegressionModel(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
batch_size = 64

train_dataset = TensorDataset(X_train_embeddings, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_loss1_list = []
train_loss2_list = []
test_loss1_list = []
test_loss2_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss_wording = 0
    total_loss_content = 0
    total_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_wording = criterion(outputs[:, 0], targets[:, 0])
        loss_content = criterion(outputs[:, 1], targets[:, 1])

        loss = (loss_wording + loss_content) / 2
        loss.backward()
        optimizer.step()
        
        total_loss_wording += loss_wording.item()
        total_loss_content += loss_content.item()

        total_loss += loss.item()
        
    train_loss1_list.append(total_loss_wording / len(train_loader))
    train_loss2_list.append(total_loss_content / len(train_loader))
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Wording Loss: {total_loss_wording:.4f}, Content Loss: {total_loss_content:.4f}, Loss: {total_loss:.4f}')
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_embeddings)
        test_loss1 = criterion(test_outputs[:, 0], y_test[:, 0])
        test_loss2 = criterion(test_outputs[:, 1], y_test[:, 1])
        
    test_loss1_list.append(test_loss1.item())
    test_loss2_list.append(test_loss2.item())
    
    print(f'Test Loss Wording: {test_loss1_list[-1]:.4f}, Test Loss Content: {test_loss2_list[-1]:.4f}')


print(len(train_loss1_list))
print(len(train_loss2_list))
print(len(test_loss1_list))
print(len(test_loss2_list))

loss_data = {
    'Epoch': list(range(1, num_epochs+1)),
    'Train_Loss_Wording': train_loss1_list,
    'Train_Loss_Content': train_loss2_list,
    'Test_Loss_Wording': test_loss1_list,
    'Test_Loss_Content': test_loss2_list
}

loss_df = pd.DataFrame(loss_data)
loss_df.to_csv('/home/woody/iwso/iwso092h/student_summaries/losses.csv', index=False)