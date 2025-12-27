import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import random

from sklearn.model_selection import train_test_split

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

final_embeddings_train=torch.load("/scratch/cs21d001/action_recognition/vlm/align_final_embeddings_train_ucf101.pt")

final_embeddings_test= torch.load("/scratch/cs21d001/action_recognition/vlm/align_final_embeddings_test_ucf101.pt") 

gt_train=np.load("/scratch/cs21d001/action_recognition/vlm/hmdb_labels_train.npy")
gt_test=np.load("/scratch/cs21d001/action_recognition/vlm/hmdb_labels_test.npy")


   
def normalize_embeddings(embeddings):
    """Normalize text embeddings using L2 normalization."""
    return F.normalize(embeddings, p=2, dim=1)

with open('/scratch/cs21d001/action_recognition/vlm/hmdb_train_img_embeddings.pkl', 'rb') as f:
  img_emb_train =  pickle.load(f)
with open('/scratch/cs21d001/action_recognition/vlm/hmdb_test_img_embeddings.pkl', 'rb') as f:
   img_emb_test =  pickle.load(f)

 combined_embeddings_train = [torch.cat((img, text), dim=1) for img, text in zip(img_emb_train, final_embeddings_train)]
 combined_embeddings_train = [normalize_embeddings(embedding) for embedding in combined_embeddings_train]
 combined_embeddings_test = [torch.cat((img, text), dim=1) for img, text in zip(img_emb_test, final_embeddings_test)]
 combined_embeddings_test = [normalize_embeddings(embedding) for embedding in combined_embeddings_test]



gt_train=np.load("/scratch/cs21d001/action_recognition/vlm/labels_train.npy")
num_classes=len(np.unique(gt_train))
print(f"num of classes :{num_classes}")

def  create_label_to_data(labels):
    """Creates a mapping from label to its corresponding data indices."""
    label_to_data = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        label_to_data[label].append(idx)
    return label_to_data



# data ditribution
from collections import defaultdict
def dirichlet_client_wise(
    n,                # number of clients
    k,                # number of labels
    m,                # Dirichlet concentration
    label_to_data,    # dict {label: list of data indices}
    samples_per_client=1000,
    seed=42
):
    """
    True distributed scaling:
    - Each client gets ~samples_per_client datapoints.
    - Each client's label distribution is drawn from a Dirichlet distribution.
    - Total data grows as n increases.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    client_data_indices = defaultdict(list)

    for client_id in range(n):
        # Step 1: Draw label proportions for THIS client
        proportions = np.random.dirichlet([m] * k)  # sums to 1 over labels
        
        # Step 2: Sample data points according to those proportions
        num_samples = samples_per_client
        for label, proportion in enumerate(proportions):
            label_data = label_to_data[label]
            
            count = int(round(proportion * num_samples))
            if count > 0:
                sampled = random.sample(label_data, min(count, len(label_data)))
                client_data_indices[client_id].extend(sampled)
    
    return client_data_indices



data ,labels =  combined_embeddings_train ,gt_train 
test_data,test_labels= combined_embeddings_test ,gt_test



n = 4# Number of clients
k = num_classes # Total labels
m = 0.6 # Dirichlet concentration parameter (lower = more skewed)

# Create a sample dataset with 100 data points for each label
label_to_data = create_label_to_data(labels)

#client_data_indices = dirichlet_label_distribution(n, k, m, label_to_data)
client_data_indices = dirichlet_client_wise(n, k, m, label_to_data,1000,42)


class Client:
    def __init__(self, client_id, data, labels,batch,num_classes):
       
        self.client_id = client_id
        self.labels = labels
        self.batch_size=batch
        self.train_data=data
        self.train_labels=labels
        
        
        self.mlp = self.build_mlp(input_dim=2816, hidden_dims=[512, 256], output_dim=num_classes)

       def build_local_embedder(self,num_classes):
            return nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)), # Adjust kernel size here
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)), # Adjust kernel size here
                nn.AdaptiveAvgPool3d((1, 1, 1)), # Adaptive average pooling to get a single feature vector
                nn.Flatten(),
                nn.Linear(128,num_classes)
            )

     def build_mlp(self, input_dim, hidden_dims,output_dim):
            """Builds and returns the MLP model."""
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            model = nn.Sequential(*layers)
            return model  # Return the model instance

    def train(self, epochs=10, lr=0.001, server_model_state_dict=None):
        print(f"Training of client id : {self.client_id}")
        
        optimizer = optim.Adam(list(self.mlp.parameters()), lr=lr)
        
        loss_fn = nn.CrossEntropyLoss()
        
        def custom_collate(batch):
              frames, labels = zip(*batch)  # Unpack batch
              return list(frames), torch.tensor(labels)
        
        train_loader = DataLoader(
             list(zip(self.train_data, self.train_labels)), batch_size=self.batch_size, shuffle=True,collate_fn=custom_collate)

        # Initialize server_model with server's state_dict (for FedProx)
        if server_model_state_dict:
            server_model = self.build_mlp(input_dim=input_dim, hidden_dims=[512, 256], output_dim=num_classes)
            server_model.load_state_dict(server_model_state_dict)


        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                frames_batch, labels_batch = batch
                frames_batch = torch.cat([torch.tensor(frame, dtype=torch.float32) for frame in frames_batch])
                label_tensor = torch.tensor(labels_batch, dtype=torch.long)
                 output = self.mlp(frames_batch)
                loss_mlp = loss_fn(output, label_tensor)
                total_loss=loss_mlp
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

           
        return self.mlp # Return the updated local model



    def test(self):
        print(f"Testing Client {self.client_id} Model...")
        def custom_collate(batch):
              frames, labels = zip(*batch)  # Unpack batch
              return list(frames), torch.tensor(labels)
        test_loader = DataLoader(list(zip(self.train_data, self.train_labels)), batch_size=self.batch_size, shuffle=False,collate_fn=custom_collate)

        correct, total,correct_emb = 0,0,0
        loss_fn = nn.CrossEntropyLoss()
        total_loss,total_loss_emb = 0.0,0.0

        with torch.no_grad():
            for batch in test_loader:
                frames_batch, labels_batch = batch
                frames_batch = torch.cat([torch.tensor(frame, dtype=torch.float32) for frame in frames_batch])
                label_tensors = torch.tensor(labels_batch, dtype=torch.long)
                output = self.mlp(frames_batch)
                loss = loss_fn(output, label_tensors)
                total_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                correct += (pred == label_tensors).sum().item()
                total += len(label_tensors)

        accuracy1 = (correct / total) * 100 
        return accuracy1




# Federated Server Class
class Server:
    def __init__(self, input_dim=2816, hidden_dims=[512, 256], output_dim=num_classes):
        #self.model_state_dict = None
        #self.embedder_state_dict = None

        # Global MLP model
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.global_mlp = nn.Sequential(*layers)

      
    def aggregate_models(self, client_models):
        """Aggregate weights of both MLP and Embedder using FedAvg."""

        # Extract state_dicts for MLP and Embedder
        mlp_state_dicts = [client.state_dict() for client in client_models]
       
        # Aggregate MLP weights
        with torch.no_grad():
            global_mlp_state_dict = self.global_mlp.state_dict()
            for key in global_mlp_state_dict:
                if "num_batches_tracked" in key:  
                    continue  
                client_weights = [client_state_dict[key] for client_state_dict in mlp_state_dicts]
                if not all(w.shape == client_weights[0].shape for w in client_weights):
                    raise ValueError(f"Shape mismatch in MLP parameter '{key}' across clients.")
                global_mlp_state_dict[key] = torch.stack(client_weights).mean(dim=0)
            self.global_mlp.load_state_dict(global_mlp_state_dict)
            return self.global_mlp

    def server_test(self,global_model,test_data,test_labels,batch_size,r):
        self.batch_size=batch_size
       # print(f"Testing server model...")
        def custom_collate(batch):
              frames, labels = zip(*batch)  # Unpack batch
              return list(frames), torch.tensor(labels)
       
        test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=self.batch_size, shuffle=False,collate_fn=custom_collate)

        correct, total,correct_emb = 0,0,0
        loss_fn = nn.CrossEntropyLoss()
        total_loss,total_loss_emb = 0.0,0.0
        y_true,y_pred=[],[]
        with torch.no_grad():
            for batch in test_loader:
                frames_batch, labels_batch = batch
                frames_batch = torch.cat([torch.tensor(frame, dtype=torch.float32) for frame in frames_batch])
                
                label_tensors = torch.tensor(labels_batch, dtype=torch.long)
                output = global_model(frames_batch)
                loss = loss_fn(output, label_tensors)
                total_loss += loss.item()

                pred = torch.argmax(output, dim=1)
                y_pred.extend(pred)
                y_true.extend(label_tensors)

                correct += (pred == label_tensors).sum().item()
                total += len(label_tensors)
              
        accuracy1 = (correct / total) * 100 
       
       
        return accuracy1


       
    def distribute_model(self, clients):
        """Distribute updated models to all clients."""
        if not clients:
            print("Warning: No clients found to distribute models to.")
            return
        for client in clients:
            client.mlp.load_state_dict(self.global_mlp.state_dict())
            #client.local_embedder.load_state_dict(self.global_embedder.state_dict())

# get client data 


print("DONE")

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== IMPORTANT: keep dataset consistent ======
    # If you are using UCF101 embeddings, load UCF101 labels.
    # If you are using HMDB embeddings, load HMDB labels.
    # Change these 6 paths consistently.

    IMG_EMB_TRAIN_PKL   = "/scratch/cs21d001/action_recognition/vlm/ucf101_train_img_embeddings.pkl"
    IMG_EMB_TEST_PKL    = "/scratch/cs21d001/action_recognition/vlm/ucf101_test_img_embeddings.pkl"
    ALIGN_EMB_TRAIN_PT  = "/scratch/cs21d001/action_recognition/vlm/align_final_embeddings_train_ucf101.pt"
    ALIGN_EMB_TEST_PT   = "/scratch/cs21d001/action_recognition/vlm/align_final_embeddings_test_ucf101.pt"
    LABELS_TRAIN_NPY    = "/scratch/cs21d001/action_recognition/vlm/ucf101_labels_train.npy"
    LABELS_TEST_NPY     = "/scratch/cs21d001/action_recognition/vlm/ucf101_labels_test.npy"

    import pickle
    with open(IMG_EMB_TRAIN_PKL, "rb") as f:
        img_train_list = pickle.load(f)
    with open(IMG_EMB_TEST_PKL, "rb") as f:
        img_test_list = pickle.load(f)

    align_train_list = torch.load(ALIGN_EMB_TRAIN_PT, map_location="cpu")
    align_test_list  = torch.load(ALIGN_EMB_TEST_PT,  map_location="cpu")

    y_train = np.load(LABELS_TRAIN_NPY)
    y_test  = np.load(LABELS_TEST_NPY)

    # Convert to tensors [N, D]
    img_train  = to_2d_tensor_list(img_train_list).float()
    img_test   = to_2d_tensor_list(img_test_list).float()
    align_train = to_2d_tensor_list(align_train_list).float()
    align_test  = to_2d_tensor_list(align_test_list).float()

    # Sanity checks
    assert img_train.shape[0] == align_train.shape[0] == len(y_train), "Train N mismatch"
    assert img_test.shape[0] == align_test.shape[0] == len(y_test), "Test N mismatch"

    # Concatenate and normalize
    x_train = torch.cat([img_train, align_train], dim=1)
    x_test  = torch.cat([img_test,  align_test],  dim=1)
    x_train = l2_normalize(x_train)
    x_test  = l2_normalize(x_test)

    input_dim = x_train.shape[1]
    num_classes = int(len(np.unique(y_train)))
    print("input_dim:", input_dim, "num_classes:", num_classes)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    # Federated params
    n_clients = 10
    alpha = 0.6
    samples_per_client = 1000
    rounds = 5
    local_epochs = 1
    lr = 1e-3
    batch_size = 128

    label_to_indices = create_label_to_indices(y_train)
    client_indices = dirichlet_client_wise(
        n_clients=n_clients,
        n_classes=num_classes,
        alpha=alpha,
        label_to_indices=label_to_indices,
        samples_per_client=samples_per_client,
        seed=42
    )
clients = []
batch=128

for i in range(n):
    data_indices = client_data_indices[i]  # Get the list of indices for the current client
    client_data = [data[j] for j in data_indices]  # Use list comprehension to get data
    client_labels_data = [labels[j] for j in data_indices]  # Get corresponding labels

rounds=1
for c in [2,4,8]:
    server = Server()
    set_clients=[]
    mean_acc=[]
    mean_acc_local=[]
    client_select=np.random.choice(10,c,replace=False)
    print(f"clients seleced for c ={c} are {client_select}")
    for i in client_select:
        set_clients.append(Client(i, client_data, client_labels_data,batch,num_classes))  
    for r in range(1, rounds + 1):  # Number of communication rounds
          
            local_models = [client.train() for client in  set_clients]
            global_model=server.aggregate_models(local_models)
            mean_acc.append(server.server_test(global_model,test_data,test_labels,batch,r))
            server.distribute_model(set_clients)
            

    print(f"Average accuracy achieved after round {r+1}: {np.mean(mean_acc):.2f}%")
    print(f"Maximum accuracy achieved at round {mean_acc.index(max(mean_acc))+1}: {max(mean_acc):.2f}%")
    print(f"Minimum accuracy achieved at round {mean_acc.index(min(mean_acc))+1}: {min(mean_acc):.2f}%")
