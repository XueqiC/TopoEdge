import pandas as pd
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset as BaseDataset
import os
import requests
import tarfile
import zipfile

def extract_zip(input_filename, dataset_name):
    # Destination folder path
    destination_folder = f'./dataset/{dataset_name}'

    # Check if the destination folder exists. If not, create it.
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with zipfile.ZipFile(input_filename, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    print(f"Extracted {input_filename} to {destination_folder}")

def download_file(url, path):
    """Download file from the given URL and save it to the specified path."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            print(f"Failed to download the file from {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")
        return False

def extract_tar_gz(input_filename):
    # Check if the destination folder exists. If not, create it.
    if not os.path.exists('./dataset/epinions'):
        os.makedirs('./dataset/epinions')

    with tarfile.open(input_filename, "r:gz") as tar:
        tar.extractall(path='./dataset/epinions')
    print(f"Extracted {input_filename} to ./dataset/epinions")
        
def get_data(args):
    # Ensure the dataset directory exists
    os.makedirs('./dataset', exist_ok=True)

    # Mapping datasets to their URLs, due to the anonymous requirement, will make public when the paper is published, for now, please see the link in readme.md for dataset:
    dataset_urls = {xxx}

    file_extension = ".zip" if args.dataset in ["epinions", "reddit", "ppi", "mag"] else ".csv"
    file_path = f"./dataset/{args.dataset}{file_extension}"

    # Check if the dataset already exists
    if os.path.exists(file_path) or (args.dataset != 'bitcoin_alpha' and args.dataset != 'Intrusion' and os.path.exists(f'./dataset/{args.dataset}')):
        return

    print(f'Downloading dataset {args.dataset}{file_extension}')

    # Download the file
    success = download_file(dataset_urls[args.dataset], file_path)

    if success:
        print(f"File downloaded successfully to {file_path}")
    else:
        print(f"Failed to download {args.dataset}{file_extension}")

    # Extract ZIP files
    if file_extension == ".zip":
        extract_zip(file_path, args.dataset)
        os.remove(file_path)
        

def load_dataset(args, encoder):
    get_data(args)
    dataset = args.dataset
    
    if dataset in ['bitcoin_alpha']:
        csv_file = './dataset/{}.csv'.format(dataset)
        data = pd.read_csv(csv_file)
        n_nodes = len(set(data['source'].tolist() + data['target'].tolist()))
        n_edges = len(data)

        edge_index = torch.tensor([data['source'].tolist(), \
                                data['target'].tolist()], dtype=torch.long)
        

        edge_attr = torch.tensor(encoder.encode(data['comment'].fillna('').tolist()))

        if args.task == 0:
            y = torch.where(torch.tensor(data['rating']) > 0, \
                                        torch.tensor([0]), torch.tensor([1])).view(-1)
        elif args.task == 1:
            y = torch.tensor(data['rating'].apply(lambda x: 0 if 1 <= x <= 7 \
                                                            else 1 if 8 <= x <= 10 \
                                                            else 2 if -7 <= x <= -1 else 3).tolist()\
                                                            , dtype = torch.long).view(-1)
            
        datag = Data(edge_index = edge_index, edge_attr = edge_attr, \
                num_nodes = n_nodes, num_edges = n_edges, y = y)
    
    elif dataset == 'intrusion':
        csv_file = './dataset/{}.csv'.format(dataset)
        data = pd.read_csv(csv_file)
        data['Label'] = pd.Categorical(data['Attack']).codes
        n_nodes = len(set(data['src_id'].tolist() + data['dest_id'].tolist()))
        n_edges=len(data)

        edge_index = torch.tensor([data['src_id'].tolist(), \
                                        data['dest_id'].tolist()],dtype=torch.long)
        
        edge_fea = data.iloc[:, data.columns.get_loc('Label') + 1:]
        edge_attr = torch.tensor(edge_fea.values, dtype=torch.float)
        
        y = torch.tensor(data['Label'].tolist(), dtype=torch.long).view(-1)
        # if args.task == 1:
        #     y = torch.tensor(data['Attack'].apply(lambda x: 1 if x == 'Benign' else (0 if x in ['injection', 'ddos'] else 2)).tolist(), dtype=torch.long).view(-1)
            
        datag = Data(edge_index = edge_index, edge_attr = edge_attr, \
                num_nodes = n_nodes, num_edges = n_edges, y = y)
    
    elif dataset == 'ppi':
        datag = torch.load("./dataset/ppi/PPI/ppi.pth", map_location="cpu")
            
    elif dataset == 'epinions':
        datag = torch.load("./dataset/epinions/epinions/pyg_data.pth", map_location="cpu")
        datag.y[datag.y == 1] = 0
        datag.y[datag.y == -1] = 1
        
    elif dataset == 'reddit':
        datag = torch.load("./dataset/reddit/reddit/reddit.pth", map_location="cpu")
        
    elif dataset == 'mag':
        datag = torch.load("./dataset/mag/MAG/mag.pth", map_location="cpu")
    
    return datag

def split_edge(data, args):
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    label_ratio = args.label_ratio
    num_edges = data.num_edges
    
    # Create a permutation of all edges
    torch.manual_seed(1028)
    perm = torch.randperm(num_edges)
    
    # Split indices
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    
    train_idx = perm[:int(train_size*label_ratio)]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


class EdgeDataset(BaseDataset):
    def __init__(self, idxs):
        self.idxs = idxs

    def __len__(self):
        return self.idxs.shape[0]

    def _get_feed_dict(self, idx):
        return [self.idxs[idx], idx]
    
    def __getitem__(self, idx):
        return self._get_feed_dict(idx)
    
    def collate_fn(self, feed_dicts):
        return torch.tensor([_[0] for _ in feed_dicts]), torch.tensor([_[1] for _ in feed_dicts])
