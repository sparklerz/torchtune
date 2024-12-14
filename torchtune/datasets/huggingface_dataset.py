# torchtune/datasets/huggingface_dataset.py
from torch.utils.data import Dataset
from datasets import load_dataset

class HuggingFaceDataset(Dataset):
    def __init__(self, tokenizer, path: str, train_test_split: float, start_index: int = 0, end_index: int = None):
        # Load dataset from Hugging Face
        raw_ds = load_dataset(path)
        
        # Assume raw_ds has 'train' split. Adjust logic if needed.
        # For simplicity, we only use the training split here.
        # If 'train_test_split' is needed, manually split the data.
        data = raw_ds['train']
        
        print("Columns available: ", data.column_names)
        # Convert to list of texts (adjust the field name 'abstract' if different):
        texts = data['text']
        
        # If you need splitting:
        split_index = int(train_test_split * len(texts))
        train_texts = texts[:split_index]
        
        # Slice if end_index is provided
        if end_index is not None:
            train_texts = train_texts[start_index:end_index]

        # Tokenize
        encoded_inputs = []
        for text in train_texts:
            encoded = tokenizer.encode(text, add_bos=False, add_eos=True)
            encoded_inputs.append(encoded)

        # Truncate if needed
        max_seq_length = 1024
        for i in range(len(encoded_inputs)):
            encoded_inputs[i] = encoded_inputs[i][:max_seq_length]

        # Store in encodings dictionary
        encodings = {"tokens": encoded_inputs}
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["tokens"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Use tokens as both inputs and labels for causal LM
        item["labels"] = item["tokens"]
        return item
