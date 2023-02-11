import os
import requests
import tiktoken
import numpy as np

# Baixar o conjunto de dados pequeno de Shakespeare
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

# Lê o arquivo em input_file_path como um arquivo de texto e armazena o conteúdo na variável "data".
with open(input_file_path, 'r') as f:
    data = f.read()
# Armazena o comprimento total de caracteres de "data" na variável "n".
n = len(data)
# Armazena os primeiros 90% dos caracteres de "data" na variável "train_data".
train_data = data[:int(n*0.9)]
# Armazena os últimos 10% dos caracteres de "data" na variável "val_data".
val_data = data[int(n*0.9):]

# codificar com tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"treinamento tem {len(train_ids):,} tokens")
print(f"validação tem {len(val_ids):,} tokens")

# exportar para arquivos binários
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin tem 301.966 tokens
# val.bin tem 36.059 tokens
