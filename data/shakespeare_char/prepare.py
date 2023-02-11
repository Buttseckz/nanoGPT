# Prepara o conjunto de dados de Shakespeare para modelagem de linguagem de nível de caractere.
# Em vez de codificar com tokens BPE GPT-2, nós apenas mapeamos caracteres para inteiros.
# Salvará train.bin, val.bin contendo os ids e meta.pkl contendo o codificador e decodificador e outras informações relacionadas.

import os
import pickle
import requests
import numpy as np

# faz o download do conjunto de dados tiny shakespeare
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"comprimento do conjunto de dados em caracteres: {len(data):,}")

# obtém todos os caracteres únicos que ocorrem neste texto
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("todos os caracteres únicos:", ''.join(chars))
print(f"tamanho do vocabulário: {vocab_size:,}")

# cria um mapeamento de caracteres para inteiros
#Criar um mapeamento de personagens para inteiros usando uma compreensão de dicionário
stoi = {ch: i for i, ch in enumerate(chars)}
#Criar um mapeamento de inteiros para caracteres usando outra compreensão de dicionário
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s] # codificador: pega uma string, saída uma lista de inteiros
def decode(l):
    ''.join([itos[i] for i in l]) # decodificador: pega uma lista de inteiros, saída uma string

# cria as divisões de treino e teste
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# codifica ambos para inteiros
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"treinamento tem {len(train_ids):,} tokens")
print(f"val tem {len(val_ids):,} tokens")

# Exportar para arquivos binários
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Salve também as informações meta para ajudar a codificar/decodificar mais tarde
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Comprimento do conjunto de dados em caracteres: 1115394
# Todos os caracteres únicos:
# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# Tamanho do vocabulário: 65
# O treinamento tem 1003854 tokens
# O val tem 111540 tokens
