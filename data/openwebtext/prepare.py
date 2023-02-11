# Salva o conjunto de dados openwebtext em um arquivo binário para treinamento. O seguinte foi útil:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Número de trabalhadores na chamada .map()
# Um bom número a ser usado é ~número de núcleos de CPU // 2
num_proc = 8

# leva 54GB no diretório .cache do huggingface, cerca de 8M de documentos (8,013,769)
dataset = load_dataset("openwebtext")

# o owt por padrão só contém a divisão 'train', então crie uma divisão de teste
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test') # renomeie a divisão de teste para val

# isso resulta em:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# agora queremos tokenizar o conjunto de dados. Primeiro, definimos a função de codificação (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignora qualquer token especial
    ids.append(enc.eot_token) # adicione o token de fim de texto, por exemplo, 50256 para gpt2 bpe
    # nota: acho que o eot deve ser adicionado e não precedido... hmm. é chamado de "eot"...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize o conjunto de dados
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizando as divisões",
    num_proc=num_proc,
)

# Este trecho de código concatena todos os ids em cada conjunto de dados em um único arquivo grande
#  que pode ser usado para treinamento. Primeiro, é calculado o comprimento total do array, depois 
# é definido o nome do arquivo a ser salvo. Em seguida, é definido o tipo de dados a ser salvo como
#  np.uint16.
#O arquivo é criado usando a função np.memmap, que permite criar uma memória mapeada do numpy. 
# O nome do arquivo é exibido com o comando print e, em seguida, é iniciado um loop através dos 
# dados para escrever os ids no arquivo. Finalmente, as mudanças são escritas na memória com a 
# função flush().
# adiciona comentários nas linhas não comentadas

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len']) # calcula o comprimento total do array
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin') # define o nome do arquivo
    dtype = np.uint16 # define o tipo de dados a ser salvo (np.uint16 pois enc.max_token_value == 50256 é < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) # cria uma memmap do numpy

    print(f"writing {filename}...") # mostra o nome do arquivo sendo escrito
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids'] # escreve os ids no arquivo
        idx += example['len']
    arr.flush() # escreve as mudanças na memória


# train.bin tem cerca de 17GB, val.bin tem cerca de 8,5MB
#train tem cerca de 9B tokens (9.035.582.198)
#val tem cerca de 4M tokens (4.434.897)
#para ler os arquivos bin mais tarde, por exemplo, com numpy:
#m = np.memmap('train.bin', dtype=np.uint16, mode='r')