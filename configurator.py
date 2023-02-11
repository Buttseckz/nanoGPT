# Configurador simples. Provavelmente uma ideia terrível. Exemplo de uso:
# $ python train.py config/override_file.py --batch_size=32
# isto primeiro executará o arquivo config/override_file.py, em seguida sobrescreverá o batch_size para 32

# O código neste arquivo será executado da seguinte forma a partir, por exemplo, do train.py:
# >>> exec(open('configurator.py').read())

# Então, não é um módulo Python, apenas está enviando este código longe do train.py
# O código neste script, então, substitui os globals()

# Eu sei que as pessoas não vão gostar disso, eu simplesmente realmente detesto a complexidade de configuração
# e ter que preceder config. a cada única variável. Se alguém vier com uma solução Python simples
# Estou de ouvidos abertos.

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # Suponha que seja o nome de um arquivo de configuração
        assert not arg.startswith('--')
        config_file = arg
        print(f"Substituindo configuração com {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # Suponha que seja um argumento --key=value
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # tentativa de evaluá-lo (por exemplo, se for booleano, número ou etc.)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # se isso der errado, use apenas a string
                attempt = val
            # garanta que os tipos combinem bem
            assert type(attempt) == type(globals()[key])
            # cruze os dedos
            print(f"Substituindo: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Chave de configuração desconhecida: {key}")
