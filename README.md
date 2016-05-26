# Deep_parser
An implementation of http://cs.stanford.edu/~danqi/papers/emnlp2014.pdf in python with Keras/Theano library.

--------------------------

Comando para rodar:

python3.4 rede.py

versao do keras = 0.2
-------------------------

Para rodar para uma linguagem é necessária a seguinte configuração....

Arquivos teste e train para a lingua alvo nos respectivos diretórios ./corpus/teste e ./corpus/train.... Como o no exemplo.

No embeddings deve conter os arquivos word.txt, pos.txt e label.txt. Estes são vetores na forma wor2vec de todos os tokens possíveis no conjunto de treinamento e teste. O arquivo "pre_process.py" tolkeniza nos três arquivos acima citados. Após isso só rodar "chamada.py" para cada um dos três arquivos de tokens gerados. Esses arquivos são gerados para os córpus presentes em ./corpus/teste e ./corpus/train. Recomendo fazer uma língua por vez.


O arquivo "load_model.py" pode carregar um modelo já treinado e "parsear". Em breve ele será resposável por todo o treino e teste. Atualmente ele não treina, por isso o arquivo "rede.py".

Para mudar a língua além do dito anteriormente... é necessário mudar alguns caminhos fixos em arquivos chave como "rede.py" e "load_model.py".

por exemplo no arquivo rede.py:

	Model(model, "local do córpus de teste", "nome do arquivo de modelo salvo", parser_std)

