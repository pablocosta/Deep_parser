from nltk.parse import DependencyGraph

arquivo = open("./corpus/train/portuguese_train.conll")
a = arquivo.read()
graphs = [DependencyGraph(entry) for entry in a.split('\n\n') if entry]
for depgraph in graphs:
    print(depgraph.nodes)
    input()