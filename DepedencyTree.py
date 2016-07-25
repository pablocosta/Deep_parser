class DependencyTree():
    n = 0
    heads = []
    labels = []
    counter = 0


    def __init__(self):
        self.heads = []
        self.labels = []
        self.heads.append(-1)
        self.labels.append("NULL")
    def add(self, h, l):
        self.n += 1
        self.heads.append(h)
        self.labels.append(l)
    def set(self, k, h, l):
        self.heads[k] = h
        self.labels[k] = l
    def get_head(self, k):
        if (k <= 0) or (k > self.n):
            return -1
        return self.heads[k]
    def get_label(self, k):
        if (k <= 0) or (k > self.n):
            return -1
        return self.labels[k]
    def get_root(self):
        for k in range(1, self.n +1):
            if (self.get_head(k)==0):
                return k
        return 0
    def is_tree(self):
        h = []
        h.append(-1)
        for k in range(1, self.n+1):
            if self.get_head(k) < 0 or self.get_head(k) > self.n:
                return False
            h.append(-1)

        for k in range(1, self.n+1):
            i=k
            while( i>0):
                if h[i] >= 0 and h[i] < k:
                    break
                if h[i] == k:
                    return False

                h[i]=k
                i = self.get_head(i)
        return True
    def visit_tree(self, w):
        for k in range(1, w):
            if self.get_head(k) == w and self.visit_tree(k) == False:
                return False
        self.counter += 1
        if w != self.counter:
            return False
        for k in range(w+1, self.n+1):
            if self.get_head(k) == w and self.visit_tree(k) == False:
                return False
        return True
    def is_projective(self):
        if not self.is_tree():
            return False
        self.counter -= 1
        return self.visit_tree(0)


class DependencySent():
    n = 0
    words = []
    poss = []
    def __init__(self):
        self.n = 0
        self.words = []
        self.poss = []

    def add(self, word, pos):
        self.n += 1
        self.words.append(word)
        self.poss.append(pos)

class ReadTrees():
    def __init__(self, path):
        self.path = path
    def load_corpus(self):
        i = 0
        trees = []
        sents = []
        tree = DependencyTree()
        sent = DependencySent()
        arquivo = open(self.path)

        for line in arquivo:
            line = line.rstrip()
            fields = line.split("\t")
            if line is "" :

                trees.append(tree)
                sents.append(sent)
                tree = DependencyTree()
                sent = DependencySent()
            elif line.startswith("#"):
                i += 1
                continue
            else:

                word    = fields[1]
                pos     = fields[3]
                deprel  = fields[7]
                head = int(fields[6])
                tree.add(head, deprel)
                sent.add(word,pos)
        arquivo.close()
        return (trees, sents)
