import os

class MySentences(object):
   def __init__(self, dirname):
       self.dirname = dirname

   def __iter__(self):

        for line in open(os.path.join(self.dirname)):
            yield line.split()
