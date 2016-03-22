file_prob = open("./corpus/par/lex.f2e","r")
file_dict = open("./corpus/par/dictionary_es-pt","w")
dictionary_prob = dict()


for line in file_prob:
    line = line.rstrip()
    fields = line.split()
    if not fields[0]+"|||"+fields[1] in dictionary_prob.keys():

        dictionary_prob[fields[0]+"|||"+fields[1]] = fields[2]

    else:
        if float(fields[2]) > float(dictionary_prob[fields[0]+"|||"+fields[1]]):
            dictionary_prob[fields[0]+"|||"+fields[1]] = fields[2]
        else:
            continue


for key in dictionary_prob.keys():
    file_dict.write(key+"\n")

file_prob.close()
file_dict.close()