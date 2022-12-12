import numpy as np
from collections import OrderedDict


def lecturaDatos(fileE):
    f = open(fileE, 'r')
    line = f.readline()
    att = OrderedDict()
    ds = dataset()
    ds.data = []
    ds.target = []
    ds.feature_names = []
    ds.target_names = []
    ds.categorical = []
    clase = 'Class'
    while line.find("@data") < 0:
        if line.find("@attribute") >= 0 and ((line.find("real") >= 0) or (line.find("integer") >= 0)):
            #line = line.split()
            minimo = line.split("[")
            minimo = float(minimo[1].split(",")[0])
            maximo = float(line.split("]")[0].split(',')[1])
            attAux = {line.split()[1].split('[')[0]: [minimo, maximo]}
            att.update(attAux)
            ds.categorical.append(False)
        elif line.find("@attribute") >= 0 and line.find("real") < 0:
            #line = line.split()
            values = []
            #l = line[2]
            l = line.split('{')
            values.append(l[1].split(',')[0].strip())
            line2 = line.split(',')
            for l in line2[1:-1]:
                values.append(l.strip())
            l = line2[-1]
            l = l.split('}')[0]
            values.append(l.strip())
            attAux = {line.split()[1].split('{')[0]: values}
            att.update(attAux)
            ds.categorical.append(True)
        elif line.find("@output") >= 0 or line.find("@outputs") >= 0:
            clase = line.split()
            clase = clase[1]
        line = f.readline()
    auxClases = att.pop(clase)
    ds.categorical = ds.categorical[:-1]
    clases = auxClases[:]
    attAux = {clase: clases}
    att.update(attAux)
    line = f.readline()
    exClases = []
    examples = []
    exOriginal = []
    while line != "":
        line = line.replace(",", " ")
        l = line.split()
        values = l[0:len(l) - 1]
        val = []
        valOriginal = []
        for i, v in enumerate(values):
            if ds.categorical[i]:
                listaClaves = list(att)
                lista = att[listaClaves[i]]
                val.append(lista.index(v))
                # val.append(att[att.keys()[i]].index(v))
                valOriginal.append(v)
            else:
                val.append(float(v))
                valOriginal.append(float(v))
        examples.append(val)
        exOriginal.append(valOriginal)
        lista = att[clase]
        exClases.append(lista.index(l[len(l) - 1]))
        # exClases.append(att[clase].index(l[len(l)-1]))
        line = f.readline()
    examples = np.array(examples)
    f.close()
    ds.data = examples
    ds.target = np.array(exClases)
    aux = list(att)
    # aux = list(att.keys())
    ds.feature_names = aux[:-1]
    ds.target_names = att[clase]
    return ds, exOriginal


class dataset:
    data = []
    target = []
    feature_names = []
    target_names = []
    categorical = []
