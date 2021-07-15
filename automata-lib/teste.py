import csv

arquivoTeste = 'teste.cases'

with open (arquivoTeste, 'r') as f:
    for i in csv.reader(f, delimiter='\t'):
        print(i)
