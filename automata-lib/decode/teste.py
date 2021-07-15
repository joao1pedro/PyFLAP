
import decode

arquivo = 'teste.jff'
arquivoTeste = 'teste.cases'


nfa = decode.jffToNFA(arquivo)
print(decode.teste(nfa, arquivoTeste))
