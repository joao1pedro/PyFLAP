
import pm_lib 
from PySimpleAutomata import automata_IO

arquivo = 'nfa.jff'
arquivoTeste = 'nfa.cases'

arquivo_dfa = 'dfa.jff'
arquivo_teste_dfa = 'dfa.cases'


print("NFA example: ")
nfa = pm_lib.jffToNFA(arquivo)
print("\n")
param1= pm_lib.nfa_validate(nfa, arquivoTeste)
pm_lib.uncouple(param1)
automata_IO.nfa_to_dot(nfa, 'teste_nfa', './')


print("\n")
print("DFA example: ")
dfa = pm_lib.jffToDFA(arquivo_dfa)
print("\n")
param1_= pm_lib.dfa_validate(dfa, arquivo_teste_dfa)
pm_lib.uncouple(param1_)
automata_IO.dfa_to_dot(dfa, 'teste_dfa', './')


