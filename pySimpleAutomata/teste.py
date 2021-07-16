
import pm_lib 
from PySimpleAutomata import automata_IO


#lLog = pm_lib.convertToListOfTraces("teste.xes",max_traces=7554, sort=True)
arquivo = 'nfa.jff'
arquivoTeste = 'nfa.cases'

arquivo_dfa = 'dfa.jff'
arquivo_teste_dfa = 'dfa.cases'

#path = "imagens"


#lLog = pm_lib.convertToListOfTraces("teste.xes",max_traces=7554, sort=True)


print("NFA example: ")
nfa = pm_lib.jffToNFA(arquivo)
print("\n")
param1= pm_lib.nfa_validate(nfa, arquivoTeste)
pm_lib.uncouple(param1)
automata_IO.nfa_to_dot(nfa, 'teste_nfa', './')

#nfa = pm_lib.toNFA(lLog)
#print('NFA: \t\talphabet ={:3d}, states = {:5d}, transitions = {:5d}, accepting states = {:5d}'.format(len(nfa["alphabet"]),len(nfa["states"]),len(nfa["transitions"]),len(nfa["accepting_states"])))
#print(nfa['transitions'])

print("\n")
print("DFA example: ")
dfa = pm_lib.jffToDFA(arquivo_dfa)
print("\n")
param1_= pm_lib.dfa_validate(dfa, arquivo_teste_dfa)
pm_lib.uncouple(param1_)
automata_IO.dfa_to_dot(dfa, 'teste_dfa', './')



""" nfa1, nfa2 = pm_lib.automatonDecomposition(nfa)
print('nfa 1: ')
print('NFA: \t\talphabet ={:3d}, states = {:5d}, transitions = {:5d}, accepting states = {:5d}'.format(len(nfa1["alphabet"]),len(nfa1["states"]),len(nfa1["transitions"]),len(nfa1["accepting_states"])))
print('estados 1: ', nfa1['states'])
print('transições 1: ', nfa1['transitions'])


print('nfa 2: ')
print('NFA: \t\talphabet ={:3d}, states = {:5d}, transitions = {:5d}, accepting states = {:5d}'.format(len(nfa2["alphabet"]),len(nfa2["states"]),len(nfa2["transitions"]),len(nfa2["accepting_states"])))
print('estados 2: ', nfa2['states'])
print('transições 2: ', nfa2['transitions'])


nfa3, nfa4 = pm_lib.automatonDecomposition(nfa1)
nfa5, nfa6 = pm_lib.automatonDecomposition(nfa2)

#print(nfa1['transitions'])  """

#print("NFA DOT")
#automata_IO.nfa_to_dot(nfa, 'nfa_original', path)
"""print("NFA 1")
automata_IO.nfa_to_dot(nfa1, 'nfa1', path)
print("NFA 2")
automata_IO.nfa_to_dot(nfa2, 'nfa2', path)
print("NFA 3")
automata_IO.nfa_to_dot(nfa3, 'nfa3', path)
print("NFA 4")
automata_IO.nfa_to_dot(nfa4, 'nfa4', path)
print("NFA 5")
automata_IO.nfa_to_dot(nfa5, 'nfa5', path)
print("NFA 6")
automata_IO.nfa_to_dot(nfa6, 'nfa6', path)

nfaNovo1 = NFA.nfa_intersection(nfa3, nfa4)
nfaNovo2 = NFA.nfa_intersection(nfa5, nfa6)

nfaIntersecao = NFA.nfa_intersection(nfaNovo2, nfaNovo1)


automata_IO.nfa_to_dot(nfaNovo1, 'nfaNovo1', path)
automata_IO.nfa_to_dot(nfaNovo2, 'nfaNovo2', path)
automata_IO.nfa_to_dot(nfaIntersecao, 'nfaJunto', path)

#nfa = pm_lib.toNFA(lLog)
#print('NFA: \t\talphabet ={:3d}, states = {:5d}, transitions = {:5d}, accepting states = {:5d}'.format(len(nfa["alphabet"]),len(nfa["states"]),len(nfa["transitions"]),len(nfa["accepting_states"])))

 """

