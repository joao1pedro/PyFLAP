from automata.fa.nfa import NFA
from pm4py.objects.log.importer.xes import importer as xes_importer

import xmltodict
import csv

def convertToJffFile(file_name, file_xes):
    fSave = open(file_name, 'w')
    log = xes_importer.apply(file_xes)
    fSave.write("<structure>\n"+"<type>fa</type>\n<automaton>\n")
    i = 1
    traces = set()
    fSave.write("\t<state id=\"0\" name=\"q0\"><initial/></state>\n")
    for case_index, case in enumerate(log):
        for event_index, event in enumerate(case):
            if(event_index!=len(case)-1):
                fSave.write("\t<state id=\""+str(i)+"\" name=\"q"+str(i)+"\"></state>\n")
            else:
                fSave.write("\t<state id=\""+str(i)+"\" name=\"q"+str(i)+"\"><final/></state>\n")

            #fSave.write("\tq"+str(i)+"\n")
            if(event_index==0):
                trace = log[case_index][event_index]["concept:name"]
                fSave.write("\t\t<transition><from>0</from><to>"+str(i)+"</to><read>"+log[case_index][event_index]["concept:name"]+"</read></transition>\n")
            else:
                trace+=", "+log[case_index][event_index]["concept:name"]
                fSave.write("\t\t<transition><from>"+str(i-1)+"</from><to>"+str(i)+"</to><read>"+log[case_index][event_index]["concept:name"]+"</read></transition>\n")
            i += 1
        #print(trace)    
        traces.add(trace)
    #print(traces)
    fSave.write("</automaton>\n</structure>\n")
    fSave.close()

def jffToNFA(file):
    states = set()
    input_symbols = set()
    transitions = {}
    initial_state=vars()
    final_states=set()

    i = 0

    with open(file) as fd:
        doc = xmltodict.parse(fd.read())
    
    print(doc)

    states.add('')
    for j in doc['structure']['automaton']['state']:
        states.add('q' + str(i))
        if 'initial' in j:
            initial_state = 'q' + str(i)
        if 'final' in j:
            final_states.add('q' + str(i))
        i = i+1 
    
    for k in doc['structure']['automaton']['transition']:
        transitions.setdefault(('q' + k['from'], k['read']), set()).add('q' + k['to'])
        if k['read'] not in input_symbols:
            input_symbols.add(k['read'])
    
    nfa = NFA(
        states=states,
        input_symbols=input_symbols,
        transitions= transitions,
        #{
        #'q0': {'0': {'q1'}, '1':{'q0'}},
        #'q1': {'0': {'q1'}, '1':{'q2'}},
        #'q2': {'0': {''}, '1': {''}}
        #},
        initial_state=initial_state,
        final_states=final_states
    )

    return nfa

def teste(nfa, arquivoTeste):
    erros = []
    with open (arquivoTeste, 'r') as f:
        for elem in csv.reader(f, delimiter='\t'):
            if str(NFA.accepts_input(nfa, elem[0])) != elem[1]:
                erros.append(elem)
                return False
        return True
