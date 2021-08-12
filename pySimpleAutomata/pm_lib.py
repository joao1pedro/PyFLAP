# import re
# import os
import csv
import xmltodict


from PySimpleAutomata import NFA, DFA, automata_IO
from pm4py.objects.log.importer.xes import importer as xes_importer
# from pm4py.objects.log.util import dataframe_utils
# from pm4py.objects.conversion.log import converter as log_converter
# from pm4py.objects.conversion.wf_net import converter as wf_net_converter
# from xml.dom import minidom
from graphviz import Digraph


def convertToJffFile(file_name, file_xes):
    fSave = open(file_name, 'w')
    log = xes_importer.apply(file_xes)
    fSave.write("<structure>\n" + "<type>fa</type>\n<automaton>\n")
    i = 1
    traces = set()
    fSave.write("\t<state id=\"0\" name=\"q0\"><initial/></state>\n")
    for case_index, case in enumerate(log):
        for event_index, event in enumerate(case):
            if(event_index != len(case) - 1):
                fSave.write("\t<state id=\"" + str(i) +
                            "\" name=\"q" + str(i) + "\"></state>\n")
            else:
                fSave.write("\t<state id=\"" + str(i) + "\" name=\"q" +
                            str(i) + "\"><final/></state>\n")

            # fSave.write("\tq"+str(i)+"\n")
            if(event_index == 0):
                trace = log[case_index][event_index]["concept:name"]
                fSave.write("\t\t<transition><from>0</from><to>" + str(i) + "</to><read>" +
                            log[case_index][event_index]["concept:name"] + "</read></transition>\n")
            else:
                trace += ", " + log[case_index][event_index]["concept:name"]
                fSave.write("\t\t<transition><from>" + str(i - 1) + "</from><to>" + str(i) + "</to><read>" +
                            log[case_index][event_index]["concept:name"] + "</read></transition>\n")
            i += 1
        # print(trace)
        traces.add(trace)
    # print(traces)
    fSave.write("</automaton>\n</structure>\n")
    fSave.close()


def jffToNFA(arquivo):
    """ Convert a .jff file as a NFA.

    :param: jff file;
    :return: *(dict)* representing a NFA.
    """
    states = set()
    initial_states = set()
    accepting_states = set()
    alphabet = set()
    transitions = {}  # key [state ∈ states, action ∈ alphabet]
    #                   value [arriving state ∈ states]

    # states.add('s0')
    # initial_states.add('s0')
    i = 0

    with open(arquivo) as fd:
        doc = xmltodict.parse(fd.read())

    # doc['mydocument']['@has'] # == u'an attribute'
    # doc['mydocument']['and']['many'] # == [u'elements', u'more elements']
    # doc['mydocument']['plus']['@a'] # == u'complex'
    # doc['mydocument']['plus']['#text'] # == u'element as well'

    # print(doc)
    for j in doc['structure']['automaton']['state']:

        states.add('s' + str(i))
        if 'initial' in j:
            initial_states.add('s' + str(i))
        if 'final' in j:
            accepting_states.add('s' + str(i))

        i = i + 1

    for k in doc['structure']['automaton']['transition']:
        transitions.setdefault(
            ('s' + k['from'], k['read']), set()).add('s' + k['to'])
        if k['read'] not in alphabet:
            alphabet.add(k['read'])

    nfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_states': initial_states,
        'accepting_states': accepting_states,
        'transitions': transitions

    }
    automata_IO
    return nfa


def jffToDFA(arquivo):
    """ Convert a .jff file as a DFA.

    :param: jff file;
    :return: *(dict)* representing a DFA.
    """
    states = set()
    initial_state = str()
    accepting_states = set()
    alphabet = set()
    transitions = {}  # key [state ∈ states, action ∈ alphabet]
    #                   value [arriving state ∈ states]

    # states.add('s0')
    # initial_states.add('s0')
    i = 0

    with open(arquivo) as fd:
        doc = xmltodict.parse(fd.read())

    # print(doc)
    for j in doc['structure']['automaton']['state']:

        states.add('s' + str(i))
        if 'initial' in j:
            initial_state = ('s' + str(i))
        if 'final' in j:
            accepting_states.add('s' + str(i))

        i = i + 1

    for k in doc['structure']['automaton']['transition']:
        transitions.setdefault(('s' + k['from'], k['read']), ('s' + k['to']))
        if k['read'] not in alphabet:
            alphabet.add(k['read'])

    dfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions

    }
    return dfa


def nfa_validate(nfa, arquivoTeste):
    errors = []
    hit = []
    results = []
    with open(arquivoTeste, 'r') as f:
        for elem in csv.reader(f, delimiter='\t'):
            if str(NFA.nfa_word_acceptance(nfa, elem[0])) != elem[1]:
                errors.append(elem[1])
                results.append(elem[1])
            hit.append(elem[1])
            results.append(elem[1])
        return results


def dfa_validate(dfa, arquivoTeste):
    errors = []
    hit = []
    results = []
    with open(arquivoTeste, 'r') as f:
        for elem in csv.reader(f, delimiter='\t'):
            if str(DFA.dfa_word_acceptance(dfa, elem[0])) != elem[1]:
                errors.append(elem[1])
                results.append(elem[1])
            hit.append(elem[1])
            results.append(elem[1])
        return results


def save_dfa_to_graph(dfa: dict, name: str, path: str = './'):
    dfa_to_graph(dfa).render(filename=name, directory=path)


def dfa_to_graph(dfa: dict, formatfile='svg'):
    """ Returns a Digraph of the input DFA using graphviz library.

    :param dict dfa: DFA to export;
    """
    g = Digraph(format=formatfile)
    g.attr(rankdir='LR')
    g.node('fake', style='invisible')
    for state in dfa['states']:
        if state == dfa['initial_state']:
            if state in dfa['accepting_states']:
                g.node(str(state), root='true',
                       shape='doublecircle')
            else:
                g.node(str(state), root='true')
        elif state in dfa['accepting_states']:
            g.node(str(state), shape='doublecircle')
        else:
            g.node(str(state))

    g.edge('fake', str(dfa['initial_state']), style='bold')
    for transition in dfa['transitions']:
        g.edge(str(transition[0]),
               str(dfa['transitions'][transition]),
               label=transition[1])

    return g


def save_nfa_to_graph(nfa: dict, name: str, path: str = './'):
    nfa_to_graph(nfa).render(filename=name, directory=path)


def nfa_to_graph(nfa: dict, formatfile='svg'):
    """ Returns a Digraph of the input NFA using graphviz library.

    :param dict nfa: input NFA;
    """
    g = Digraph(format=formatfile)
    g.attr(rankdir='LR')
    fakes = []
    for i in range(len(nfa['initial_states'])):
        fakes.append('fake' + str(i))
        g.node('fake' + str(i), style='invisible')

    for state in nfa['states']:
        if state in nfa['initial_states']:
            if state in nfa['accepting_states']:
                g.node(str(state), root='true',
                       shape='doublecircle')
            else:
                g.node(str(state), root='true')
        elif state in nfa['accepting_states']:
            g.node(str(state), shape='doublecircle')
        else:
            g.node(str(state))

    for initial_state in nfa['initial_states']:
        g.edge(fakes.pop(), str(initial_state), style='bold')
    for transition in nfa['transitions']:
        for destination in nfa['transitions'][transition]:
            g.edge(str(transition[0]), str(destination),
                   label=transition[1])
    return g


def min_aux_2(P, X, transitions, statesAlphabet):
    R = []
    if(len(X) < 2):
        return [X]
    new_R = [X[0]]
    R.append(new_R)
    for i in range(len(X) - 1):
        equal_equivalence = False
        for j in range(len(R)):
            equal_states = True
            alphabet = set(statesAlphabet[R[j][0]]).union(
                set(statesAlphabet[X[i + 1]]))
            for a in alphabet:
                dest1 = None
                dest2 = None
                if (R[j][0], a) in transitions:
                    dest1 = transitions[R[j][0], a]
                if (X[i + 1], a) in transitions:
                    dest2 = transitions[X[i + 1], a]
                equal_dest = True
                if(dest1 != dest2):
                    equal_dest = False
                    for k in range(len(P)):
                        if(dest1 in P[k] and dest2 in P[k]):
                            equal_dest = True
                            break
                if(equal_dest is False):
                    equal_states = False
                    break
            if(equal_states):
                R[j].append(X[i + 1])
                equal_equivalence = True
                break
        if(equal_equivalence is False):
            R.append([X[i + 1]])
    return R


def rename(dfa, prefix_name="s"):
    alphabet = dfa["alphabet"]
    states = set()
    initial_state = None
    accepting_states = set()
    transitions = {}
    i = 1
    tStates = {}
    for s in dfa["states"]:
        if(s == dfa["initial_state"]):
            new_state = prefix_name + '0'
            states.add(new_state)
            tStates[s] = new_state
            initial_state = new_state
            if (s in dfa['accepting_states']):
                accepting_states.add(new_state)
        else:
            new_state = prefix_name + str(i)
            states.add(new_state)
            tStates[s] = new_state
            if (s in dfa['accepting_states']):
                accepting_states.add(new_state)
            i = i + 1
    for t in dfa["transitions"]:
        transitions[tStates[t[0]], t[1]
                    ] = tStates[dfa["transitions"][t[0], t[1]]]

    new_dfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }

    return new_dfa


def getStateTransitionAlphabet(dfa):
    statesAlphabet = {}
    states = dfa["states"]
    transitions = dfa["transitions"]
    for t in transitions:
        if(t[0] in statesAlphabet):
            statesAlphabet[t[0]].add(t[1])
        else:
            statesAlphabet[t[0]] = {t[1]}
    for s in states:
        if(s not in statesAlphabet):
            statesAlphabet[s] = {}
    return statesAlphabet


def minimization(dfa):
    P = []
    transitions = dfa["transitions"]
    statesAlphabet = getStateTransitionAlphabet(dfa)
    list_accepting_states = list(dfa["accepting_states"])  # .sort()
    list_non_accepting_states = list(
        dfa["states"].difference(dfa["accepting_states"]))  # .sort()
    P.append(sorted(list_non_accepting_states))
    P.append(sorted(list_accepting_states))
    new_partition = True
    Q = []
    while(new_partition):
        Q = []
        new_partition = False
        for i in range(len(P)):
            R = min_aux_2(P, P[i], transitions, statesAlphabet)
            if(not R.__eq__([P[i]])):
                new_partition = True
            for X in R:
                Q.append(X)
        P = Q.copy()

    alphabet = dfa["alphabet"]
    transitions = dfa["transitions"]
    min_states = set()
    min_initial_state = None
    min_accepting_states = set()
    min_transitions = {}
    min_alphabet = alphabet.copy()
    N_S = []
    for S in P:
        n_s = []
        for s in S:
            n_s.append(s)
        N_S.append(n_s)
    P = N_S

    for S in P:
        state = str(S)
        min_states.add(state)
        if(dfa["initial_state"] in S):
            min_initial_state = state
        for final in list_accepting_states:
            if(final in S):
                min_accepting_states.add(state)
                break
        for a in alphabet:
            state_destination = None
            if((S[0], a) in transitions):
                state_destination = transitions[(S[0], a)]
            if(state_destination is not None):
                for D in P:
                    if(state_destination in D):
                        min_transitions[state, a] = str(D)
                        break
    min_dfa = {
        'alphabet': min_alphabet,
        'states': min_states,
        'initial_state': min_initial_state,
        'accepting_states': min_accepting_states,
        'transitions': min_transitions
    }

    return min_dfa


def exemplo1():
    states = set(['A', 'B', 'C', 'D', 'E'])
    initial_state = 'A'
    accepting_states = set('E')
    alphabet = set(['0', '1'])
    transitions = {('A', '0'): 'B', ('A', '1'): 'C',
                   ('B', '0'): 'B', ('B', '1'): 'D',
                   ('C', '0'): 'B', ('C', '1'): 'C',
                   ('D', '0'): 'B', ('D', '1'): 'E',
                   ('E', '0'): 'B', ('E', '1'): 'C'
                   }  # key [state ∈ states, action ∈ alphabet]
    #                   value [arriving state ∈ states]

    dfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }
    automata_IO.dfa_to_dot(dfa, 'dfa', 'exemplo1')
    # print(minimization(dfa))
    automata_IO.dfa_to_dot(rename(minimization(dfa)),
                           'dfa_min_trim', 'exemplo1')


def exemplo2():
    states = set(['A', 'B', 'C', 'D', 'E', 'F'])
    initial_state = 'A'
    accepting_states = set(['C', 'D', 'E'])
    alphabet = set(['0', '1'])
    transitions = {('A', '0'): 'B', ('A', '1'): 'C',
                   ('B', '0'): 'A', ('B', '1'): 'D',
                   ('C', '0'): 'E', ('C', '1'): 'F',
                   ('D', '0'): 'E', ('D', '1'): 'F',
                   ('E', '0'): 'E', ('E', '1'): 'F',
                   ('F', '0'): 'F', ('F', '1'): 'F'
                   }
    dfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }
    automata_IO.dfa_to_dot(dfa, 'dfa', 'exemplo2')
    # print(minimization(dfa))
    automata_IO.dfa_to_dot(rename(minimization(dfa)),
                           'dfa_min_trim', 'exemplo2')


def exemplo3():
    states = set(['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
    initial_state = 'q0'
    accepting_states = set(['q2'])
    alphabet = set(['0', '1'])
    transitions = {('q0', '0'): 'q1', ('q0', '1'): 'q5',
                   ('q1', '0'): 'q6', ('q1', '1'): 'q2',
                   ('q2', '0'): 'q0', ('q2', '1'): 'q2',
                   ('q3', '0'): 'q2', ('q3', '1'): 'q6',
                   ('q4', '0'): 'q7', ('q4', '1'): 'q5',
                   ('q5', '0'): 'q2', ('q5', '1'): 'q6',
                   ('q6', '0'): 'q6', ('q6', '1'): 'q4',
                   ('q7', '0'): 'q6', ('q7', '1'): 'q2'
                   }
    dfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_state': initial_state,
        'accepting_states': accepting_states,
        'transitions': transitions
    }
    automata_IO.dfa_to_dot(dfa, 'dfa', 'exemplo3')
    # print(minimization(dfa))
    automata_IO.dfa_to_dot(rename(minimization(dfa)),
                           'dfa_min_trim', 'exemplo3')


def convertToListOfTraces(file_xes, max_traces=-1, sort=False):
    variant = xes_importer.Variants.ITERPARSE
    if(max_traces != -1):
        parameters = {variant.value.Parameters.MAX_TRACES: max_traces}
    else:
        parameters = None
    log = xes_importer.apply(file_xes, parameters)
    lLog = []
    for case_index, case in enumerate(log):
        l = []
        for event_index, event in enumerate(case):
            l.append(log[case_index][event_index]["concept:name"])
        if(lLog.__contains__(l) is False):
            lLog.append(l)
    if(sort is True):
        lLog.sort()
    return lLog


def toNFA(lLog):
    """ Convert a list of traces (logs) as a NFA.

    :param: List of Traces (e.g. [ ['a','b'], ['a','b','e'], ['a','e'] ]);
    :return: *(dict)* representing a NFA.
    """
    states = set()
    initial_states = set()
    accepting_states = set()
    alphabet = set()
    transitions = {}  # key [state ∈ states, action ∈ alphabet]
    #                   value [arriving state ∈ states]

    states.add('s0')
    initial_states.add('s0')
    i = 1
    for j in range(len(lLog)):
        for k in range(len(lLog[j])):
            states.add('s' + str(i))
            if(k == len(lLog[j]) - 1):
                accepting_states.add('s' + str(i))
            alphabet.add(lLog[j][k])

            if(k == 0):
                transitions.setdefault(
                    ('s0', lLog[j][k]), set()).add('s' + str(i))
            else:
                transitions.setdefault(
                    ('s' + str((i - 1)), lLog[j][k]), set()).add('s' + str(i))
            i += 1

    nfa = {
        'alphabet': alphabet,
        'states': states,
        'initial_states': initial_states,
        'accepting_states': accepting_states,
        'transitions': transitions
    }

    return nfa


def automatonDecomposition(nfa):

    initial_states1 = set()
    accepting_states1 = set()
    alphabet1 = set()
    transitions1 = {}
    states1 = set()

    initial_states2 = set()
    accepting_states2 = set()
    alphabet2 = set()
    transitions2 = {}
    states2 = set()

    media = len(nfa['states']) / 2
    # print("media: " + str(media))
    # print("tamanho" + str(len(nfa['states'])))
    # for i in range(len(nfa)):
    for i in nfa['states']:
        if len(states1) < media:
            states1.add(i)
            if i in nfa['initial_states']:
                initial_states1.add(i)
            if i in nfa['accepting_states']:
                accepting_states1.add(i)
        else:
            if(i not in states1):
                states2.add(i)
                """for j in transicoes:
                    if(j not in states1):
                        states1.add(j)
                        if j in nfa['initial_states']:
                            initial_states1.add(j)
                        if j in nfa['accepting_states']:
                            accepting_states1.add(j) """
                if i in nfa['initial_states']:
                    initial_states2.add(i)
                if i in nfa['accepting_states']:
                    accepting_states2.add(i)
    alphabet1 = nfa['alphabet']
    alphabet2 = nfa['alphabet']
    states1.add('R')
    states2.add('S')
    accepting_states1.add('R')
    accepting_states2.add('S')

    if len(initial_states2) == 0:
        initial_states2.add('S')
    if len(initial_states1) == 0:
        initial_states1.add('R')

    print("states 1: ", states1)
    print("states 2: ", states2)

    for i in nfa['transitions']:
        print("estado e o que ele lê: ", i)
        print("pra onde vai: ", list(nfa['transitions'].setdefault(i)))
        transicoes = list(nfa['transitions'].setdefault(i))
        # print('i[0]', i[0])
        if i[0] in states1:
            # aux = set()
            for j in transicoes:
                print('j', j)
                # aux.add(j)
                # print('aux: ', aux)
                if j in states1:
                    print("entrou1")
                    transitions1.setdefault((i[0], i[1]), set()).add(j)

                    # transitions1.setdefault((i[0], i[1]), aux)

                else:
                    print("cruzou1")
                    # if transitions1.setdefault((i[0], i[1])):
                    transitions1.setdefault((i[0], i[1]), set()).add('R')
                    # transitions2.setdefault(('S', i[1]), aux)
                    transitions2.setdefault(('S', i[1]), set()).add(j)
                print('transitions1: ', transitions1)
        elif i[0] in states2:
            for j in transicoes:
                print('j', j)
                aux = set()
                aux.add(j)
                if j in states2:
                    print("entrou2")
                    transitions2.setdefault((i[0], i[1]), set()).add(j)

                else:
                    print("cruzou2")
                    transitions2.setdefault((i[0], i[1]), set()).add('S')
                    transitions1.setdefault(('R', i[1]), set()).add(j)
                print('transitions2: ', transitions2)

    nfa1 = {
        'alphabet': alphabet1,
        'states': states1,
        'initial_states': initial_states1,
        'accepting_states': accepting_states1,
        'transitions': transitions1
    }

    nfa2 = {
        'alphabet': alphabet2,
        'states': states2,
        'initial_states': initial_states2,
        'accepting_states': accepting_states2,
        'transitions': transitions2
    }

    return nfa1, nfa2


def convertToDOTFile2(file_name, file_xes):
    fSave = open(file_name, 'w')
    log = xes_importer.apply(file_xes)
    lLog = []
    for case_index, case in enumerate(log):
        l = []
        for event_index, event in enumerate(case):
            l.append(log[case_index][event_index]["concept:name"])
        if(lLog.__contains__(l) is False):
            lLog.append(l)
        # print(l)
    lLog.sort()
#    for l in lLog:
#        print(l)

    fSave.write("digraph{\n" +
                "\tfake [style=invisible]\n" +
                "\tfake -> s0 [style=bold]\n" +
                "\tfake -> s0 [style=bold]\n" +
                "\ts0 [root=true]\n")
    i = 1
    for j in range(len(lLog)):
        for k in range(len(lLog[j])):
            if(k != len(lLog[j]) - 1):
                fSave.write("\ts" + str(i) + "\n")
            else:
                fSave.write("\ts" + str(i) + " [shape=doublecircle]\n")
            if(k == 0):
                fSave.write("\t\ts0 -> s" + str(i) +
                            "[label=\"" + lLog[j][k] + "\"]\n")
            else:
                fSave.write("\t\ts" + str((i - 1)) + " -> s" + str(i) +
                            "[label=\"" + lLog[j][k] + "\"]\n")
            i += 1
    fSave.write("}")


def convertToDOTFile(file_name, file_xes):
    fSave = open(file_name, 'w')
    log = xes_importer.apply(file_xes)
    fSave.write("digraph{\n" +
                "\tfake [style=invisible]\n" +
                "\tfake -> s0 [style=bold]\n" +
                "\tfake -> s0 [style=bold]\n" +
                "\ts0 [root=true]\n")
    i = 1
    traces = set()
    for case_index, case in list(enumerate(log)).sort():
        trace = ""
        for event_index, event in enumerate(case):
            if(event_index != len(case) - 1):
                fSave.write("\ts" + str(i) + "\n")
            else:
                fSave.write("\ts" + str(i) + " [shape=doublecircle]\n")

            if(event_index == 0):
                trace = log[case_index][event_index]["concept:name"]
                fSave.write("\t\ts0 -> s" + str(i) +
                            "[label=\"" + event["concept:name"] + "\"]\n")
            else:
                trace += ", " + log[case_index][event_index]["concept:name"]
                fSave.write("\t\ts" + str((i - 1)) + " -> s" + str(i) +
                            "[label=\"" + event["concept:name"] + "\"]\n")
            i += 1
        print("t " + log[case_index])
        traces.add(trace)
    fSave.write("}")


def automata_nfa_union(automata1: dict, automata2: dict) -> dict:
    return NFA.nfa_union(automata1, automata2)


def automata_nfa_intersection(automata1: dict, automata2: dict) -> dict:
    return NFA.nfa_intersection(automata1, automata2)


def automata_dfa_union(automata1: dict, automata2: dict) -> dict:
    return DFA.dfa_union(automata1, automata2)


def automata_dfa_intersection(automata1: dict, automata2: dict) -> dict:
    return DFA.dfa_intersection(automata1, automata2)
