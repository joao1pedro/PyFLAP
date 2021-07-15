from automata.fa.nfa import NFA
from automata.fa.dfa import DFA
# NFA which matches strings beginning with 'a', ending with 'a', and containing
# no consecutive 'b's
nfa = NFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'a', 'b'},
    transitions={
        'q0': {'a': {'q1'}},
        # Use '' as the key name for empty string (lambda/epsilon) transitions
        'q1': {'a': {'q1'}, '': {'q2'}},
        'q2': {'b': {'q0'}}
    },
    initial_state='q0',
    final_states={'q1'}
)

teste = nfa.read_input('aba')  # returns {'q1', 'q2'}
print('estados alcançados pelo nfa ao ler a palavra w = aba', teste)

dfa = DFA.from_nfa(nfa)

print(dfa.accepts_input('aba'))
