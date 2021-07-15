from automata.fa.dfa import DFA

dfa = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0','1'},
    transitions={
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'0': 'q0', '1': 'q2'},
        'q2': {'0': 'q2', '1': 'q1'},
        #'q2': {'0': {'q0'}, '1': 'q2'}
    },
    initial_state='q0',
    final_states={'q1'}
)

if dfa.accepts_input('0111'):
    print('accepted')
else:
    print('rejected')
