class TranspositionTable():
    def __init__(self):
        self.table = {} # key = state, value = (probs, value, model_age)
        
    def add(self, state, probs, value):        
        self.table[state] = (probs, value)
        
    def get(self, state):
        return self.table.get(state, (None, None))
        