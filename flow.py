class Flow:
    def __init__(self, data, error=None):
        self.data = data
        self.error = error
    def bind(self, func):
        if self.error: # error, skip
            return self
        try:
            result = func(self.data)
            return Flow({**self.data, **(result or {})}) # extend data
        except Exception as e:
            return Flow(self.data, error=f"Error in [{func.__name__}]: {e}")
    def get(self):
        return self.data, self.error
    
