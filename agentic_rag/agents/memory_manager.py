# memory_manager.py
class MemoryManager:
    def __init__(self):
        self.short_term = {}
        self.long_term = {}

    def store_short(self, key, value):
        self.short_term[key] = value

    def store_long(self, key, value):
        self.long_term[key] = value

    def get_context(self):
        return {"short": self.short_term, "long": self.long_term}
