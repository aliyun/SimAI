from queue import PriorityQueue


class EventQueue:
    def __init__(self):
        self.event_queue = PriorityQueue()

    def empty(self):
        return self.event_queue.empty()

    def put(self, event):
        self.event_queue.put(event)

    def get(self):
        return self.event_queue.get()

    def __len__(self):
        return self.event_queue.qsize()
