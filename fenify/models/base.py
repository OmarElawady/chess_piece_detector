class Model:
    def __init__(self, expected_shape=()):
        self.model = None
        self.name = "basic"

    def train(self, epochs):
        pass
    
    def save_snapshot(self):
        raise NotImplementedError()

    def load_latest(self):
        raise NotImplementedError()

    def metadata(self):
        raise NotImplementedError()

    def confusion_matrix(self):
        raise NotImplementedError()

    def accuracy(self):
        raise NotImplementedError()
    
    def miscolored(self):
        raise NotImplementedError()

    def get_samples(self, board_image):
        raise NotImplementedError()