class Picture(object):

    def __init__(self, image):
        self.image = image
        self.height = image.shape[1]
        self.width = image.shape[0]
        self.color = True
        self.grayscale = False
        self.binary = False
        self.operations = []
