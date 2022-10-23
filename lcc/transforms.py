

class Normalize(object):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):        
        sample['image'] = (sample['image'] - self.min) / (self.max - self.min)
        return sample

