import pygame


class MousePointer(object):
    def __init__(self, width=320, height=240):
        pygame.init()
        size = self.width, self.height = width, height
        screen = pygame.display.set_mode(size)
        self.x, self.y = 0., 1.

    @property
    def xy(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                self.x, self.y = pygame.mouse.get_pos()
                self.x = 2. * float(self.x) / float(self.width) - 1.
                self.y = - 2. * float(self.y) / float(self.height) + 1.
        #print self.x, self.y
        return self.x, self.y

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    x, y = pygame.mouse.get_pos()
                    x = 2. * float(x) / float(self.width) - 1.
                    y = - 2. * float(y) / float(self.height) + 1.
                    return x, y

    def update(xy, ms):
        pass
