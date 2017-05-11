class Core(object):
    def __init__(self):
        self.feature = "A"
        pass

    def funcA(self):
        print("A")
        pass

    def funcB(self):
        print("A2")
        pass


class Addon(object):
    def __init__(self):
        self.feature = "B"
        pass

    def funcB(self):
        print("B")
        pass

class Mix(Addon, Core):
    pass


a = Core()
b = Addon()
c = Mix()

a.funcB()
b.funcB()
c.funcB()

print c.feature