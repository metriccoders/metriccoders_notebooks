class Dog:
    kind = "canine"

    def __init__(self, name):
        self.name = name



d = Dog("Tommy")
print(d.kind, d.name)

d = Dog("Bunny")
print(d.kind, d.name)