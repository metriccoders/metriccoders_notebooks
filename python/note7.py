class Dog:

    def __init__(self, name):
        self.name = name
        self.friends = []

    def add_friends(self, friend):
        self.friends.append(friend)


d = Dog("tommy")
d.add_friends("mini")
d.add_friends("kini")

print(d.friends, d.name)

d = Dog("Bunny")
d.add_friends("kobi")
d.add_friends("bogi")

print(d.name, d.friends)