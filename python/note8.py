class SolarSystem:

    def __init__(self, galaxy):
        self.galaxy = galaxy


class Planet(SolarSystem):

    def tell_galaxy(self):
        print("Inner:",self.galaxy)



p = Planet("Milky Way")
p.tell_galaxy()
