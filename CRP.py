import random

class ChineseRestaurant():
    def __init__(self, concentration, measure):
        self.tables = {}
        self.concentration = concentration
        self.measure = measure
        self.n = 0

    def sampleNew(self):
        u = random.random()*(self.n + self.concentration)
        self.n += 1

        a = 0
        for k,v in self.tables.items():
            if u < v + a:
                self.tables[k] += 1
                return k
            a += v

        new = self.measure()
        self.tables[new] = self.tables.get(new,0) + 1
        return new

    def sampleExisting(self):
        u = random.random()*(self.n)

        a = 0
        for k,v in self.tables.items():
            if u < v + a: return k
            a += v

        assert False

    def copy(self):
        r = ChineseRestaurant(self.concentration,self.measure)
        r.n = self.n
        r.tables = dict(self.tables)
        return r



if __name__ == "__main__":
    r = ChineseRestaurant(1.0,lambda : random.choice(list(range(1000))))

    s1 = []
    for _ in range(100):
        s1.append(r.sampleNew())

    s1 = dict([ (s, s1.count(s)) for s in set(s1) ])

    r = r.copy()
    s2 = []
    for _ in range(100): s2.append(r.sampleExisting())
    s2 = dict([ (s, s2.count(s)) for s in set(s2) ])
    
    print(list(sorted(iter(s1.items()),key = lambda kf: -kf[1])))
    print(list(sorted(iter(s2.items()),key = lambda kf: -kf[1])))
    
