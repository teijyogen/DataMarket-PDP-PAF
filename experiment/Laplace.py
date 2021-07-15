import numpy as np
import random

class Laplace:
    def __init__(self, sensitivity=1.0):
        self.sensi = sensitivity
        self.epsi = 0.0

    def var(self, loss):
        return 2 * (self.sensi / loss) ** 2

    def min_var(self, query):
        loss = query.D.find_min_budget()
        if loss == 0.0:
            query.min_var = float('inf')
        else:
            query.min_var = 2 * (self.sensi / loss) ** 2

    def decom_var(self, query):
        if query.var == query.min_var:
            self.epsi = query.D.find_min_budget()
            return
        self.epsi = self.sensi / (query.var * 0.5) ** 0.5
        #print(query.D.find_min_budget())
        min_budget = query.D.find_min_budget()

        if self.epsi > min_budget:
            self.epsi = min_budget
            query.var = query.min_var

    def loss(self, query):
        query.D.uni_loss(self.epsi)

    def perturb(self, query):
        self.decom_var(query)
        #query.D.print_losses()
        self.loss(query)

        answer = 0
        for key in query.his.keys():
            answer = answer + query.his[key]

        if self.epsi == 0.0:
            beta = float('inf')
        else:
            beta = self.sensi / self.epsi

        answer = answer + np.random.laplace(0, beta)
        query.answer = answer

        query.quote()

    def patterning(self, db):
        D.pattern = [1.0 for i in range(db.users.length)]

    def quote_var(self, query, var):
        self.epsi = self.sensi / (var * 0.5) ** 0.5
        query.D.loss(self.epsi)
        query.quote()
        price = query.price
        self.epsi = 0.0
        query.price = 0.0
        query.D.empty_loss()
        return price






