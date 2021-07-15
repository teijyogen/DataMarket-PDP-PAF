import random
import math
#from Laplace import Lap, Q

#import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from Database import *
import scipy.optimize as opt
from Math import *

class Sample:
    def __init__(self, sub_mech):
        self.sub_mech = sub_mech
        self.t = 0.0
        self.eq_pattern = []
        self.flexible = False

    def min_var(self, query):
        self.eq_pattern = query.D.users.adjusted_pattern
        pattern = self.eq_pattern

        if self.flexible == True:
            self.compute_eq_pattern(query)
            pattern = self.eq_pattern

        def f(x):
            expr = 2 / x ** 2
            for i in range(len(pattern)):
                expr = expr + ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)) * (1 - ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)))
            return expr
        budgets = list(query.D.list_budgets().values())
        budgets = patterning(pattern, budgets)
        maxB = max(budgets)
        if maxB == 0.0:
            query.min_var = float('inf')
        else:
            query.min_var = f(maxB)

        # if self.flexible == True and query.var < query.min_var:
        #     print("pattern to change")
        #     self.compute_eq_pattern(query)
        #     pattern = self.eq_pattern
        #
        #     def f(x):
        #         expr = 2 / x ** 2
        #         for i in range(len(pattern)):
        #             expr = expr + ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)) * (
        #                         1 - ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)))
        #         return expr
        #
        #     budgets = list(query.D.list_budgets().values())
        #     budgets = patterning(pattern, budgets)
        #     maxB = max(budgets)
        #     print(query.min_var)
        #     if maxB == 0.0:
        #         query.min_var = float('inf')
        #     else:
        #         query.min_var = f(maxB)
        #
        #     print(query.min_var)

    def decom_var(self, query):
        pattern = self.eq_pattern
        budgets = list(query.D.list_budgets().values())
        budgets = patterning(pattern, budgets)
        maxB = max(budgets)
        if query.var == query.min_var:
            self.t = maxB
            return


        def f(x):
            expr = 2 / x ** 2 - query.var
            for i in range(len(pattern)):
                expr = expr + ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)) * (1 - ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)))
            return expr

        start = 0.0000000000001
        try:
            self.t = opt.fsolve(f, start)[0]
        except:
            self.t = 0.0

        # except:
        #     print(query.var)

        # while self.t < 0:
        #     start = start * 0.1
        #     self.t = opt.fsolve(f, start)[0]
        if self.t > maxB:
            self.t = maxB
            query.var = query.min_var

        if self.t <= 0.0:
            self.t = 0.0
            query.var = float('inf')


    def loss(self, db):
        pattern = self.eq_pattern
        # pattern = db.users.adjusted_pattern
        for i in range(db.length):
            db.data[i].loss = self.t * pattern[i]


    def compute_eq_pattern(self, query):
        db = query.D
        pattern = db.users.adjusted_pattern
        # print(pattern)
        rates = db.users.compute_rates()
        # sizes = db.users.compute_sizes()
        # remaining = db.remaining
        n = len(pattern)
        # re_pattern = [remaining[i] / sizes[i] for i in range(n)]
        # re_pattern = [re_pattern[i] / max(re_pattern) for i in range(n)]
        # re_pattern = [remaining[i] / max(remaining) for i in range(n)]

        budgets = list(query.D.list_budgets().values())
        adjusted_budgets = patterning(pattern, budgets)
        total_budget = sum(adjusted_budgets)

        eq_pattern = [[i, pattern[i], rates[i], budgets[i]] for i in range(n)]

        rate_set = set(rates)
        eq_pattern_rate = []
        for rate in rate_set:
            rate_sub_list = []
            rate_pattern = []
            for i in range(n):
                if eq_pattern[i][2] == rate:
                    rate_sub_list.append(eq_pattern[i])
                    rate_pattern.append(eq_pattern[i][1])
            rate_pattern.sort()
            rate_sub_list_budget = sorted(rate_sub_list, key=lambda x:x[3])

            for j in range(len(rate_sub_list_budget)):
                rate_sub_list_budget[j][1] = rate_pattern[j]
                eq_pattern_rate.append(rate_sub_list_budget[j])

        eq_pattern_rate_id = sorted(eq_pattern_rate, key=lambda x:x[0])
        self.eq_pattern = [eq_pattern_rate_id[i][1] for i in range(n)]
        # print(eq_pattern)
        # for i in range(n):
        #     # if eq_pattern[i][1] != 1.0:
        #         for j in range(n):
        #             if eq_pattern[i][2] == eq_pattern[j][2]:
        #                 if eq_pattern[i][1] != eq_pattern[j][1]:
        #                             # print("Candidate founded")
        #                 # if eq_pattern[-j][1] - eq_pattern[-j][3] < 0:
        #                     # if abs(eq_pattern[-j][1] - eq_pattern[i][3]) < abs(eq_pattern[i][1] - eq_pattern[i][3]):
        #                     #     if abs(eq_pattern[i][1] - eq_pattern[-j][3]) < abs(eq_pattern[-j][1] - eq_pattern[-j][3]):
        #                             candidate = eq_pattern.copy()
        #                             candidate = [candidate[k][1] for k in range(n)]
        #                             pi = candidate[i]
        #                             pj = candidate[j]
        #                             candidate[i] = pj
        #                             candidate[j] = pi
        #                             if sum(patterning(candidate, budgets)) > total_budget:
        #                                 print("Pattern changed")
        #                                 p1 = eq_pattern[i][1]
        #                                 p2 = eq_pattern[j][1]
        #                                 eq_pattern[i][1] = p2
        #                                 eq_pattern[j][1] = p1
        #                                 total_budget = sum(patterning(candidate, budgets))
        # # eq_pattern = sorted(eq_pattern, key=lambda eq_pattern: eq_pattern[0])
        # self.eq_pattern = [eq_pattern[i][1] for i in range(n)]
        # # print(self.eq_pattern)





    def perturb(self, query):
        db = query.D
        data = db.data
        sub_db = Database(db.time_stamp, db.users)

        self.decom_var(query)
        self.loss(db)

        for i in range(db.length):
            if data[i].loss < self.t:
                pi = (math.exp(data[i].loss) - 1) / (math.exp(self.t) - 1)
                r = random.random()
                if r <= pi:
                    sub_db.add_datapoint(data[i])
            else:
                sub_db.add_datapoint(data[i])

        answer = 0

        for key in sub_db.his.keys():
            answer = answer + sub_db.his[key]

        if self.t == 0.0:
            beta = float('inf')
        else:
            beta = self.sub_mech.sensi / self.t

        if beta < 0.0:
            print(self.t)
            print(query.var)

        answer = answer + np.random.laplace(0, beta)

        query.answer = answer

        query.quote()

            # for key in query.his.keys():
            #     if key in sub_db.answer.keys():
            #         query.his[key] = sub_db.answer[key]
            #     else:
            #         query.his[key] = 0
            #
            # if self.t == 0.0:
            #     beta = float('inf')
            # else:
            #     beta = self.sub_mech.sensi / self.t
            #
            # for key in query.his.keys():
            #         query.his[key] = query.D.answer[key] + np.random.laplace(0, beta)
            # query.quote()

    def quote_var(self, query, var):
        self.eq_pattern = query.D.users.adjusted_pattern
        pattern = self.eq_pattern
        def f(x):
            expr = 2 / x ** 2 - var
            for i in range(len(pattern)):
                expr = expr + ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)) * (1 - ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)))
            return expr

        start = 0.000000000000001
        self.t = opt.fsolve(f, start)[0]
        self.loss(query.D)
        query.quote()
        price = query.price
        self.eq_pattern = []
        query.price = 0.0
        query.D.empty_loss()
        return price
