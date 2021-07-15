import random
import numpy as np
import time
import operator
from Math import *

class User:
    __slots__ = ['id', 'w', 'bound', 'c']

    def __init__(self, user_id, w, bound, c):
        self.id = user_id
        self.w = w
        self.bound = bound
        self.c = c


class Users:
    def __init__(self):
        self.users = []
        self.length = 0
        self.adjusted_pattern = []
        self.ab_pattern = []
        self.partial_pattern = []

    def compute_rates(self):
        return [self.users[i].c for i in range(self.length)]

    def compute_sizes(self):
        return [self.users[i].w for i in range(self.length)]

    def add_pattern(self):
        initP = []
        for i in range(self.length):
            avearge_bound = self.users[i].bound / self.users[i].w
            initP.append(avearge_bound)
        max_bound = max(initP)
        for i in range(self.length):
            initP[i] = initP[i] / max_bound
        self.adjusted_pattern = patternSearch(initP)

    def add_user(self, user):
        self.users.append(user)
        self.length += 1

    def check_id(self, user_id):
        for i in range(self.length):
            if self.users[i].id == user_id:
                return True
        return False

    def find_w_by_id(self, user_id):
        for i in range(self.length):
            if self.users[i].id == user_id:
                return self.users[i].w

    def find_bound_by_id(self, user_id):
        for i in range(self.length):
            if self.users[i].id == user_id:
                return self.users[i].bound

    def find_c_by_id(self, user_id):
        for i in range(self.length):
            if self.users[i].id == user_id:
                return self.users[i].c


class Datapoint:
    __slots__ = ['user_id', 'time_stamp', 'loc', 'budget', 'loss']

    def __init__(self, user_id, time_stamp, loc):
        self.user_id = user_id
        self.time_stamp = time_stamp
        self.loc = loc
        self.budget = 0.0
        self.loss = 0.0


class Database:
    def __init__(self, time_stamp, users):
        self.time_stamp = time_stamp
        self.length = 0
        self.num_of_locs = 0
        self.data = []
        self.users = users
        self.his = {}
        self.remaining = []

    def sort_by_budget(self):
        cmpfun = operator.attrgetter('budget')
        self.data.sort(key=cmpfun, reverse=False)

    def sort_by_id(self):
        cmpfun = operator.attrgetter('user_id')
        self.data.sort(key=cmpfun)

    def empty_loss(self):
        for i in range(self.length):
            self.data[i].budget = 0.0
            self.data[i].loss = 0.0

    def generate_data(self, num_of_locs):
        self.num_of_locs = num_of_locs
        self.data = []
        self.his.clear()
        for i in range(self.users.length):
            user_id = str(i + 1)
            loc = str(random.randint(1, num_of_locs))
            datapoint = Datapoint(user_id, self.time_stamp, loc)
            if loc in self.his.keys():
                self.his[loc] += 1
            else:
                self.his[loc] = 1
            self.data.append(datapoint)
            self.length += 1
        #print(self.his)

    def add_datapoint(self, datapoint):
        if self.users.check_id(datapoint.user_id):
            self.data.append(datapoint)
            if datapoint.loc in self.his.keys():
                self.his[datapoint.loc] += 1
            else:
                self.his[datapoint.loc] = 1
                self.num_of_locs += 1
            self.length += 1
        else:
            print('user not in db.Users')

    def list_budgets(self):
        budgets = {self.data[i].user_id: self.data[i].budget for i in range(self.length)}
        return budgets

    def find_min_budget(self):
        return min(self.list_budgets().values())

    def find_max_budget(self):
        return max(self.list_budgets().values())

    def find_larger_budget(self, proportion):
        return sorted(self.list_budgets().values())[int(self.length * proportion - 1)]

    def uni_loss(self, loss):
        for i in range(self.length):
            if self.data[i].budget >= loss:
                self.data[i].loss = loss
            else:
                print("error")
                time.time()

    def loss(self, loss):
        for i in range(self.length):
            self.data[i].loss = loss

    def print_losses(self):
        loss = []
        for i in range(self.length):
            loss.append(self.data[i].loss)

        print(loss)

    def print_indexes(self):
        indexes = []
        for i in range(self.length):
            indexes.append(id(self.data[i]))
        print(indexes)


    def uni_pro_loss(self, proportion=0.5):
        for i in range(self.length):
            self.data[i].loss = self.data[i].budget * proportion

    def avg_loss(self):
        return sum([self.data[i].loss for i in range(self.length)]) / self.length


class Databases:
    def __init__(self):
        self.length = 0
        self.sub_bases = []

    def empty_loss(self):
        for i in range(self.length):
            self.sub_bases[i].empty_loss()

    def add_db(self, db):
        self.sub_bases.append(db)
        self.length += 1

    def find_db(self, time_stamp):
        for i in range(self.length):
            if self.sub_bases[i].time_stamp == time_stamp:
                return self.sub_bases[i]

    def uni_alloc(self, time_stamp):
        db = self.find_db(time_stamp)
        users = db.users

        for i in range(db.length):
            user_id = db.data[i].user_id
            w = users.find_w_by_id(user_id)
            bound = users.find_bound_by_id(user_id)
            db.data[i].budget = bound / w

    def ap_alloc(self, time_stamp):
        db = self.find_db(time_stamp)
        users = db.users
        pre_db = self.find_db(time_stamp - 1)
        if type(pre_db) == Database:
            for i in range(db.length):
                user_id = db.data[i].user_id
                w = users.find_w_by_id(user_id)
                bound = users.find_bound_by_id(user_id)
                pre_budget = pre_db.data[i].budget
                pre_loss = pre_db.data[i].loss
                db.data[i].budget = bound / w + pre_budget - pre_loss

                losses = 0.0
                for j in range(w - 1):
                    if time_stamp - 1 - j >= 1:
                        pre_db = self.find_db(time_stamp - 1 - j)
                        losses += pre_db.data[i].loss
                remaining = bound - losses
                #print(remaining)
                db.data[i].budget = min(db.data[i].budget, remaining)

        else:
            for i in range(db.length):
                user_id = db.data[i].user_id
                w = users.find_w_by_id(user_id)
                bound = users.find_bound_by_id(user_id)
                db.data[i].budget = bound / w

    def proportion_alloc(self, time_stamp, proportion=0.8):
        db = self.find_db(time_stamp)
        users = db.users

        for i in range(db.length):
            user_id = db.data[i].user_id
            w = users.find_w_by_id(user_id)
            bound = users.find_bound_by_id(user_id)
            losses = 0.0

            for j in range(w - 1):
                if time_stamp - 1 - j >= 1:
                    pre_db = self.find_db(time_stamp - 1 - j)
                    losses += pre_db.data[i].loss
                    # if proportion == 1.0:
                    #     print("budget:"+str(pre_db.data[i].budget))
                    #     print("loss:" + str(pre_db.data[i].loss))
                    if pre_db.data[i].budget < pre_db.data[i].loss:
                        print("budget error")
                        print(pre_db.data[i].budget)
                        print(pre_db.data[i].loss)
                    #     print(time_stamp)
                    #     print(pre_db.find_min_budget())
                    #     print(pre_db.data[i].budget)
                    #     print(pre_db.data[i].loss)

            remaining = bound - losses
            if remaining < 0.0:
                print("remaining error")
                print(remaining)
                remaining = 0.0
            # if proportion == 1.0 and remaining < 0.0:
            #     print(time_stamp)
            #     print(remaining)
            db.data[i].budget = remaining * proportion

    def seize_the_moment(self, time_stamp):
        self.proportion_alloc(time_stamp, 1.0)

    def signal(self, time_stamp):
        db = self.find_db(time_stamp)
        users = db.users

        for i in range(db.length):
            user_id = db.data[i].user_id
            w = users.find_w_by_id(user_id)
            bound = users.find_bound_by_id(user_id)
            losses = 0.0

            for j in range(w - 1):
                if time_stamp - 1 - j >= 1:
                    pre_db = self.find_db(time_stamp - 1 - j)
                    losses += pre_db.data[i].loss


            remaining = bound - losses

            pro = 1.0
            if time_stamp > 1:
                pre_db = self.find_db(time_stamp - 1)
                if pre_db.data[i].loss == pre_db.data[i].budget:
                    pro = 0.5
            db.data[i].budget = remaining * pro

    def adv_signal(self, time_stamp):
        db = self.find_db(time_stamp)
        users = db.users

        for i in range(db.length):
            user_id = db.data[i].user_id
            w = users.find_w_by_id(user_id)
            bound = users.find_bound_by_id(user_id)
            losses = 0.0

            for j in range(w - 1):
                if time_stamp - 1 - j >= 1:
                    pre_db = self.find_db(time_stamp - 1 - j)
                    losses += pre_db.data[i].loss

            remaining = bound - losses

            if time_stamp > 1:
                signal_count = 0
                for j in range(1, time_stamp):
                    pre_db = self.find_db(j)
                    if pre_db.data[i].loss == pre_db.data[i].budget:
                        signal_count += 1

                pro = 1.0 - signal_count / (time_stamp - 1) * 0.5
            else:
                pro = 1.0
            db.data[i].budget = remaining * pro

    def compute_remaining(self, time_stamp):
        db = self.find_db(time_stamp)
        users = db.users
        remaining = []
        for i in range(db.length):
            user_id = db.data[i].user_id
            w = users.find_w_by_id(user_id)
            bound = users.find_bound_by_id(user_id)
            losses = 0.0

            for j in range(w - 1):
                if time_stamp - 1 - j >= 1:
                    pre_db = self.find_db(time_stamp - 1 - j)
                    losses += pre_db.data[i].loss
                    # if proportion == 1.0:
                    #     print("budget:"+str(pre_db.data[i].budget))
                    #     print("loss:" + str(pre_db.data[i].loss))
                    if pre_db.data[i].budget < pre_db.data[i].loss:
                        print("budget error")
            remaining.append(bound - losses)
        db.remaining = remaining


class Query:
    def __init__(self, db):
        self.D = db
        self.price = 0.0
        self.min_var = 0.0
        self.var = 0.0
        self.his = db.his.copy()
        self.answer = -1

    def choose_var_random(self, proportion=2.0):
        self.var = random.uniform(self.min_var, proportion * self.min_var)

    def choose_var_range(self, min=0.0, max=100.0):
        if max < min:
            self.var = float("inf")
            return
        else:
            self.var = random.uniform(min, max)
            return

    def choose_var(self, var):
        if var >= self.min_var:
            self.var = var
            return True
        else:
            self.var = self.min_var
            return False

    def quote(self, proportion=1.0):
        price = 0.0
        count = 0
        for i in range(self.D.length):
            user_id = self.D.data[i].user_id
            c = self.D.users.find_c_by_id(user_id)
            loss = self.D.data[i].loss

            # if c == 1.0:
            #     price += loss * 1.5
            #     count = count + 1
            #     # print(count)
            #     continue
            # if c == 2.0:
            #     price += math.exp(loss) -1
            #     continue
            # if c == 0.5:
            #     price += loss ** 0.5

            if c == 'B':
                price += 2 * loss ** 0.5
            if c == 'C1':
                price += loss + loss ** 0.5
            if c == 'C2':
                price += 1.5 * loss + 0.5 * loss ** 0.5
            if c == 'L':
                price += 2 * loss
            if c == 'S':
                price += math.exp(loss) - 1
        self.price = price


class Measurement:
    def __init__(self):
        self.length = 0
        self.queries = []
        self.var = []
        self.MSE = []
        self.MAE = []
        self.TIMES = 100

    def add_query(self, query):
        self.queries.append(query)
        self.length += 1

    def sort_query(self):
        temp = [(self.queries[i], self.queries[i].price) for i in range(self.length)]
        temp = sorted(temp, key=lambda x: x[1])
        print(temp)
        self.queries = [temp[i][0] for i in range(self.length)]

    def measure(self, mech):
        self.var = []
        self.MSE = []
        self.MAE = []

        self.sort_query()
        for j in range(self.length):
            print(j)
            answer_list = []
            query = self.queries[j]
            for i in range(self.TIMES):
                mech.perturb(query)
                answer_list.append(query.answer)

            avg = sum(answer_list) / len(answer_list)
            var = 0.0
            for i in range(len(answer_list)):
                var = (answer_list[i] - avg)**2
            var = var / len(answer_list)
            self.var.append(var)

            # truth = []
            # for k in range(query.D.num_of_locs):
            #     try:
            #         truth.append(query.D.answer[str(k + 1)])
            #     except KeyError:
            #         truth.append(0.0)
            #
            # truth = np.array(truth)
            # self.MSE.append(np.sum(np.square(answers - truth)) / answers.size)
            # self.MAE.append(np.sum(np.abs(answers - truth)) / answers.size)




















