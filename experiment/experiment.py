from tqdm import tqdm
from Laplace import *
from Sample import *
import matplotlib.pyplot as plt
import numpy as np


class Experiment:
    def __init__(self, user_num=200, loc_num=20, db_num=100, w=6, c=0.5, partial=False, thetaL=1.5, thetaU=10.0, changed_bound=0.5):
        self.USER_NUM = user_num
        self.LOC_NUM = loc_num
        self.T = db_num
        self.W = w
        self.C = c
        self.users = Users()
        self.dbs = Databases()
        self.partial = partial
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.changed_bound = changed_bound
        self.max_var = float('inf')
        self.min_var = 0.0
        self.sup = False
        self._generate_datasets()

    def _generate_datasets(self):
        if self.partial:
            filename = r'../dataset/%d_ptp_thetaL%.1f_thetaU%.1f.json'%(self.USER_NUM, self.thetaL, self.thetaU)
        else:
            if self.changed_bound < 2.0:
                filename = r'../dataset/%d_minb%.1f.json'%(self.USER_NUM, self.changed_bound)
            else:
                filename = r'../dataset/%d_maxb%.1f.json' % (self.USER_NUM, self.changed_bound)

        # filename = r'%d_test_partial.json' % (self.USER_NUM)

        users = json.load(open(filename))

        for i in range(self.USER_NUM):
            id = str(i+1)
            user = users[id]
            # w = self.W - user['w']
            # bound = user['bound'] * w
            w = self.T
            bound = user['bound']
            c = user['c']
            self.users.add_user(User(id, w, bound, c))
        # print(type(users['pattern'])
        self.users.ab_pattern = users['pattern']

        if self.partial:
            self.users.partial_pattern = users['partial']
            pattern = self.users.partial_pattern
            def f(x):
                expr = 2 / x ** 2
                for i in range(len(pattern)):
                    expr = expr + ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)) * (1 - ((math.exp(pattern[i] * x) - 1) / (math.exp(x) - 1)))
                return expr

            if users['thetaL'] != 0.0:
                # print(f(users['thetaL']))
                self.max_var = f(users['thetaL'])
            self.min_var = f(users['thetaU'])

            self.users.adjusted_pattern = self.users.partial_pattern

        else:
            self.users.adjusted_pattern = self.users.ab_pattern

        for i in range(self.T):
            time_stamp = i + 1
            db = Database(time_stamp, self.users)
            db.generate_data(self.LOC_NUM)
            self.dbs.add_db(db)
            #print(db.find_min_budget())

    def sup_for_utp(self, thetaU):
        self.min_var = 2 * (1 / thetaU) ** 2
        self.max_var = float('inf')
        self.change_compensation_mode('supper-additive')

    def change_compensation_mode(self, mode='selectable'):
        if mode=='selectable':
            for i in range(self.USER_NUM):
                c = random.choice(['B', 'C1', 'C2', 'L'])
                self.users.users[i].c = c

        elif mode=='unselectable':
            for i in range(self.USER_NUM):
                # c = str(self.users.users[i].bound)
                # self.users.users[i].c = c
                bound = self.users.users[i].bound
                if bound == 0.5:
                    c = random.choice(['B'])
                elif bound == 2.0:
                    c = random.choice(['C1'])
                elif bound == 4.0:
                    c = random.choice(['C2'])
                else:
                    c = random.choice(['L'])
                self.users.users[i].c = c

        elif mode=='semi-selectable':
            for i in range(self.USER_NUM):
                bound = self.users.users[i].bound
                if bound == 0.5:
                    c = random.choice(['B'])
                elif bound == 2.0:
                    c = random.choice(['B', 'C1'])
                elif bound == 4.0:
                    c = random.choice(['C1', 'L'])
                else:
                    c = random.choice(['L'])
                self.users.users[i].c = c

        elif mode=='supper-additive':
            for i in range(self.USER_NUM):
                self.users.users[i].c = 'S'
        return



    def change_partial(self, partial):
        if partial == True:
            self.partial = True
            self.users.adjusted_pattern = self.users.partial_pattern
        else:
            self.partial = False
            self.users.adjusted_pattern = self.users.ab_pattern

    def add_exceptional_user(self, bound):
        self.USER_NUM += 1
        user_id = str(self.users.length + 1)
        w = random.randint(self.W - 1, self.W + 1)
        c = self.C
        user = User(user_id, w, bound, c)
        self.users.add_user(user)

        for i in range(self.T):
            time_stamp = i + 1
            loc = str(random.randint(1, self.LOC_NUM))
            datapoint = Datapoint(user_id, time_stamp, loc)
            self.dbs.sub_bases[i].add_datapoint(datapoint)


    def avg_loss_used(self, mech, var, alloc):
        avg_loss_used_list = []

        for i in range(self.dbs.length):
            # print(i)
            time_stamp = i + 1
            self.dbs.compute_remaining(time_stamp)
            db = self.dbs.sub_bases[i]
            #db.print_losses()
            alloc(time_stamp)
            query = Query(db)
            query.var = var
            mech.min_var(query)
            query.var = 0.0
            if var < query.min_var or var < self.min_var:
                # print("rejected")
                break
            if self.partial == True and var > self.max_var:
                break
            query.choose_var(var)
            mech.perturb(query)
            avg_loss_used_list.append(db.avg_loss())

        self.empty_loss()

        if len(avg_loss_used_list) == 0:
            return 0

        # return sum(avg_loss_used_list) / len(avg_loss_used_list)
        return sum(avg_loss_used_list)

    def avg_loss_used_random(self, mech, pro, alloc):
        avg_loss_used_list = []

        for i in range(self.dbs.length):
            time_stamp = i + 1
            db = self.dbs.sub_bases[i]
            alloc(time_stamp)
            query = Query(db)
            mech.min_var(query)

            if math.isinf(query.min_var) or math.isnan(query.min_var):
                break

            # if self.partial == True:
            #     max_var = min(self.max_var, max_var)

            query.choose_var_random(pro)

            if math.isinf(query.var) or math.isnan(query.var):
                break

            if query.var < query.min_var or query.var < self.min_var:
                # print("rejected")
                break

            mech.perturb(query)
            avg_loss_used_list.append(db.avg_loss())
        self.empty_loss()
        # return sum(avg_loss_used_list) / len(avg_loss_used_list)

        if len(avg_loss_used_list) == 0:
            return 0.0

        return sum(avg_loss_used_list)

    def avg_loss_used_pro(self, mech, pro, max_var=20.0):
        round = 100
        avg_loss = 0.0

        for j in range(round):
            avg_loss_used_list = []

            for i in range(self.dbs.length):
                time_stamp = i + 1
                db = self.dbs.sub_bases[i]
                self.dbs.proportion_alloc(time_stamp, pro)
                query = Query(db)
                mech.min_var(query)
                if math.isinf(query.min_var) or math.isnan(query.min_var):
                    break

                # if self.partial == True:
                #     max_var = min(self.max_var, max_var)

                query.choose_var_range(query.min_var, max_var)

                if math.isinf(query.var) or math.isnan(query.var):
                    break

                if query.var < query.min_var or query.var < self.min_var:
                    # print("rejected")
                    break


                if self.partial == True:
                    if query.var > self.max_var:
                        continue

                mech.perturb(query)
                avg_loss_used_list.append(db.avg_loss())
            self.empty_loss()
            if len(avg_loss_used_list) == 0:
                return 0.0
            avg_loss += sum(avg_loss_used_list)

        return avg_loss / round

    def answered_queries_pro(self, mech, pro, max_var=10.0):
        round = 10
        count = 0

        for j in range(round):

            for i in range(self.dbs.length):
                time_stamp = i + 1
                db = self.dbs.sub_bases[i]
                self.dbs.proportion_alloc(time_stamp, pro)
                query = Query(db)
                mech.min_var(query)
                # print(query.min_var)
                if math.isinf(query.min_var) or math.isnan(query.min_var):
                    # print("rejected")
                    break

                # if self.partial == True:
                #     max_var = min(self.max_var, max_var)

                query.choose_var_range(query.min_var, max_var)

                if math.isinf(query.var) or math.isnan(query.var):
                    # print("rejected")
                    break

                if query.var < query.min_var or query.var < self.min_var:
                    # print("rejected")
                    break
                if self.partial == True:
                    if query.var > self.max_var:
                        continue

                mech.perturb(query)
                count += 1
            self.empty_loss()

        return count / round / self.dbs.length

    def var_price(self, mech, var, alloc):
        self.dbs.compute_remaining(1)
        db = self.dbs.sub_bases[0]
        alloc(1)
        query = Query(db)
        mech.min_var(query)
        if var < query.min_var:
            return -1
        # print(self.partial)
        if self.partial == True and var > self.max_var:
            return -1

        query.choose_var(var)
        mech.perturb(query)
        self.empty_loss()
        return query.price

    def var_loss(self, mech, var, alloc):
        db = self.dbs.sub_bases[0]
        alloc(1)
        query = Query(db)
        mech.min_var(query)
        if var < query.min_var:
            return -1
        # print(self.partial)
        if self.partial == True and var > self.max_var:
            return -1

        query.choose_var(var)
        mech.perturb(query)
        loss = db.avg_loss()
        self.empty_loss()
        return loss

    def answered_queries(self, mech, var, alloc):
        count = 0

        for i in range(self.dbs.length):
            # print(i)
            time_stamp = i + 1
            self.dbs.compute_remaining(time_stamp)
            db = self.dbs.sub_bases[i]
            #db.print_losses()
            alloc(time_stamp)
            query = Query(db)
            query.var = var
            mech.min_var(query)
            query.var = 0.0
            if var < query.min_var:
                # print("rejected")
                break
            if self.partial == True and var > self.max_var:
                break
            query.choose_var(var)
            mech.perturb(query)
            count = count + 1
            # print(db.avg_loss())

        self.empty_loss()

        return count / self.dbs.length

    def arbitrage(self, mech, var, M=10):
        db = self.dbs.sub_bases[0]
        query = Query(db)
        true_price = mech.quote_var(query, var)


        ab_price = float('inf')
        for i in range(2, M+1):
            m = i * 1.0
            ab_var = m * var
            if ab_var > self.max_var:
                break
            if ab_var < self.min_var:
                break
            ab_price = min(mech.quote_var(query, ab_var) * m, ab_price)
        return [true_price, ab_price]

    def arbitrage_rate(self, mech, var, M=10):
        true_price, ab_price = self.arbitrage(mech, var, M=M)

        if true_price == 0.0:
            return float('nan')
        return ab_price / true_price

    def min_var(self, mech, alloc):
        db = self.dbs.sub_bases[0]
        alloc(1)
        query = Query(db)
        mech.min_var(query)
        self.empty_loss()
        return query.min_var

    def measurement(self, mech, metrics, pro=2.0):
        for i in range(self.dbs.length):
            time_stamp = i + 1
            db = self.dbs.sub_bases[i]

            self.dbs.uni_alloc(time_stamp)
            query = Query(db)
            mech.min_var(query)
            query.choose_var_random(pro)
            mech.perturb(query)
            metrics.add_query(query)

        return metrics

    def empty_loss(self):
        self.dbs.empty_loss()

def rq1_ploss(bounds=[]):

    loss_lap_list = []
    loss_sam_list = []

    for bound in bounds:

        expt = Experiment(changed_bound=bound)
        mech = Laplace()
        loss_lap_list.append(expt.avg_loss_used_random(mech, 1.0, expt.dbs.proportion_alloc))

        expt = Experiment(changed_bound=bound)
        mech = Sample(Laplace())
        loss_sam_list.append(expt.avg_loss_used_random(mech, 1.0, expt.dbs.proportion_alloc))

    # print(loss_lap_list)
    # print(loss_sam_list)

    labels = [str(bound) for bound in bounds]

    plt.subplot()
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width/2, loss_lap_list, width, label='UT', color='red')
    plt.bar(x + width/2, loss_sam_list, width, label='PT', color='blue')

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.xticks(x, labels=labels)
    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('minimum privacy loss bound', font2)
    plt.ylabel('average traded privacy loss', font2)
    # plt.title("UniformTrading vs. PersonalizedTrading", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()

def rq2_pab(START, END, STEP):
    mech1 = Laplace()
    mech2 = Sample(Laplace())
    mech2.flexible = False
    mech3 = Sample(Laplace())
    mech3.flexible = False
    mech4 = Sample(Laplace())
    mech4.flexible = False
    mech5 = Sample(Laplace())
    mech5.flexible = False
    mech6 = Sample(Laplace())
    mech6.flexible = False


    var_list_ut = []
    avg_loss_ut = []
    var_list_pt = []
    avg_loss_pt = []
    var_list_ptp1 = []
    avg_loss_ptp1 = []
    var_list_ptp2 = []
    avg_loss_ptp2 = []
    var_list_ptp3 = []
    avg_loss_ptp3 = []
    var_list_ptp4 = []
    avg_loss_ptp4 = []

    expt = Experiment()
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_pt.append(var)
        avg_loss_pt.append(expt.avg_loss_used_pro(mech2, 0.8, var))

    expt = Experiment(partial=True, thetaL=0.5)
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_ptp1.append(var)
        avg_loss_ptp1.append(expt.avg_loss_used_pro(mech3, 0.8, var))

    expt = Experiment(partial=True, thetaL=1.0)
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_ptp2.append(var)
        avg_loss_ptp2.append(expt.avg_loss_used_pro(mech4, 0.8, var))

    expt = Experiment(partial=True, thetaL=1.5)
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_ptp3.append(var)
        avg_loss_ptp3.append(expt.avg_loss_used_pro(mech5, 0.8, var))

    expt = Experiment(partial=True, thetaL=2.0)
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_ptp4.append(var)
        avg_loss_ptp4.append(expt.avg_loss_used_pro(mech6, 0.8, var))

    expt = Experiment()
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_ut.append(var)
        avg_loss_ut.append(expt.avg_loss_used_pro(mech1, 1.0, var))

    plt.plot(var_list_ut, avg_loss_ut, marker='x', color='red', label='UT')
    plt.plot(var_list_pt, avg_loss_pt, marker='v', color='blue', label='PT')
    plt.plot(var_list_ptp1, avg_loss_ptp1, marker='^', color='yellow', label='PTP/0.5')
    plt.plot(var_list_ptp2, avg_loss_ptp2, marker='o', color='olivedrab', label='PTP/1.0')
    plt.plot(var_list_ptp3, avg_loss_ptp3, marker='s', color='teal', label='PTP/1.5')
    plt.plot(var_list_ptp4, avg_loss_ptp4, marker='*', color='violet', label='PTP/2.0')


    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('maximum worst-case variance', font2)
    plt.ylabel('average traded privacy loss', font2)
    # plt.title("Uniform vs. Personalized", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()

def rq2_res(START, END, n_steps):
    mech1 = Laplace()
    mech2 = Sample(Laplace())
    mech2.flexible = False
    mech3 = Sample(Laplace())
    mech3.flexible = True
    mech4 = Sample(Laplace())
    mech4.flexible = False
    mech5 = Sample(Laplace())
    mech5.flexible = True

    rate_list_lap = []
    avg_loss_lap = []
    rate_list_sam = []
    avg_loss_sam = []
    rate_list_sam_pt = []
    avg_loss_sam_pt = []
    rate_list_sam_plus = []
    avg_loss_sam_plus = []
    rate_list_sam_plus_pt = []
    avg_loss_sam_plus_pt = []

    step = (END - START) / n_steps

    expt = Experiment()
    for i in tqdm(range(0, n_steps + 1)):
        rate = START + i * step
        rate_list_sam.append(rate)
        avg_loss_sam.append(expt.avg_loss_used_pro(mech2, 1 - rate))

    expt = Experiment()
    expt.change_compensation_mode('semi-selectable')
    for i in tqdm(range(0, n_steps + 1)):
        rate = START + i * step
        rate_list_sam_pt.append(rate)
        avg_loss_sam_pt.append(expt.avg_loss_used_pro(mech3, 1 - rate))

    expt = Experiment(partial=True)
    for i in tqdm(range(0, n_steps + 1)):
        rate = START + i * step
        rate_list_sam_plus.append(rate)
        avg_loss_sam_plus.append(expt.avg_loss_used_pro(mech4, 1 - rate))

    expt = Experiment(partial=True)
    expt.change_compensation_mode('semi-selectable')
    for i in tqdm(range(0, n_steps + 1)):
        rate = START + i * step
        rate_list_sam_plus_pt.append(rate)
        avg_loss_sam_plus_pt.append(expt.avg_loss_used_pro(mech5, 1 - rate))


    expt = Experiment()
    for i in tqdm(range(0, n_steps + 1)):
        rate = START + i * step
        rate_list_lap.append(rate)
        avg_loss_lap.append(expt.avg_loss_used_pro(mech1, 1 - rate))

    plt.plot(rate_list_lap, avg_loss_lap, marker='x', color='red', label='UT')
    plt.plot(rate_list_sam, avg_loss_sam, marker='v', color='blue', label='PT')
    plt.plot(rate_list_sam_pt, avg_loss_sam_pt, marker='^', color='yellow', label='PT+PE')
    plt.plot(rate_list_sam_plus, avg_loss_sam_plus, marker='o', color='olivedrab', label='PTP')
    plt.plot(rate_list_sam_plus_pt, avg_loss_sam_plus_pt, marker='*', color='teal', label='PTP+PE')


    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('reservation rate', font2)
    plt.ylabel('average traded privacy loss', font2)
    # plt.title("Uniform vs. Personalized", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()

def rq3_pe(START, END, STEP, partial=False):
    mech1 = Laplace()
    mech2 = Sample(Laplace())
    mech2.flexible = False
    mech3 = Sample(Laplace())
    mech3.flexible = True
    mech4 = Sample(Laplace())
    mech4.flexible = True
    mech5 = Sample(Laplace())
    mech5.flexible = True

    var_list_ut = []
    avg_loss_ut = []
    var_list_pt = []
    avg_loss_pt = []
    var_list_pte_fully = []
    avg_loss_pte_fully = []
    var_list_pte_semi = []
    avg_loss_pte_semi = []
    var_list_pte_un = []
    avg_loss_pte_un = []

    expt = Experiment(partial=partial)
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_pt.append(var)
        avg_loss_pt.append(expt.avg_loss_used_pro(mech2, 0.8, var))

    expt = Experiment(partial=partial)
    expt.change_compensation_mode(mode='selectable')
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_pte_fully.append(var)
        avg_loss_pte_fully.append(expt.avg_loss_used_pro(mech3, 0.8, var))

    expt = Experiment(partial=partial)
    expt.change_compensation_mode(mode='semi-selectable')
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_pte_semi.append(var)
        avg_loss_pte_semi.append(expt.avg_loss_used_pro(mech4, 0.8, var))

    expt = Experiment(partial=partial)
    expt.change_compensation_mode(mode='unselectable')
    for i in tqdm(range(START, END)):
        var = i * STEP
        var_list_pte_un.append(var)
        avg_loss_pte_un.append(expt.avg_loss_used_pro(mech5, 0.8, var))

    if partial:
        plt.plot(var_list_pt, avg_loss_pt, marker='v', color='blue', label='PTP')
        plt.plot(var_list_pte_fully, avg_loss_pte_fully, marker='^', color='yellow', label='PTP+PE<selectable>')
        plt.plot(var_list_pte_semi, avg_loss_pte_semi, marker='o', color='olivedrab', label='PTP+PE<semi-selectable>')
        plt.plot(var_list_pte_un, avg_loss_pte_un, marker='s', color='teal', label='PTP+PE<unselectable>')
    else:
        plt.plot(var_list_pt, avg_loss_pt, marker='v', color='blue', label='PT')
        plt.plot(var_list_pte_fully, avg_loss_pte_fully, marker='^', color='yellow', label='PT+PE<selectable>')
        plt.plot(var_list_pte_semi, avg_loss_pte_semi, marker='o', color='olivedrab', label='PT+PE<semi-selectable>')
        plt.plot(var_list_pte_un, avg_loss_pte_un, marker='s', color='teal', label='PT+PE<unselectable')

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('maximum worst-case variance', font2)
    plt.ylabel('average traded privacy loss', font2)
    # plt.title("Uniform vs. Personalized", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()

def rq4_ab():
    START = 1
    END = 101
    expt = Experiment()
    expt.change_compensation_mode(mode='semi-selectable')
    # print(expt.max_var)
    var_list = []

    ab_rate_pt = []
    ab_rate_ut = []

    pt = Sample(Laplace())
    pt.flexible = False

    lap = Laplace()

    for i in tqdm(range(START, END)):
        var = i * 1.0
        var_list.append(var)

        ab_rate_pt.append(expt.arbitrage_rate(pt, var))
        ab_rate_ut.append(expt.arbitrage_rate(lap, var))


    plt.plot(var_list, ab_rate_ut, label='UT')
    plt.plot(var_list, ab_rate_pt, label='PT')

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('worst-case variance', font2)
    plt.ylabel('arbitrage rate', font2)
    # plt.title("UniformTrading", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()

def rq5_pab():
    START = 1
    END = 201

    var_pt = []
    var_ut = []
    ab_rate_pt = []
    ab_rate_ut = []

    pt = Sample(Laplace())
    pt.flexible = False

    lap = Laplace()

    expt = Experiment(partial=True)
    expt.change_compensation_mode(mode='semi-selectable')
    for i in tqdm(range(START, END)):
        var = i * 0.1
        var_pt.append(var)
        ab_rate_pt.append(expt.arbitrage_rate(pt, var))

    expt = Experiment(partial=True)
    expt.sup_for_utp(1.5)
    for i in tqdm(range(START, END)):
        var = i * 0.1
        var_ut.append(var)
        ab_rate_ut.append(expt.arbitrage_rate(lap, var))

    plt.plot(var_ut, ab_rate_ut, label='UTP')
    plt.plot(var_pt, ab_rate_pt, label='PTP')

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 18,
             }

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    plt.tick_params(labelsize=8)
    plt.legend(prop=font1)

    plt.xlabel('worst-case variance', font2)
    plt.ylabel('arbitrage rate', font2)
    # plt.title("UniformTrading", fontsize=18)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 400
    plt.show()
    plt.close()


'''Experiments present in the paper'''

# RQ1
# rq1_ploss([0.1, 0.5, 1.0, 1.5])
# rq1_ploss([7.0, 8.0, 9.0, 10.0])

# RQ2
# rq2_pab(1, 21, 1.0)
# rq2_res(0.0, 1.0, 20)

# RQ3
# rq3_pe(1, 21, 1.0)
# rq3_pe(1, 21, 1.0, partial=True)

# RQ4
# rq4_ab()

# RQ5
# rq5_pab()
