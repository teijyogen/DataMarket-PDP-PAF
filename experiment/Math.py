import operator
import random
import sys
import math
import scipy.optimize as opt
import json

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

def proTwo(pattern, thetaL=False, thetaU=False):
    wlpath="C:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\MathKernel.exe"
    session = WolframLanguageSession(wlpath)
    n = len(pattern)
    exp = '''frac[x_, k_] = (Exp[k*x] - 1)/(Exp[x] - 1)*(1 - (Exp[k*x] - 1)/(Exp[x] - 1));
             U[x_,p_]:=2/x^2'''
    for i in range(n):
        exp = exp + '''+frac[x,p[[%d]]]''' % (i+1)
    exp = exp + ''';
    '''

    exp = exp + '''pattern={'''
    for i in range(n):
        exp = exp + str(pattern[i])
        # exp = exp + '''k%d''' % (i+1)
        if i != n-1:
            exp = exp + ''','''
    exp = exp + '''};
    '''

    exp = exp + '''g[x_]:=U[x,pattern]*D[U[x,pattern],{x,2}]-2*(D[U[x,pattern],x])^2;
    '''
    if thetaL and thetaU:
        exp = exp + '''FindMaximum[{g[x],x>%f,x<%f},{x,%f}]
                    ''' % (thetaL, thetaU, thetaL)
    elif thetaL:
        exp = exp + '''FindMaximum[{g[x],x>%f},{x,%f}]
                                ''' % (thetaL, thetaL)
    elif thetaU:
        exp = exp + '''FindMaximum[{g[x],x>0,x<%f},{x,0.001}]
                                ''' % (thetaU)
    else:
        exp = exp + '''FindMaximum[{g[x],x>0},{x,0.001}]
        '''

    # exp = exp + '''NMinimize[{NMaxValue[{g[x,pattern],x>0},x],k1>=2 && k2>=2 && k3>=2},{k1,k2,k3}]'''

    expr = wlexpr(exp)
    result = session.evaluate(expr)
    if float(result[0]) > 0:
        return False

    start = float(tuple(result[1][0])[1])
    if thetaL and thetaU:
        exp = '''FindMaximum[{g[x],x>%f,x<%f},{x,%f}]
                    ''' % (thetaL, thetaU, start)
    elif thetaL:
        exp = '''FindMaximum[{g[x],x>%f},{x,%f}]
                                ''' % (thetaL, start)
    elif thetaU:
        exp = '''FindMaximum[{g[x],x>0,x<%f},{x,%f}]
                                ''' % (thetaU, start)
    else:
        exp = '''FindMaximum[{g[x],x>0},{x,%f}]
        ''' % (start)

    expr = wlexpr(exp)
    result = session.evaluate(expr)
    print(result)
    if float(result[0]) > 0:
        return False

    return True

def proOne(pattern):
    wlpath="C:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\MathKernel.exe"
    session = WolframLanguageSession(wlpath)
    n = len(pattern)
    exp = '''frac[x_, k_] = (Exp[k*x] - 1)/(Exp[x] - 1)*(1 - (Exp[k*x] - 1)/(Exp[x] - 1));
             U[x_,p_]:=2/x^2'''
    for i in range(n):
        exp = exp + '''+frac[x,p[[%d]]]''' % (i+1)
    exp = exp + ''';
    '''

    exp = exp + '''pattern={'''
    for i in range(n):
        exp = exp + str(pattern[i])
        # exp = exp + '''k%d''' % (i+1)
        if i != n-1:
            exp = exp + ''','''
    exp = exp + '''};
    '''

    exp = exp + '''g[x_]:=D[U[x,pattern],x];
    '''
    exp = exp + '''FindMaximum[{g[x],x>0},{x,0.001}]
    '''

    # exp = exp + '''NMinimize[{NMaxValue[{g[x,pattern],x>0},x],k1>=2 && k2>=2 && k3>=2},{k1,k2,k3}]'''

    expr = wlexpr(exp)
    result = session.evaluate(expr)[0]
    print(result)
    if float(result) < 0:
        return True
    return False

# def proThree(pattern, thetaL=False):
#     if thetaL == False:
#         return True
#
#     wlpath="C:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\MathKernel.exe"
#     session = WolframLanguageSession(wlpath)
#     n = len(pattern)
#     exp = '''frac[x_, k_] = (Exp[k*x] - 1)/(Exp[x] - 1)*(1 - (Exp[k*x] - 1)/(Exp[x] - 1));
#              U[x_,p_]:=2/x^2'''
#     for i in range(n):
#         exp = exp + '''+frac[x,p[[%d]]]''' % (i+1)
#     exp = exp + ''';
#     '''
#
#     exp = exp + '''pattern={'''
#     for i in range(n):
#         exp = exp + str(pattern[i])
#         # exp = exp + '''k%d''' % (i+1)
#         if i != n-1:
#             exp = exp + ''','''
#     exp = exp + '''};
#     '''
#
#
#     exp = exp + '''g[x_]:=2*U[2*x,pattern] - U[x,pattern];
#         '''
#
#     exp = exp + '''g[%f] <= 0
#             ''' %(thetaL)
#
#     expr = wlexpr(exp)
#     result = session.evaluate(expr)
#
#     return result

def proThree(pattern, thetaL=False, thetaU=False):
    if thetaL == False and thetaU==False:
        return True

    wlpath="C:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\MathKernel.exe"
    session = WolframLanguageSession(wlpath)
    n = len(pattern)
    exp = '''frac[x_, k_] = (Exp[k*x] - 1)/(Exp[x] - 1)*(1 - (Exp[k*x] - 1)/(Exp[x] - 1));
             U[x_,p_]:=2/x^2'''
    for i in range(n):
        exp = exp + '''+frac[x,p[[%d]]]''' % (i+1)
    exp = exp + ''';
    '''

    exp = exp + '''pattern={'''
    for i in range(n):
        exp = exp + str(pattern[i])
        # exp = exp + '''k%d''' % (i+1)
        if i != n-1:
            exp = exp + ''','''
    exp = exp + '''};
    '''


    exp = exp + '''g[x_]:=U[x+%f,pattern] - 1/(1/U[x,pattern]+1/U[%f,pattern]);
        ''' % (thetaL, thetaL)

    exp = exp + '''FindMaximum[{g[x],x>%f},{x,%f}]
            ''' % (thetaL, thetaU-thetaL)

    expr = wlexpr(exp)
    result = session.evaluate(expr)[0]

    print(result)
    if float(result) < 0:
        return True
    return False

def patternSearch(initP, thetaL=False, thetaU=False, reverse=False):
    pattern = initP * 1
    startP = initP * 1
    endP = initP * 1

    if reverse:
        for i in range(len(endP)):
            endP[i] = 1.0
    else:
        for i in range(len(endP)):
            if endP[i] != 1.0:
                endP[i] = 0.0

    print(pattern)

    while True:
        preP = pattern * 1
        print(pattern)
        if proOne(pattern) and proTwo(pattern, thetaL, thetaU) and proThree(pattern, thetaL, thetaU):
            endP = pattern * 1
            for i in range(len(pattern)):
                if pattern[i] != 1.0:
                    pattern[i] = (pattern[i] + startP[i]) / 2
        else:
            startP = pattern * 1
            for i in range(len(pattern)):
                if pattern[i] != 1.0:
                    pattern[i] = (pattern[i] + endP[i]) / 2
        minus = [pattern[i]-preP[i] for i in range(len(pattern))]
        if abs(sum(minus)) < 0.0001 * len(pattern):
            return endP

def patterning(pattern, budgets):
    maxB = max(budgets)
    for i in range(len(budgets)):
        if budgets[i] < maxB * pattern[i]:
            maxB = budgets[i] / pattern[i]
    adjusted = []
    for i in range(len(budgets)):
        adjusted.append(maxB * pattern[i])
    return adjusted



