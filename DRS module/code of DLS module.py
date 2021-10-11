import numpy as np
import matplotlib.pyplot as plt
import math 
import pandas as pd
from scipy.optimize import minimize,brute,optimize
%matplotlib inline


def processData(cricketData):
    cricketData = cricketData[cricketData['Error.In.Data'] == 0]
    cricketModifiedData = cricketData[['Match','Innings','Over','Innings.Total.Runs','Runs.Remaining','Wickets.in.Hand','Total.Overs','new.game']]
    cricketModifiedData = cricketModifiedData.rename({'Runs.Remaining': 'TotalRemainingRuns','Innings.Total.Runs': 'TotalInningsRuns','Wickets.in.Hand':'WicketsInHand','Total.Overs':'TotalOvers','new.game': 'NewGame'},axis = 1)
    cricketModifiedData = cricketModifiedData[cricketModifiedData['Innings'] == 1]
    cricData = cricketModifiedData[cricketModifiedData['NewGame'] == 1]
    cricData['TotalRemainingRuns'] = cricData['TotalInningsRuns']
    cricData['Over'] = cricData['Over'] - 1 
    cricData['WicketsInHand'] = 10
    cricData = cricData[cricData['Match'] != 410557]
    cricketModifiedData = pd.concat([cricketModifiedData, cricData], ignore_index=True)
    cricketModifiedData['OversRemaining'] = cricketModifiedData['TotalOvers'] - cricketModifiedData['Over']

    return cricketModifiedData

# Function to calculate run with model with given b-value and max run value
def runPrediction(Z0, b, u):
    Z0u = Z0 * (1 - np.exp(-b * u))
    return Z0u

#Function to calculate the sum squared error of the predicted run and given run
def ErrorFunction(params,*cricArg):
    z0 = params[0]
    b = params[1]
    wkt = cricArg[1]
    ActualRun = cricArg[0]
    TotalError = 0
    for u in range(1,51):
        totalRuns = ActualRun[(ActualRun['OversRemaining'] == u)]
        predictedRuns = runPrediction(z0,b,u)
        SumSquaredError = np.sum(np.square(predictedRuns - np.array(totalRuns['TotalRemainingRuns'])))
        TotalError = TotalError + SumSquaredError
    return TotalError

def DuckworthLewis20Params(cricketModifiedData):
    optimizedError = 0
    optimizedZ0 = []
    optimizedB = []
    for wkt in range(1,11):
        rranges = (slice(1,300,10),slice(0,1,0.1))
        runsPerWicket = cricketModifiedData[cricketModifiedData['WicketsInHand'] == wkt]
        solution = brute(ErrorFunction, rranges, args=(runsPerWicket,wkt),
                                finish=optimize.fmin)
        optimizedZ0.append(solution[0])
        optimizedB.append(solution[1])
        optimizedError = optimizedError + ErrorFunction(solution,runsPerWicket,wkt)

    # Error per point
    print("-----Error Per point (MSE)--------\n")
    print(optimizedError/len(cricketModifiedData))
    return optimizedZ0,optimizedB

def plotErrorFunc1(z,b):
    plt.figure(figsize=(8,8), dpi=100)
    x = np.arange(51)
    plt.xlim(0,50)
    plt.ylim(0,100)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('Percentage of resource remaining')
    x = np.arange(0, 51, 1)
    modified_x = np.array([50.0 - i for i in x])
    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#234a21', '#876e34', '#a21342']

    for i in range(10):
        y = 100*runPrediction(z[i], b[i], modified_x)/runPrediction(z[9],b[9],50)
        plt.plot(x, y, c=color_list[i], label='Z['+str(i+1)+']')
        plt.legend()

    y_linear = [-2*i + 100 for i in x]
    plt.plot(x, y_linear, '#5631a2')


def runPrediction2(Z0, l, u,wkt):
    Z0u = Z0[wkt] * (1 - np.exp((-l * u)/Z0[wkt]))
    return Z0u


def plotErrorFunc2(z,l):
    plt.figure(figsize=(8,8), dpi=100)
    x = np.arange(51)
    plt.xlim(0,50)
    plt.ylim(0,100)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('Percentage of resource remaining')
    x = np.arange(0, 51, 1)
    modified_x = np.array([50.0 - i for i in x])
    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#234a21', '#876e34', '#a21342']
    z0 = [0]
    z0.extend(z) #for easy handling of indices
    for i in range(1,11):
        y = 100 * runPrediction2(z0, l, modified_x,i) / runPrediction2(z0,l,50,10)
        plt.plot(x, y, c=color_list[i-1], label='Z['+str(i)+']')
        plt.legend()

    y_linear = [-2*i + 100 for i in x]
    plt.plot(x, y_linear, '#5631a2')

def ErrorFunctionForQ2(params,cricArg):
    z0 = [0] + list(params[0:10])
    l = params[-1]
    # wkt = cricArg[1]
    ActualRun = cricArg
    TotalError = 0
    for u in range(1,51):
        for wkt in range(1,11):
            totalRuns = ActualRun[(ActualRun['OversRemaining'] == u) & (ActualRun['WicketsInHand'] == wkt)]
            predictedRuns = runPrediction2(z0,l,u,wkt)
            SumSquaredError = np.sum(np.square(predictedRuns - np.array(totalRuns['TotalRemainingRuns'])))
            TotalError = TotalError + SumSquaredError
    return TotalError

def DuckworthLewis11Params(cricketModifiedData,InitalMRPW_initialBValForQ2):
    sol = minimize(ErrorFunctionForQ2, InitalMRPW_initialBValForQ2, 
               args=(cricketModifiedData),
               method='L-BFGS-B')
    
    print("-------Error per point for Q2 (MSE)---------\n")
    print(sol.fun/len(cricketModifiedData))
    return sol.x[0:10],sol.x[-1]

if __name__ == '__main__':
    cricketData = pd.read_csv('/content/drive/MyDrive/04_cricket_1999to2011.csv')
    cricketModifiedData = processData(cricketData)

    z0 ,b = DuckworthLewis20Params(cricketModifiedData)
    print("----------------parameters of 1st function---------\n")
    print(z0)
    print('\n')
    print(b)

    plotErrorFunc1(z0,b)

    InitalMRPW_initialBValForQ2 = []
    InitalMRPW_initialBValForQ2.extend(z0)
    InitalMRPW_initialBValForQ2.extend([5])


    Z0, L = DuckworthLewis11Params(cricketModifiedData,InitalMRPW_initialBValForQ2)
    print("----------------parameters of 2nd function---------\n")
    print(Z0)
    print('\n')
    print(L)

    plotErrorFunc2(z0,L)

