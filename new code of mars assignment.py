#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import math 
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# In[3]:


marsData = pd.read_csv('/home/talk2robots/Documents/Data Analytics/01_data_mars_opposition_updated.csv')


# In[8]:


marsHeliocentric_longitude = marsData.values[:, 5:9]
   # print( marsHeliocenric data)
print(marsHeliocentric_longitude)


# In[4]:


marsHeliocentric_longitude_InDegree = np.array(marsData['ZodiacIndex'] * 30 +                                           marsData['Degree'] +                                           marsData['Minute.1']/60.0 +                                           marsData['Second']/3600.0)


# In[13]:


marsHeliocentric_longitude_InDegree_InRad = marsHeliocentric_longitude_InDegree * math.pi / 180.0


# In[ ]:





# In[14]:


times = list([0])       #list which will hold all times
for i in range(1, len(marsData)):
        date1 = datetime.datetime(
            marsData['Year'][i-1],marsData['Month'][i-1],
            marsData['Day'][i-1],marsData['Hour'][i-1],
            marsData['Minute'][i-1])
    
        date2 = datetime.datetime(
            marsData['Year'][i],marsData['Month'][i],
            marsData['Day'][i],marsData['Hour'][i],
            marsData['Minute'][i])
        
        duration = date2 - date1
        numOfDays = duration.days + duration.seconds / (60*60*24)
        times.append(numOfDays)
times = np.array(times)
print(times)


# In[15]:


marsHeliocentric_longitude_InDegree = np.array(marsHeliocentric_longitude_InDegree)
oppositions = np.stack((times,marsHeliocentric_longitude_InDegree), axis = 1)
    # print(\"opposition shape:\", oppositions.shape)
    
print('opposition: \n', oppositions)


# In[18]:


#plotting spokes wrt sun-aries axis 
    
plt.figure(figsize=(5,5), dpi=100)
for i in range(0,12):
    xpos = math.cos(math.radians(oppositions[i][1]))
    ypos = math.sin(math.radians(oppositions[i][1]))
        
    x = [0,xpos]
    y = [0,ypos]
        
        
    plt.plot(x,y)
    
plt.show()


# In[ ]:


def plotSpokeWRTSunAriesAxis(oppositions):
    """
    plotting spokes wrt sun-aries axis 
    """
    
    plt.figure(figsize=(5,5), dpi=100)
    for i in range(0,12):
        xpos = math.cos(math.radians(oppositions[i][1]))
        ypos = math.sin(math.radians(oppositions[i][1]))
        
    x = [0,xpos]
    y = [0,ypos]
        
        
    plt.plot(x,y)
    
    plt.show()
    


# In[ ]:


def plotSpokeWRTEquant(oppositions):
    """
    plotting spokes wrt to equant
    """
    c = 60                # angle of centre from sun-aries axis----Assumed
    
    for i in range(1,12):
        s = 360/687                                       #angular speed of mars
        t = ((oppositions[i][0] * s ) + t) % 360  # not fixed 687 , can change in neighborhood
        xposAtTimeT = math.cos(t * (math.pi / 180))
        yposAtTimeT = math.sin(t * (math.pi / 180))
    
    x = [0,xposAtTimeT]
    y = [0,yposAtTimeT]
    
    plt.plot(x, y)
    
    plt.show()


# In[19]:


#plotting spokes wrt to equant
    
plt.figure(figsize=(5,5), dpi=100)
z = 60
t = (oppositions[0][0] / 687 ) * 360 + z
    
xt = math.cos(t * (math.pi / 180))
yt = math.sin(t * (math.pi / 180))
    
x = [0,xt]
y = [0,yt]

plt.plot(x, y)


# In[68]:


get_ipython().system('conda list')


# In[58]:


c = 60                # angle of centre from sun-aries axis----Assumed
t = 0
for i in tqdm(range(1,12)):
    s = 360/687                                       #angular speed of mars
    t = ((oppositions[i][0] * s ) + t) % 360  # not fixed 687 , can change in neighborhood
    xposAtTimeT = math.cos(t * (math.pi / 180))
    yposAtTimeT = math.sin(t * (math.pi / 180))
    
    x = [0,xposAtTimeT]
    y = [0,yposAtTimeT]
    
    plt.plot(x, y)
    
plt.show()


# In[16]:


def getIntersectionPoint(h,k,theta, r,c):
        """
        Return intersection point of 0-centered circle and line from equant\n",
        \n",
        h     : x-coordinate of point through which line passes through\n",
        k     : y-coordinate of point through which line passes through\n",
        theta : angle the line makes with x-axis\n",
        r     : radius of the circle centered at the (1,c) \n",
                where c is the angle centre makes with sun-aries line\n",
        """
    
        b = 2 * ((h * math.cos(math.radians(theta))) + k * math.sin(math.radians(theta))
                -(math.cos(math.radians(c)) * math.cos(math.radians(theta)))
                -(math.sin(math.radians(c)) * math.sin(math.radians(theta))))
        
        c1 = h**2 + k**2 + 1 - (2 * h *math.cos(math.radians(c))) - (2 * k * math.sin(math.radians(c))) - r**2
        l1 = -b / 2
        
        try:
            l2 = math.sqrt(b ** 2 -(4 * c1)) / 2
        except :
            # print(" Value Error ",(b ** 2 -(4 * c1)) / 2)
            l2 = 0
            
        root1 = l1 + l2
        root2 = l1 - l2
        
        if root1 > 0:
            ell = root1
        else:
            ell = root2
        
        return (h + ell * math.cos(math.radians(theta))), (k + ell * math.sin(math.radians(theta)))


# In[17]:


def MarsEquantModel(c,r,e1,e2,z,s,oppositions):
        """
        Return 12 errors of angle delta wrt to 12 oppositions and max error among these 12
        """
        errors = []
        xpos = list()
        ypos = list()
        
        h = e1 * math.cos(math.radians(e2 + z))
        k = e1 * math.sin(math.radians(e2 + z))
    
#         thetaNew = 0
        thetaNew = z
        for i in range(12):
            theta = (s * times[i]) + thetaNew
            x,y = getIntersectionPoint(h,k,theta,r,c)
            xpos.append(x)
            ypos.append(y)
            angle = np.degrees(np.arctan2(y,x))%360
            errors.append(abs(oppositions[i][1] - angle))
            thetaNew = theta
        maxError = max(errors)
        return errors, maxError


# In[ ]:


max_error = 1e20
    
    # print("Max Error, C value, e1 value, e2_value, Z value")
    for  c in np.arange(0,360,5):
        for e2 in np.arange(0,360,5):
            for z in np.arange(0,360,5):
                for e1 in np.arange(0.5, 2,0.2):
                    errors, maxError = MarsEquantModel(c,9,e1,e2,z,360/687,oppositions)
                    if(max_error > maxError):
                        max_error = round(maxError,4)
                        max_c = c
                        max_e1 = round(e1,4)
                        max_e2 = e2
                        max_z = z
                        list_errors = errors
                        print("Max Error:-",max_error," C value:- ",max_c," e1 value:-",max_e1," e2_value",max_e1, " Z value:-", max_z)

    print(list_errors)
    # errors, maxError = MarsEquantModel(150,9,1.4,100,50,360/687,oppositions)
    # errors, maxError = MarsEquantModel(20,5,3,0,45,360/687,oppositions)
    # c = 150, r = 9, z = 60, e1 = 1.4, e2 = 270
    # print(errors)
    # print('------------------')
    # print(maxError)


# In[40]:


errors, maxError = MarsEquantModel(155.199999,7.999999,1.48,93.2999,55.999,360/687,oppositions)
print("12 spokes errors-:",errors)
print("---------------------------------------------------------")
print('maximum error:-',maxError)
# S value:- 0.5229517722254489 C value:- 296  E1 value:- 1.9  E2 value:- 104  z value:- 9  Maximum Error:- 34.965
# S value:- 0.5227998838222465 C value:- 294  E1 value:- 1.9  E2 value:- 100  z value:- 9  Maximum Error:- 34.6343\n",


# In[60]:


def bestOrbitInnerParams(r,s,oppositions):          
        maxError = 1e20
        for c in tqdm(np.arange(149,149.5,0.01)):
            for e2 in np.arange(93,94,0.1):
                for z in np.arange(55,56,0.1):
                    for e1 in np.arange(1.4, 1.6,0.01):
                        errors, max_Error = MarsEquantModel(c,r,e1,e2,z,s,oppositions)
                        if(maxError > max_Error):
                            maxError = round(max_Error,4)
                            max_c = c
                            max_e1 = round(e1,4)
                            max_e2 = e2
                            max_z = z
                            ErrorList = errors
                            print("C value:-",max_c," E1 value:-",max_e1," E2 value:-",max_e2," z value:-",max_z," Maximum Error:-",maxError)
        return max_c,max_e1,max_e2,max_z,ErrorList,maxError


# In[69]:


c,e1,e2,z,errors,maxError = bestOrbitInnerParams(8.09,0.524,oppositions)
print("------------------------------------------------------------------------------")
print("C value:-",c," E1 value:-",e1," E2 value:-",e2," z value:-",z," Error List:-",errors," Maximum Error:-",maxError)


# C value:- 160  E1 value:- 1.7  E2 value:- 94  z value:- 56  Error List:- [0.5182677613559008, 0.4810821782993173, 0.2668597178863479, 0.0720217030286392, 0.008572715391579777, 0.26040476223653286, 0.44249984394002695, 0.015493020770101396, 0.11373086339274607, 0.058426605546117116, 0.2950759394015847, 0.4052584239866519]  Maximum Error:- 0.5183
# 
# C value:- 156  E1 value:- 1.7  E2 value:- 94  z value:- 56  Error List:- [0.4335879959283204, 0.43411727748262763, 0.26327225305570323, 0.06012622586749217, 0.05525068904415775, 0.3279752378614944, 0.44356738151401487, 0.060481490284736594, 0.04603284418153919, 0.07477753699399159, 0.2964579003206893, 0.44714505681105265]  Maximum Error:- 0.4471

# In[42]:


def bestS(r, oppositions):
        leastError = 1e20
        for s in np.arange(686, 688,0.1):
            max_c,max_e1,max_e2,max_z,errors,maxError= bestOrbitInnerParams(r,360/s,oppositions)
            if(leastError > maxError):
                bestS = 360/s
                ErrorList = errors
                leastError = round(maxError,4)
                print("S value:-",bestS,"C value:-",max_c," E1 value:-",max_e1," E2 value:-",max_e2," z value:-",max_z," Maximum Error:-",maxError)
    #             print(\"S value:-\",bestS,\" Error List:-\",errors,\" Maximum Error:-\",maxError)\n",
        return bestS,ErrorList,leastError


# In[ ]:


s,errors,maxError = bestS(7,oppositions)
print("------------------------------------------------------------------------------")
print("S value:-",s," Error List:-",errors," Maximum Error:-",maxError)


# In[43]:


def bestR(s, oppositions):
        leastError = 1e20
        for r in np.arange(7, 9,0.1):
            max_c,max_e1,max_e2,max_z,errors,maxError= bestOrbitInnerParams(r,s,oppositions)
            if(leastError > maxError):
                bestR = r
                ErrorList = errors
                leastError = round(maxError,4)
                print("R value:-",bestR,"C value:-",max_c," E1 value:-",max_e1," E2 value:-",max_e2," z value:-",max_z," Maximum Error:-",maxError),
        return bestR,ErrorList,leastError


# In[ ]:


r,errors,maxError = bestR(0.524,oppositions)
print("------------------------------------------------------------------------------")
print("Best R val:- ",r," Error List:- ",errors," max Error:- ",maxError)


# In[61]:


def bestMarsOrbitParams(oppositions):
    bestError = 1e24
    for r in tqdm(np.arange(8,9,0.1)):
        for s in np.arange(686,687,0.1):
            max_c,max_e1,max_e2,max_z,errors,maxError= bestOrbitInnerParams(r,360/s,oppositions)
            if(bestError > maxError):
                optimumCVal = max_c
                optimume1Val = max_e1
                optimume2Val = max_e2
                optimumZVal = max_z
                bestErrorList= errors
                bestError = maxError
                optimumRVal = r
                optimumSVal = 360/s
                print("Best Error:- ",bestError,"optimumCVal:- ",optimumCVal,"optimume1Val:- ",optimume1Val,"optimume2Val:- ",optimume2Val,"optimumZVal",optimumZVal,"optimumRVal:- ",optimumRVal,"optimumSVal",optimumSVal)
    return optimumRVal,optimumSVal,optimumCVal,optimume1Val,optimume2Val,optimumZVal,bestErrorList,bestError


# In[62]:


r,s,c,e1,e2,z,errors,maxError = bestMarsOrbitParams(oppositions)
print("====================================================================================================")
print("Best Error:- ",maxError,"optimumCVal:- ",c,"optimume1Val:- ",e1,"optimume2Val:- ",e2,"optimumZVal",z,"optimumRVal:- ",r,"optimumSVal",s)


# C value:- 158  E1 value:- 1.6  E2 value:- 93  z value:- 56  Maximum Error:- 0.2679
# C value:- 155  E1 value:- 1.6  E2 value:- 93  z value:- 56  Maximum Error:- 0.313
# Best Error:-  0.2672 optimumCVal:-  156 optimume1Val:-  1.5 optimume2Val:-  93 optimumZVal 56 optimumRVal:-  8.099999999999993 optimumSVal 0.524017467248908
# Best Error:-  0.1327 optimumCVal:-  156.0 optimume1Val:-  1.48 optimume2Val:-  93.0 optimumZVal 55.70000000000001 optimumRVal:-  8.0 optimumSVal 0.5240937545494248
# Best Error:-  0.1213 optimumCVal:-  156.0 optimume1Val:-  1.5 optimume2Val:-  93.1 optimumZVal 55.70000000000001 optimumRVal:-  8.1 optimumSVal 0.5240937545494248
# Best Error:-  0.1097 optimumCVal:-  156.1 optimume1Val:-  1.52 optimume2Val:-  93.1 optimumZVal 55.70000000000001 optimumRVal:-  8.2 optimumSVal 0.5240937545494248
# Best Error:-  0.1023 optimumCVal:-  156.0 optimume1Val:-  1.54 optimume2Val:-  93.19999999999999 optimumZVal 55.70000000000001 optimumRVal:-  8.299999999999999 optimumSVal 0.5240937545494248
# Best Error:-  0.0981 optimumCVal:-  156.0 optimume1Val:-  1.56 optimume2Val:-  93.29999999999998 optimumZVal 55.70000000000001 optimumRVal:-  8.399999999999999 optimumSVal 0.5240937545494248
# Best Error:-  0.1087 optimumCVal:-  153.1 optimume1Val:-  1.49 optimume2Val:-  93.1 optimumZVal 55.70000000000001 optimumRVal:-  8.0 optimumSVal 0.5240937545494248
# Best Error:-  0.094 optimumCVal:-  153.79999999999995 optimume1Val:-  1.52 optimume2Val:-  93.29999999999998 optimumZVal 55.70000000000001 optimumRVal:-  8.2 optimumSVal 0.5240937545494248
# Best Error:-  0.0868 optimumCVal:-  153.89999999999995 optimume1Val:-  1.54 optimume2Val:-  93.29999999999998 optimumZVal 55.70000000000001 optimumRVal:-  8.299999999999999 optimumSVal 0.5240937545494248
# Best Error:-  0.0853 optimumCVal:-  153.89999999999995 optimume1Val:-  1.58 optimume2Val:-  93.19999999999999 optimumZVal 55.70000000000001 optimumRVal:-  8.499999999999998 optimumSVal 0.5240937545494248
# Best Error:-  0.0822 optimumCVal:-  153.89999999999995 optimume1Val:-  1.6 optimume2Val:-  93.19999999999999 optimumZVal 55.70000000000001 optimumRVal:-  8.599999999999998 optimumSVal 0.5240937545494248
# Best Error:-  0.0757 optimumCVal:-  149.1 optimume1Val:-  1.56 optimume2Val:-  93.0 optimumZVal 55.80000000000001 optimumRVal:-  8.399999999999999 optimumSVal 0.5240937545494248
# Best Error:-  0.0702 optimumCVal:-  149.0 optimume1Val:-  1.58 optimume2Val:-  93.1 optimumZVal 55.80000000000001 optimumRVal:-  8.499999999999998 optimumSVal 0.5240937545494248
# Best Error:-  0.0677 optimumCVal:-  149.0 optimume1Val:-  1.6 optimume2Val:-  93.19999999999999 optimumZVal 55.80000000000001 optimumRVal:-  8.599999999999998 optimumSVal 0.5240937545494248
# Best Error:-  0.0677 optimumCVal:-  149.0 optimume1Val:-  1.6 optimume2Val:-  93.19999999999999 optimumZVal 55.80000000000001 optimumRVal:-  8.599999999999998 optimumSVal 0.5240937545494248

# In[63]:


def plotMarsOrbit(c,r,e1,e2,oppositions):
    """
        plot mars orbit given centre of circle and radius along with spokes from equant and sun.
    """
    #change default range 
#     ax.set_xlim((0, 10))
#     ax.set_ylim((0, 10))
    figure, axes = plt.subplots()
    
    CentreXPos = math.cos(math.radians(c))
    CentreyPos = math.sin(math.radians(c))
    
    orbit = plt.Circle((CentreXPos, CentreYPos), r, color='blue', fill = 'false')
    
    plt.title( 'Mars Predicted Orbit' )
    plt.show()


# In[65]:


plotMarsOrbit(149,8.599999999,1.6,93.199999999,oppositions)


# In[ ]:


def __main__():
    
    marsData = pd.read_csv('/home/talk2robots/Documents/Data Analytics/01_data_mars_opposition_updated.csv')
    
    marsHeliocentric_longitude_InDegree = np.array(marsData['ZodiacIndex'] * 30 +                                           marsData['Degree'] +                                           marsData['Minute.1']/60.0 +                                           marsData['Second']/3600.0)
    
    marsHeliocentric_longitude_InDegree_InRad = marsHeliocentric_longitude_InDegree * math.pi / 180.0
    
    times = list([0])       #list which will hold all times
    for i in range(1, len(marsData)):
        date1 = datetime.datetime(
            marsData['Year'][i-1],marsData['Month'][i-1],
            marsData['Day'][i-1],marsData['Hour'][i-1],
            marsData['Minute'][i-1])
    
        date2 = datetime.datetime(
            marsData['Year'][i],marsData['Month'][i],
            marsData['Day'][i],marsData['Hour'][i],
            marsData['Minute'][i])
        
        duration = date2 - date1
        numOfDays = duration.days + duration.seconds / (60*60*24)
        times.append(numOfDays)
    times = np.array(times)
    print(times)
    
    marsHeliocentric_longitude_InDegree = np.array(marsHeliocentric_longitude_InDegree)
    oppositions = np.stack((times,marsHeliocentric_longitude_InDegree), axis = 1)
#     print(\"opposition shape:\", oppositions.shape)
    
    print('opposition: \n', oppositions)
    
    
    #plotting spokes wrt sun-aries axis 
    plotSpokeWRTSunAriesAxis(oppositions)
    
    # plotting spokes wrt to equant
    plotSpokeWRTEquant
    
    
    errors, maxError = MarsEquantModel(149.00000000001,8.59999999999,1.60,93.19999999999,55.8000000000001,0.524093,oppositions)
    print("12 spokes errors-:",errors)
    print("====================================================================================================")
    print('\n maximum error:-',maxError)
    
    c,e1,e2,z,errors,maxError = bestOrbitInnerParams(8.09,0.524,oppositions)
    print("====================================================================================================\n")
    print("C value:-",c," E1 value:-",e1," E2 value:-",e2," z value:-",z," Error List:-",errors," Maximum Error:-",maxError)
    
    
    s,errors,maxError = bestS(8.4,oppositions)
    print("====================================================================================================\n")
    print("S value:-",s," Error List:-",errors," Maximum Error:-",maxError)
    
    r,errors,maxError = bestR(0.524,oppositions)
    print("====================================================================================================\n")
    print("Best R val:- ",r," Error List:- ",errors," max Error:- ",maxError)
    
    
    r,s,c,e1,e2,z,errors,maxError = bestMarsOrbitParams(oppositions)
    print("====================================================================================================\n")
    print(" Best Error:- ",maxError," optimumCVal:- ",c," optimume1Val:- ",e1," optimume2Val:- ",e2," optimumZVal",z," optimumRVal:- ",r," optimumSVal",s)
    
    
    

