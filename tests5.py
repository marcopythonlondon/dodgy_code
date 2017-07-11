#C:/Users/PC_marco/AppData/Local/Programs/Python/Python36-32/scripts/pip install pandas-datareader

######   To-Do  &  Notes  ######
#   how to get Candlestick_ohlc

   

###### PACKAGES ######
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from matplotlib.finance import candlestick_ohlc 
import numpy as np
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from subprocess import Popen

from operator import truediv


###### DATA ######

#starttime = dt.datetime(2000, 1, 1)
#endtime = dt.datetime(2017, 12, 31)
#
#dataTS = web.DataReader('JPM', 'yahoo', starttime, endtime)
#dataTS.to_csv('YHOO.csv')


Stock = 'YHOO'
text_for_CSV = Stock+'.csv'

TS = pd.read_csv(text_for_CSV, parse_dates=True, index_col=0)
TS_df = TS.copy()

#TS.reset_index(inplace=True)

#TS.to_csv('YHOO_backup.csv')

#CloseP = TS['Adj Close']

#TS[['Open', 'High', 'Low', 'Adj Close']].plot()
#TS['Adj Close'].plot()



###### INDICATORS ######

    ### MOVING AVERAGES ###
TS['MA10']  = TS['Adj Close'].rolling(window = 10, min_periods=None).mean()
TS['MA100'] = TS['Adj Close'].rolling(window = 100, min_periods=None).mean()
TS['MA200'] = TS['Adj Close'].rolling(window = 200, min_periods=None).mean()

TS['MAVolume10']  = TS['Volume'].rolling(window = 10, min_periods=None).mean()
TS['MAVolume100'] = TS['Volume'].rolling(window = 100, min_periods=None).mean()

    ### RSI ###
RSI_period = 14

RSI_price_chg = TS['Adj Close'] - TS['Adj Close'].shift(1)
TS['RSI_price_chg'] = RSI_price_chg
 
#TS['RSI_price_chg'] = pd.Series(TS['RSI_price_chg']).astype(float)
#.convert_objects(convert_numeric=True)


def RSI_Gain(RSI_price_chg):
    if RSI_price_chg > 0:
        return RSI_price_chg
    else:
        return 0

def RSI_Loss(RSI_price_chg):
    if RSI_price_chg < 0:
        return RSI_price_chg * -1
    else:
        return 0


TS['RSI_Gain'] = list(map(RSI_Gain, TS['RSI_price_chg']))
TS['RSI_Loss'] = list(map(RSI_Loss, TS['RSI_price_chg']))
TS['Avg_RSI_Gain'] = TS['RSI_Gain'].rolling(window = RSI_period, min_periods=None).mean()
TS['Avg_RSI_Loss'] = TS['RSI_Loss'].rolling(window = RSI_period, min_periods=None).mean()
TS['RS'] = TS['Avg_RSI_Gain'] / TS['Avg_RSI_Loss']
TS['RSI'] = 100 - (100 / (1 + TS['RS']))



    ### MACD ###
TS['MACD_ST'] = TS['Adj Close'].ewm(com=12, min_periods=0, adjust=True, ignore_na=False).mean()
TS['MACD_LT'] = TS['Adj Close'].ewm(com=26, min_periods=0, adjust=True, ignore_na=False).mean()
TS['MACD'] = TS['MACD_ST'] - TS['MACD_LT']
TS['MACD_signal'] = TS['MACD'].rolling(window = 9, min_periods=None).mean()

def MACD_trigger(MACD, MACD_signal):
    if MACD > MACD_signal:
        return 1
    else:
        return 0  
  
TS['MACD_trigger'] = list(map(MACD_trigger, TS['MACD'], TS['MACD_signal']))



    ### STOCHASTICS ###
TasST_nbr_of_days = 14
TasLT_nbr_of_days = 3

Lowest_low = TS['Low'].rolling(window = TasST_nbr_of_days, min_periods=None).min()
Highest_high = TS['High'].rolling(window = TasST_nbr_of_days, min_periods=None).max()

TasST = ((TS['Close'] - Lowest_low) / (Highest_high - Lowest_low)) *100
TasLT = TasST.rolling(window = TasLT_nbr_of_days, min_periods=None).mean()
 
#TasST = np.float64(TasST)
#TasLT = np.float64(TasLT)
 
TS['TasST'] = TasST       
TS['TasLT'] = TasLT


    ##################
    ### SIGNAL TAS ###
    #################


## NOTES ##

#    TS_df['TAS_signal_prepa0'] = list(map(TAS_signal_prepa_STinfLT, TasST, TasLT))
#    TS_df['TAS_signal_STinfLT'] = TS_df['TAS_signal_prepa0'].shift(1)

#    TS_df['TAS_signal_prepa_sup'] = list(map(TAS_signal_prepa_STsupLT, TasST, TasLT))
#    TS_df['TAS_signal_STsupLT'] = TS_df['TAS_signal_prepa_sup'].shift(1)

#    TS_df['TasST_prev'] = TS_df['TasST'].shift(1)
#    TS_df['TasLT_prev'] = TS_df['TasLT'].shift(1)



TS_df['TasST'] = TasST
TS_df['TasLT'] = TasLT

def TAS_signal_prepa_STinfLT(TasST, TasLT):
    if TasST < TasLT:
        return 1
    else:
        return 0

TS_df['TAS_signal_prepa_inf'] = list(map(TAS_signal_prepa_STinfLT, TasST, TasLT))
TS_df['TAS_signal_STinfLT'] = TS_df['TAS_signal_prepa_inf'].shift(1)


def TAS_signal_prepa_STsupLT(TasST, TasLT):
    if TasST > TasLT:
        return 1
    else:
        return 0

TS_df['TAS_signal_prepa_sup'] = list(map(TAS_signal_prepa_STsupLT, TasST, TasLT))
TS_df['TAS_signal_STsupLT'] = TS_df['TAS_signal_prepa_sup'].shift(1)


TS_df['TasST_prev'] = TS_df['TasST'].shift(1)
TS_df['TasLT_prev'] = TS_df['TasLT'].shift(1)

def TAS_cross_pos(TAS_signal_STinfLT, TasST, TasLT):
    if (TAS_signal_STinfLT == 1) and (TasST > TasLT):
        return 1
    else:
        return 0

TS_df['cross_pos'] = list(map(TAS_cross_pos, TS_df['TAS_signal_STinfLT'], TS_df['TasST'], TS_df['TasLT']))    


def TAS_cross_neg(TAS_signal_STsupLT, TasST, TasLT):
    if (TAS_signal_STsupLT == 1) and (TasST < TasLT):
        return 1
    else:
        return 0

TS_df['cross_neg'] = list(map(TAS_cross_neg, TS_df['TAS_signal_STsupLT'], TS_df['TasST'], TS_df['TasLT']))    



### TAS_signal_0

Level_TAS_signal_0 = 30

def TAS_signal_0 (TasST, TasLT, prepa):
     if TasST < Level_TAS_signal_0 and TasST > TasLT and prepa == 1:
         return 1
     else:
         return 0

TS_df['TAS_signal_0'] = list(map(TAS_signal_0, TasST, TasLT, TS_df['TAS_signal_STinfLT']))     



### TAS_nbr_days_since_TAS_signal_0

NbrOfDays0 = 0

def TAS_nbr_days_since_TAS_signal_0(trigger0):
    global NbrOfDays0
    if trigger0 == 1:
        NbrOfDays0 = 0
        return NbrOfDays0
    else:
        NbrOfDays0 += 1
        return NbrOfDays0

TS_df['TAS_nbr_days_since_TAS_signal_0'] = list(map(TAS_nbr_days_since_TAS_signal_0, TS_df['TAS_signal_0']))
     

        
### TAS_signal_1

TasST_signal1 = 50

def TAS_signal_1(TasST_prev, TasST, TasLT, prepa_signal1):
    if (TasST_prev >= TasST_signal1) and (TasST < TasLT) and (prepa_signal1 == 1):
        return 1
    else:
        return 0

TS_df['TAS_signal_1'] = list(map(TAS_signal_1, TS_df['TasST_prev'], TS_df['TasST'], TS_df['TasLT'], TS_df['TAS_signal_STsupLT']))



### TAS_nbr_days_since_TAS_signal_1

NbrOfDays1 = 0

def TAS_nbr_days_since_TAS_signal_1(trigger1):
    global NbrOfDays1
    if trigger1 == 1:
        NbrOfDays1 = 0
        return NbrOfDays1
    else:
        NbrOfDays1 += 1
        return NbrOfDays1

TS_df['TAS_nbr_days_since_TAS_signal_1'] = list(map(TAS_nbr_days_since_TAS_signal_1, TS_df['TAS_signal_1']))
        


### TAS_signal_cross

NbrOfDays_trigger0 = 51
NbrOfDays_trigger1 = 21

def TAS_signal_cross(TAS_nbr_days_since_TAS_signal_0, TAS_nbr_days_since_TAS_signal_1, TasST, TasLT, TAS_signal_STinfLT):
    if (TAS_nbr_days_since_TAS_signal_0 <= NbrOfDays_trigger0) and (TAS_nbr_days_since_TAS_signal_1 <= NbrOfDays_trigger1) and (TasST > TasLT) and (TAS_signal_STinfLT == 1):
        return 1
    else:
        return 0

TS_df['TAS_signal_cross'] = list(map(TAS_signal_cross, TS_df['TAS_nbr_days_since_TAS_signal_0'], TS_df['TAS_nbr_days_since_TAS_signal_1'], TS_df['TasST'], TS_df['TasLT'], TS_df['TAS_signal_STinfLT']))



### TAS_nbr_days_since_TAS_signal_cross

NbrOfDaysCross = 0

def TAS_nbr_days_since_TAS_signal_cross(TriggerCross):
    global NbrOfDaysCross
    if TriggerCross == 1:
        NbrOfDaysCross = 0
        return NbrOfDaysCross
    else:
        NbrOfDaysCross += 1
        return NbrOfDaysCross

TS_df['TAS_nbr_days_since_TAS_signal_cross'] = list(map(TAS_nbr_days_since_TAS_signal_cross, TS_df['TAS_signal_cross']))



### TAS_signal_NbrOfPosCross

CountCrossPos = 0
def TAS_signal_NbrOfPosCross(cross_pos):
    global CountCrossPos
    if cross_pos == 1:
        CountCrossPos += 1
        return CountCrossPos
    else:
        return CountCrossPos

TS_df['TAS_signal_NbrOfPosCross'] = list(map(TAS_signal_NbrOfPosCross, TS_df['cross_pos']))        



### TAS_signal_NbrOfNegCross

CountCrossNeg = 0
def TAS_signal_NbrOfNegCross(cross_neg, TAS_signal_0):
    global CountCrossNeg
    if cross_neg == 1:
        CountCrossNeg += 1
        return CountCrossNeg
    elif TAS_signal_0 == 1:
        CountCrossNeg = 0
        return CountCrossNeg
    else:
        return CountCrossNeg

TS_df['TAS_signal_NbrOfNegCross'] = list(map(TAS_signal_NbrOfNegCross, TS_df['cross_neg'], TS_df['TAS_signal_0']))        



### stop-loss

Stop_loss = 0.20

Price_entry = TS_df['Adj Close']


def Price_entry(signal, Price):
    global Price_entry
    if signal == 1:
        Price_entry = Price
        return Price_entry
    else:
        Price_entry = Price_entry
        return Price_entry
 
TS_df['Price_entry'] = list(map(Price_entry, TS_df['TAS_signal_cross'], TS_df['Adj Close']))


#######   what is not working starts here.......
testPE2 = TS_df['Price_entry'].tolist()
TS_df['Price_entry_3'] = str(testPE2)



TS_df['test'] = TS_df['Price_entry_2'] / TS_df['Price_entry']




#def Current_level():
#    if (TS_df['Adj_Close'] / float(TS_df['Price_entry'])) > 0:
#        Current_level = TS_df['Adj_Close'] / float(TS_df['Price_entry'])
#        return Current_level
#    else:
#        Current_level = 1
#        return Current_level


#######   ....... and finishes here




### stop-gain







### SUMMARY ###

##  IN 
# 1) TAS_signal_cross

##  OUT
# 1) TAS_signal_NbrOfNegCross = 4
# 2) stop gain
# 3) stop loss
# 4) nbr of days    

    
    
    

###### CREATE NEW FILE WITH NEW DATA ######
TS['Daily_Ret'] = TS['Adj Close'].pct_change(1)

       
### Marco_indicators = TS

Marco_indicators = TS_df.copy()
Marco_indicators.to_csv('Marco_indicators.csv')
Popen('Marco_indicators.csv', shell=True)

print(TS.head())



###### CHART ######

style.use('ggplot')
#style.use('fivethirtyeight')

ax1 = plt.subplot2grid((7,2),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((7,2),(5,0), rowspan=1, colspan=1, sharex=ax1)
ax3 = plt.subplot2grid((7,2),(6,0), rowspan=2, colspan=1, sharex=ax1)
ax4 = plt.subplot2grid((7,2),(0,1), rowspan=2, colspan=1, sharex=ax1)

ax1.plot(TS.index, TS['Adj Close'], label='Adj Close')
ax1.plot(TS.index, TS['MA200'], color='y', linewidth=3, label='MA200')
ax1.set_title(Stock, horizontalalignment='center', verticalalignment='top')

ax2.plot(TS.index, TS['Volume'], color='k', label='Volume')
ax2.plot(TS.index, TS['MAVolume100'], label='MAVolume100')

ax3.plot(TS.index, TS['MACD'], color='y', linewidth=3, label='MACD')
ax3.plot(TS.index, TS['MACD_signal'], color='r', linewidth=1, label='MACD_signal')

ax4.plot(TS.index, TS['MACD_trigger'], color='k', linewidth=1, label='MACD_trigger')


plt.legend()
plt.show()





def cls(): 
    print ("\n" * 50)
