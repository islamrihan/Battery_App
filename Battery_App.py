import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
import json
import calendar
import datetime 
import math
import streamlit as st
import os, urllib, cv2
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
from datetime import timedelta


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("README.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
        
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("Battery_App.py"))
        

@st.experimental_singleton(show_spinner=True)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/islamrihan/Battery_App/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")
        
# To run the app, type the following in Anaconda Prompt: 
# streamlit run "c:/Users/IRIHA/OneDrive - Ramboll/Documents/00 Work Tasks/221118_GridCarbon/BatteryApp/Battery_app.py"


def run_the_app():
    
    # Get hourly CO2 intensity data from API for one year
    def GridCarbonHourly(Year):
        
        headers = {'Accept': 'application/json'}
        # constructing the pull request from API
        Date = pd.date_range(str(Year)+'-01-01', periods=365,freq='D').strftime('%Y-%m-%d').tolist()
        Dates = list(Date)
        URL = 'https://api.carbonintensity.org.uk/intensity/date/'
        URLS = [URL+str(i) for i in Dates]

        r = [requests.get(url=i, params={}, headers = headers) for i in URLS]

        # constructing the DF from the pulled raw data
        raw_CO2int = pd.DataFrame(i.json()['data'][y]['intensity']['actual'] for i in r for y in range(len(i.json()['data'])))
        raw_CO2date =pd.to_datetime([i.json()['data'][y]['from'] for i in r for y in range(len(i.json()['data']))])
        raw_CO2int.index = raw_CO2date.tz_localize(None)
        CO2IntesityHourly=pd.DataFrame(raw_CO2int[0].resample('H').mean()).rename(columns={0:'Carbon_Intensity'}).iloc[1: , :].interpolate()
        return(CO2IntesityHourly)
    
    
    # Define variables in use
    CO2_GRID = 'CI_gCO2_Grid' # Hourly Carbon Intensity of the grid in gCO2
    CO2_AVG_D = 'CI_gCO2_Grid_DA' # Daily Avg of Carbon Intensity of the grid in gCO2
    EU_BLDG = 'EU_kWh_BLDG' # Enegry use of the building in kWh
    CO2_DIFF = 'CI_gCO2_Difference' # Difference between carbon Intensity of the grid and the daily average 
    TIMESTAMP = 'timestamp'
    CARBONINTENSITY = 'carbon_intensity'
    
    # Define stramlit tabs
    tab2, tab1, tab3, tab4 = st.tabs(["Carbon Emissions", "Energy Use", "All Data", "Battery Profile"])    
    
    
    # Define stramlit column containers
    with tab1:
        t1_col1, t1_col2 = st.columns([4.5, 1.2])
    
    with tab2:
        t2_col1, t2_col2 = st.columns([4.5, 1.2])
    

    # Define year selection dropdown
    Year = st.sidebar.selectbox(
        '1- Year for Carbon Intensity data:',
        (2021,2019,2018))
    
    FP = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/2fa_HourlyAverageElectricalPower.xlsx'
    excel_URL = st.sidebar.text_input('2- URL for hourly energy simulation (.csv): ', FP, key='System FP')
    

    _max_load_quantile = st.sidebar.slider(
        label = r'3- Battery capacity (% of max. daily load):',
        help = 'This value determines the battery size as a percentage from the maximum daily load of building operational energy demand.',
        value = 60,
        max_value = 100,
        min_value = 5)  # slider widget
    
    _battery_charging_rate = st.sidebar.slider(
        label = r'4- Battery charging rate (hours):',
        help = 'This value determines the number of hours for charging the battery.',
        value = 4,
        max_value = 12,
        min_value = 1)  # slider widget
    
    _battery_discharging_rate = st.sidebar.slider(
        label = r'5- Battery discharging rate (hours):',
        help = 'This value determines the number of hours for discharging the battery.',
        value = 4,
        max_value = 12,
        min_value = 1)  # slider widget

   
   
    csv_URL = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/gCO2_' + str(Year) + '.csv' 
    # @st.cache
    @st.experimental_memo
    def csv_to_df(URL):
        return pd.read_csv(csv_URL,index_col=[0],parse_dates=True)
        
    @st.experimental_memo
    def csv_to_df_timestamps(URL):
        return pd.read_csv(csv_URL)
    
    
    GridCO2 = csv_to_df(csv_URL)
    GridCO2 = GridCO2.rename(columns={GridCO2.columns[0]:CO2_GRID})
    
    df = csv_to_df_timestamps(csv_URL)
    df = df.rename(columns={df.columns[0]:TIMESTAMP})
    df = df.rename(columns={df.columns[1]:CARBONINTENSITY})
    


    
    
    # Convert the timestamp column to a datetime data type
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])

    # Extract the date from each timestamp
    df['date'] = df[TIMESTAMP].dt.date

    # Find the lowest carbon intensity value for each day
    df['min_intensity'] = df.groupby(df['date'])[CARBONINTENSITY].transform('min')

    # Find the highest carbon intensity value for each day
    df['max_intensity'] = df.groupby(df['date'])[CARBONINTENSITY].transform('max')

    df['charging'] = False
    df['discharging'] = False
    
    # Find the index of the row with the lowest carbon intensity for each day
    min_index = df.groupby('date')[CARBONINTENSITY].idxmin()

    # Find the index of the row with the highest carbon intensity for each day
    max_index = df.groupby('date')[CARBONINTENSITY].idxmax()

    # Charge the battery 2 hours before the time when the carbon intensity is at its lowest for each day
    charge_times = df.loc[min_index, TIMESTAMP]

    # Discharge the battery 2 hours after the time when the carbon intensity is at its highest for each day
    discharge_times = df.loc[max_index, TIMESTAMP] 


        
    step_charging = int(_battery_charging_rate/2)
    step_discharging = int(_battery_discharging_rate/2)
    
    if _battery_charging_rate % 2 == 0:
        for i in min_index:
            df["charging"].iloc[range(max(0,i-step_charging), min(8760,i+step_charging))] = True
    else:    
        for i in min_index:
            df["charging"].iloc[range(max(0,i-step_charging), min(8760,i+step_charging+1))] = True 
            
    if _battery_discharging_rate % 2 == 0:        
        for i in max_index:
            df["discharging"].iloc[range(max(0,i-step_discharging), min(8760,i+step_discharging))] = True    
    else:            
        for i in max_index:
            df["discharging"].iloc[range(max(0,i-step_discharging), min(8760,i+step_discharging+1))] = True
    
    df = df.set_index(TIMESTAMP)


    # Daily averaging (get one point per day to represent grid CO2 carbon)
    GridCO2_D = GridCO2.resample('D').mean()

    # Scale the daily average data to 8760 hours
    GridCO2_DS = pd.DataFrame(np.repeat(GridCO2_D.values, 24, axis=0))
    GridCO2_DS = GridCO2_DS.rename(columns={0:CO2_AVG_D})

    # redefine index column to timestamp
    GridCO2_DS['DATETIME'] = GridCO2.index 
    GridCO2_DS = GridCO2_DS.set_index(['DATETIME'])


    # @st.cache
    @st.experimental_memo
    def excel_to_df(URL):
        return pd.read_excel(excel_URL)
     
    

    # Get energy use data from excel   
    EnergyUse = excel_to_df(excel_URL)['Unnamed: 61']
    EnergyUse = EnergyUse.drop(index = [0,1]).reset_index().iloc[:, [1]].rename(columns = {'Unnamed: 61':EU_BLDG})
    
    # Set timestamp column as index
    EnergyUse['DATETIME'] = df.index
    EnergyUse = EnergyUse.set_index(['DATETIME'])
    
    EnergyUse_D = EnergyUse.resample('D').sum()





    GridCO2_vs_EU = pd.concat([GridCO2, EnergyUse, GridCO2_DS], axis=1)
    
    
    GridCO2_vs_EU[['Charging','Discharging']] = df[['charging','discharging']]


    GridCO2_vs_EU[CO2_DIFF] = GridCO2_vs_EU[CO2_GRID] - GridCO2_vs_EU[CO2_AVG_D]


       
    battery_size = round(_max_load_quantile/100 * EnergyUse_D.values.max(), 2)
    

    
    charging_rate = battery_size/_battery_charging_rate
    battery_charge = 0
    EU_battery = 0
    EU_BnB = 0
    
    battery_kWh = []
    EU_kWh_Battery = []
    EU_BldgAndBatt = []

    for i, j, k in zip(GridCO2_vs_EU['Charging'],range(len(GridCO2_vs_EU[EU_BLDG])), GridCO2_vs_EU['Discharging']):
        
        # Battery is CHARGING; (LOW CO2 intensity from the grid)
        if i == True: 
            # Find battery charge level for every hour while CHARGING
            battery_charge = min(battery_size, battery_charge + charging_rate) #insure charging does not exceed battery size
            battery_kWh.append(battery_charge)
            
            # Calculate battery energy consumption for every hour
            if j == 0: 
                EU_battery = battery_kWh[0]
                EU_kWh_Battery.append(EU_battery)
            else:
                EU_battery = battery_charge - battery_kWh[j-1]
                EU_kWh_Battery.append(EU_battery)
            
            # Building energy use + battery charging use = total (building and battery) energy use for each hour
            EU_BnB = GridCO2_vs_EU[EU_BLDG].iloc[j] + EU_battery
            EU_BldgAndBatt.append(EU_BnB)
            
        
        # Battery is DISCHARGING; (HIGH CO2 intensity from the grid)
        elif k == True: 
            # Find battery charge level for every hour while DISCHARGING
            battery_charge = max(0 , battery_charge - GridCO2_vs_EU[EU_BLDG].iloc[j]) #insure charging does not go below zero
            battery_kWh.append(battery_charge)
            
            # Set battery energy use to ZERO for each hour (as it is discharging)
            EU_battery = 0
            EU_kWh_Battery.append(EU_battery)
            
            # Find total buildng and battery energy use for each hour
            if GridCO2_vs_EU[EU_BLDG].iloc[j] > battery_charge:
                EU_BnB = GridCO2_vs_EU[EU_BLDG].iloc[j] - battery_charge
            else:
                EU_BnB = 0
            EU_BldgAndBatt.append(EU_BnB)
        
        # Battery is NOT IN USE (neither charging nor discharging);    
        else:
            battery_kWh.append(battery_charge)
            EU_battery = 0
            EU_kWh_Battery.append(EU_battery)
            
            EU_BnB = GridCO2_vs_EU[EU_BLDG].iloc[j]
            EU_BldgAndBatt.append(EU_BnB)
    
    # Convert lists to dataframe columns        
    GridCO2_vs_EU['battery_charge'] = [elem for elem in battery_kWh]
    GridCO2_vs_EU['EU_kWh_Battery'] = [elem for elem in EU_kWh_Battery]
    GridCO2_vs_EU['EU_BldgAndBatt'] = [elem for elem in EU_BldgAndBatt]
    GridCO2_vs_EU['kgCO2_BldgAndBatt'] = GridCO2_vs_EU['EU_BldgAndBatt'] * GridCO2_vs_EU[CO2_GRID] / 1000

    
    #charge_frequency = GridCO2_vs_EU['battery_charge'].value_counts().sort_values(ascending=False).head(10)
    #st.bar_chart(charge_frequency)

    compare = pd.DataFrame()
    compare['kWh_Opt0'] = GridCO2_vs_EU[EU_BLDG].apply(lambda x: round(x, 2))
    compare['kWh_Opt0_cumsum'] = compare['kWh_Opt0'].cumsum(axis=0, skipna=True)

    compare['kgCO2_Opt0'] = compare['kWh_Opt0'] * GridCO2_vs_EU[CO2_GRID] / 1000 
    compare['kgCO2_Opt0_cumsum'] = compare['kgCO2_Opt0'].cumsum(axis=0, skipna=True)

    compare['kWh_Opt1'] = GridCO2_vs_EU['EU_BldgAndBatt']
    compare['kWh_Opt1_cumsum'] = compare['kWh_Opt1'].cumsum(axis=0, skipna=True)

    compare['kgCO2_Opt1'] = GridCO2_vs_EU['kgCO2_BldgAndBatt'] 
    compare['kgCO2_Opt1_cumsum'] = compare['kgCO2_Opt1'].cumsum(axis=0, skipna=True)


    compare = compare.round(decimals=2)
    compare.head(48)


    # Create daily totals of Energy Use 
    EU_B_D = compare['kWh_Opt0'].resample('D').sum().to_frame()

    EU_BnB_D = compare['kWh_Opt1'].resample('D').sum().to_frame()
    
    EU_compared_D = pd.concat([EU_B_D, EU_BnB_D], axis=1)

    totals_EU = EU_compared_D.sum()
    

    old_EU = totals_EU.iloc[0]/1000
    new_EU = totals_EU.iloc[1]/1000
    differ_EU = old_EU - new_EU
    percent_savings_EU = ((totals_EU.iloc[0]-totals_EU.iloc[1])/totals_EU.iloc[0])*100    
    
    # Create daily cummulative summations of Energy Use 
    EU_B_cumsum = EU_compared_D['kWh_Opt0'].cumsum(axis=0, skipna=True)
    EU_BnB_cumsum = EU_compared_D['kWh_Opt1'].cumsum(axis=0, skipna=True)
    EU_compared_cumsum = pd.concat([EU_B_cumsum, EU_BnB_cumsum], axis=1)

    
    
    with tab1:
        with t1_col2:
            st.markdown('<p style="color:Grey; font-size: 22px;">Energy Use', unsafe_allow_html=True)
            st.caption('in MWh')
                       
            st.metric(label='Building Only:', value=int(old_EU))
            st.metric(label='Building with Battery:', value=int(new_EU), delta=str(-(percent_savings_EU.round(1)))+"%", delta_color="off" )
            
        with t1_col1:
            st.line_chart(EU_compared_D)
#            kWh_figure = px.line(EU_compared_cumsum, markers=False, width=520, height=500)
#            kWh_figure.update_xaxes(title = 'Day').update_yaxes(title = 'gCO2').update_layout(
#                legend=dict(x=0.75, y=0.05,traceorder="normal"),
#                margin=dict(l=0, r=0, t=0, b=0),
#                showlegend=True
#            )
#            st.plotly_chart(kWh_figure, theme = "streamlit")
                        
                       
    
    
    CO2_B_D = compare['kgCO2_Opt0'].resample('D').sum().to_frame()
    CO2_BnB_D = compare['kgCO2_Opt1'].resample('D').sum().to_frame()
    CO2_compared_D = pd.concat([CO2_B_D, CO2_BnB_D], axis=1)
    totals_CO2 = CO2_compared_D.sum()

    old_CO2 = totals_CO2.iloc[0]/1000
    new_CO2 = totals_CO2.iloc[1]/1000
    differ_CO2 = old_CO2 - new_CO2
    percent_savings_CO2 = ((totals_CO2.iloc[0]-totals_CO2.iloc[1])/totals_CO2.iloc[0])*100
    

            


    CO2_B_cumsum = CO2_compared_D['kgCO2_Opt0'].cumsum(axis=0, skipna=True)
    CO2_BnB_cumsum = CO2_compared_D['kgCO2_Opt1'].cumsum(axis=0, skipna=True)


    CO2_compared_cumsum = pd.concat([CO2_B_cumsum, CO2_BnB_cumsum], axis=1)
    
    
    # Compose carbon emissions tab 
    with tab2:
        
        # Add side bar highlighted values
        with t2_col2:
            st.caption('<p style="color:Grey; font-size: 22px;">CO<sub>2</sub> Footprint', unsafe_allow_html=True)
            st.caption('<p style="color:Grey; font-size: 12px;">in CO<sub>2</sub> Tonnes',unsafe_allow_html=True)
                       
            st.metric(label='Building Only:', value=int(old_CO2))
            st.metric(label='Building & Battery:', value=int(new_CO2), delta=str(-(percent_savings_CO2.round(1)))+"%", delta_color="inverse" )
                        
          
            battery_embodied = battery_size * 50 / 1000
            st.metric(label='Battery Embodied CO2*', value=int(battery_embodied))
            st.caption('<p style="color:Grey; font-size: 12px;">*: assuming 50 kgCO2/kWh',unsafe_allow_html=True)
            st.write("Battery capacity is", round(battery_size/1000 ,2), "MWh")
            
            st.caption('Pay-back Period:')
            
            CO2_payback_years = battery_embodied / differ_CO2
            
            
            if CO2_payback_years < 0:
                st.markdown('<p style="color:Red;">NaN', unsafe_allow_html=True)
            else:
                st.subheader(str(CO2_payback_years.round(1)) + ' years')      
    
    
        # Cumulative carbon graph
        with t2_col1:
            kgCO2_figure = px.line(CO2_compared_cumsum, markers=False, width=520, height=500)
            kgCO2_figure.update_xaxes(title = 'Day').update_yaxes(title = 'gCO2').update_layout(
                legend=dict(x=0.75, y=0.05,traceorder="normal"),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
            st.plotly_chart(kgCO2_figure, theme = "streamlit")


    # clear cache memory button 
    if st.sidebar.button("Clear Memory", type="secondary"):
        # Clear values from *all* memoized functions:
        # i.e. clear values from both square and cube
        st.experimental_memo.clear()
        
        
        
    # Display all data in a table    
    with tab3:
        st.dataframe(GridCO2_vs_EU.drop(columns=[CO2_DIFF,CO2_AVG_D]))

        st.line_chart(compare.drop(columns=['kgCO2_Opt0_cumsum','kgCO2_Opt1_cumsum','kWh_Opt0_cumsum','kWh_Opt1_cumsum']))
        
        
    with tab4:
        st.line_chart(GridCO2_vs_EU[['battery_charge']])




if __name__ == "__main__":
    main()