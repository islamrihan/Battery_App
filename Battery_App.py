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
    
    
    # Define stramlit tabs
    tab1, tab2 = st.tabs(["Energy Use", "Carbon Emissions"])    
    
    
    # Define stramlit column containers
    with tab1:
        t1_col1, t1_col2 = st.columns([3.5, 1.2])
    
    with tab2:
        t2_col1, t2_col2 = st.columns([4.5, 2])
    

    
    
    
    
    
    Year = st.sidebar.selectbox(
        '1- Year for Carbon Intensity data:',
        (2021,2019))
    
    FP = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/2fa_HourlyAverageElectricalPower.xlsx'
    excel_URL = st.sidebar.text_input('2- URL for hourly energy simulation (.csv): ', FP, key='System FP')
    
    _input_CO2_diff = st.sidebar.slider(
        label = '3- Battery activating difference in hourly vs. daily average (in gCO2):',
        help = 'This value determines when the building switches the battery ON/OFF based on the current hourly difference between grid carbont intensity hourly and daily averaged data.',
        value = 20,
        max_value = 50,
        min_value = 0)  # slider widget
    #st.sidebar.write('difference is set to: ', _input_CO2_diff, 'gCO2')

    _max_load_quantile = st.sidebar.slider(
        label = r'4- Battery capacity (% of maximum daily load):',
        help = 'This value determines which percentile reprecents battery size of the maximum load period for electrical demand.',
        value = 0.75,
        max_value = 1.00,
        min_value = 0.05)  # slider widget
    #st.sidebar.write('Battery sizing is', _max_load_quantile, 'percentile of the max load')
    


        
        
        
        
    csv_URL = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/gCO2_' + str(Year) + '.csv' 
    # @st.cache
    @st.experimental_memo
    def csv_to_df(URL):
        return pd.read_csv(csv_URL,index_col=[0],parse_dates=True)
        
    GridCO2 = csv_to_df(csv_URL)
    GridCO2 = GridCO2.rename(columns={GridCO2.columns[0]:CO2_GRID})
       
    # Daily averaging (get one point per day to represent grid CO2 carbon)
    GridCO2_D = GridCO2.resample('D').mean()

    # scale the daily avg data to 8760 hours
    GridCO2_DS = pd.DataFrame(np.repeat(GridCO2_D.values, 24, axis=0))
    GridCO2_DS = GridCO2_DS.rename(columns={0:CO2_AVG_D})

    # redefine index column to timestamp
    GridCO2_DS['DATETIME'] = GridCO2.index 
    GridCO2_DS = GridCO2_DS.set_index(['DATETIME'])






    # @st.cache
    @st.experimental_memo
    def excel_to_df(URL):
        return pd.read_excel(excel_URL)
     
    

       
    EnergyUse = excel_to_df(excel_URL)['Unnamed: 61']
    EnergyUse = EnergyUse.drop(index = [0,1]).reset_index().iloc[:, [1]].rename(columns = {'Unnamed: 61':EU_BLDG})
    # redefine index column to timestamp
    
    def get_hourly_timestamps(Year):
        date_str = '1/1/'+str(Year+1)
        start = pd.to_datetime(date_str) - pd.Timedelta(days=365)
        hourly_periods = 8760
        drange = pd.date_range(start, periods=hourly_periods, freq='H')
        return drange
   
    EnergyUse['DATETIME'] = get_hourly_timestamps(Year)
    EnergyUse = EnergyUse.set_index(['DATETIME'])
    EnergyUse_D = EnergyUse.resample('D').sum()





    GridCO2_vs_EU = pd.concat([GridCO2, EnergyUse, GridCO2_DS], axis=1)
    GridCO2_vs_EU.drop(columns=[EU_BLDG]).head(24*7*2).plot(title='Grid Carbon Intensity Vs. Daily Average', colormap="Set1", figsize=(20,3))

    GridCO2_vs_EU[CO2_DIFF] = GridCO2_vs_EU[CO2_GRID] - GridCO2_vs_EU[CO2_AVG_D]

    GridCO2_vs_EU['CHARGHING?'] = GridCO2_vs_EU[CO2_DIFF] <= -_input_CO2_diff # charge battery only if difference is greater than 10
    
    

    
#     with tab1:
#         with t1_col1:
#             fig_EU = px.line(GridCO2_vs_EU[EU_BLDG],markers=False, width=800, height=200)
#             fig_EU.update_xaxes(title = 'Hour').update_yaxes(title = 'kWh').update_layout(
#                 margin=dict(l=50, r=0, t=0, b=40),
#                 showlegend=False
#             )
#             st.plotly_chart(fig_EU, use_container_width=True)
    
#     with tab2:
#         with t2_col1:
#             fig_CO2 = px.line(GridCO2_vs_EU[CO2_GRID],markers=False, width=800, height=200)
#             fig_CO2.update_xaxes(title = 'Hour').update_yaxes(title = 'gCO2').update_layout(
#                 margin=dict(l=50, r=0, t=0, b=40),
#                 showlegend=False
#             )
#             st.plotly_chart(fig_CO2, use_container_width=True)
    
    
    
    

    loads = []
    max_loads = []
    max_load = 0

    for i,j in zip(GridCO2_vs_EU['CHARGHING?'],range(len(GridCO2_vs_EU[EU_BLDG]))):
        
        if  i == False:
            max_load = max_load + GridCO2_vs_EU[EU_BLDG].iloc[j]
        else:
            max_loads.append(max_load)
            max_load = 0

    max_loads = [x for x in max_loads if x != 0] #remove zeros from list

    #max_load = round(np.quantile(max_loads, q = _max_load_quantile), 2)
    
       
    battery_size = round(np.quantile(EnergyUse_D.values.tolist(), q = _max_load_quantile), 2)
    
    
    st.sidebar.write("Battery capacity is", battery_size, "kWh")
    


    charging_rate = battery_size/4
    battery_charge = 0
    EU_battery = 0
    EU_BnB = 0
    
    battery_kWh = []
    EU_kWh_Battery = []
    EU_BldgAndBatt = []

    for i, j in zip(GridCO2_vs_EU['CHARGHING?'],range(len(GridCO2_vs_EU[EU_BLDG]))):
        
        if i == True: # Battery IS charging; (LOW CO2 intensity from the grid)
            if battery_charge < battery_size:
                battery_charge = min(battery_size, battery_charge + charging_rate) #insure charging does not exceed battery size
            else:
                pass
            
            battery_kWh.append(battery_charge)
            
            if j == 0:
                EU_battery = battery_kWh[0]
                EU_kWh_Battery.append(EU_battery)
            else:
                EU_battery = battery_charge - battery_kWh[j-1]
                EU_kWh_Battery.append(EU_battery)
            
            EU_BnB = GridCO2_vs_EU[EU_BLDG].iloc[j] + EU_battery
            EU_BldgAndBatt.append(EU_BnB)
            
        else: # Battery is NOT charging; (HIGH CO2 intensity from the grid)
            battery_charge = max(0 , battery_charge - GridCO2_vs_EU[EU_BLDG].iloc[j]) #insure charging does not go below zero
            battery_kWh.append(battery_charge)
            
            EU_battery = 0
            EU_kWh_Battery.append(EU_battery)
            
            if GridCO2_vs_EU[EU_BLDG].iloc[j] > battery_charge:
                EU_BnB = GridCO2_vs_EU[EU_BLDG].iloc[j] - battery_charge
            else:
                EU_BnB = 0
                
            EU_BldgAndBatt.append(EU_BnB)
            
    GridCO2_vs_EU['battery_charge'] = [round(elem, 2) for elem in battery_kWh]
    GridCO2_vs_EU['EU_kWh_Battery'] = [round(elem, 2) for elem in EU_kWh_Battery]
    GridCO2_vs_EU['EU_BldgAndBatt'] = [round(elem, 2) for elem in EU_BldgAndBatt]
    GridCO2_vs_EU['kgCO2_BldgAndBatt'] = GridCO2_vs_EU['EU_BldgAndBatt'] * GridCO2_vs_EU[CO2_GRID] / 1000

    #st.dataframe(GridCO2_vs_EU.drop(columns={CO2_GRID, CO2_AVG_D,'CI_gCO2_Difference'}))
    #st.dataframe(GridCO2_vs_EU)
    
    charge_frequency = GridCO2_vs_EU['battery_charge'].value_counts().sort_values(ascending=False).head(10)
    
    
    
    

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
    
    
    #st.dataframe(compare.drop(columns={'kWh_Opt0','kgCO2_Opt0','kWh_Opt1','kgCO2_Opt1'}))


    EU_B_D = compare['kWh_Opt0'].resample('D').sum().to_frame()
    EU_BnB_D = compare['kWh_Opt1'].resample('D').sum().to_frame()

    EU_compared_D = pd.concat([EU_B_D, EU_BnB_D], axis=1)

    kWh_figure = px.line(EU_compared_D, markers=True)
    kWh_figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    ).update_layout(
                legend=dict(x=0.75, y=1,traceorder="normal"),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
    
    
    
    with tab1:
        st.plotly_chart(kWh_figure, theme = "streamlit")


    CO2_B_D = compare['kgCO2_Opt0'].resample('D').sum().to_frame()
    CO2_BnB_D = compare['kgCO2_Opt1'].resample('D').sum().to_frame()

    CO2_compared_D = pd.concat([CO2_B_D, CO2_BnB_D], axis=1)
    
    
    totals_CO2 = CO2_compared_D.sum()
    
    old_CO2 = totals_CO2.iloc[0]/1000
    new_CO2 = totals_CO2.iloc[1]/1000
    differ_CO2 = old_CO2 - new_CO2
    percent_savings_CO2 = ((totals_CO2.iloc[0]-totals_CO2.iloc[1])/totals_CO2.iloc[0])*100
    
    with tab2:
        with t2_col2:
            st.subheader("Operational CO2")
            
            st.metric(label='Building only:', value=old_CO2.round(2))
            st.metric(label='Building & battery:', value=new_CO2.round(2), delta=str(-(percent_savings_CO2.round(2)))+"%", delta_color="inverse" )
            
            st.subheader('Embodied CO2')
           
            battery_embodied = battery_size * 50 / 1000
            st.metric(label='Battery manufacturing:', value=round(battery_embodied,2))
            st.caption('Note: all numbers are in CO2 Tonnes')
            
            
            CO2_payback_years = battery_embodied / differ_CO2
            
            
            st.text('Pay-back period:')
            st.header(CO2_payback_years.round(1))
            st.text('Years')
            
            
            

    kgCO2_figure = px.line(CO2_compared_D,markers=True)
    kgCO2_figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    



    #CO2_B_cumsum = compare['kgCO2_Opt0'].resample('D').sum().to_frame()
    #CO2_BnB_cumsum = compare['kgCO2_Opt1'].resample('D').sum().to_frame()


    CO2_B_cumsum = CO2_compared_D['kgCO2_Opt0'].cumsum(axis=0, skipna=True)
    CO2_BnB_cumsum = CO2_compared_D['kgCO2_Opt1'].cumsum(axis=0, skipna=True)


    CO2_compared_cumsum = pd.concat([CO2_B_cumsum, CO2_BnB_cumsum], axis=1)


    with tab2:
        with t2_col1:
            kgCO2_figure = px.line(CO2_compared_cumsum, markers=False, width=480, height=400)
            kgCO2_figure.update_xaxes(title = 'Hour').update_yaxes(title = 'gCO2').update_layout(
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

if __name__ == "__main__":
    main()