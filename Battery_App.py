import matplotlib.pyplot as plt
import requests
import numpy as np
import pandas as pd
import math
import streamlit as st
import os, urllib
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
#from datetime import timedelta


# To run the app, type the following in Terminal: 
# "streamlit run "c:/Users/IRIHA/OneDrive - Ramboll/Documents/00 Work Tasks/221118_GridCarbon/BatteryApp/Battery_app.py"

def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("README.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    #st.sidebar.header("RamBattery")
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

    
    # Define stramlit tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Carbon Emissions", "Energy Use", "All Data", "Battery Profile"])    
    
    # Define stramlit column containers
    with tab2:
        t2_col1, t2_col2 = st.columns([4.5, 1.2])
    
    with tab1:
        t1_col1, t1_col2 = st.columns([4.5, 1.2])
    
    # Define user input fields
    Year = st.sidebar.radio(
        'Choose a year for carbon Intensity data:',
        (2021,2020,2019, 2018),disabled=True, horizontal=True)
    
    FP = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/2fa_HourlyAverageElectricalPower.xlsx'
    excel_URL = st.sidebar.text_input('2- URL for hourly energy simulation (.csv): ', FP, key='System FP')
    
    _max_load_quantile = st.sidebar.slider(
        label = r'Battery capacity (% of max. daily load):',
        help = 'This value determines the battery size as a percentage from the maximum daily load of building operational energy demand.',
        value = 60,
        max_value = 100,
        min_value = 5)  # slider widget
    
    _battery_charging_rate = st.sidebar.slider(
        label = r'Battery charging rate (hours):',
        help = 'This value determines the number of hours for charging the battery.',
        value = 4,
        max_value = 12,
        min_value = 1)  # slider widget
    
    _EU_limit_low = st.sidebar.slider(
        label = r'Low limit of energy demand',
        help = 'This value determines the battery activating energy demand (battery should be charging at LOW energy demand).',
        value = 0.95,
        max_value = 1.0,
        min_value = 0.0)  # slider widget
    
    _EU_limit_high = st.sidebar.slider(
        label = r'High limit of energy demand',
        help = 'This value determines the battery activating energy demand (battery should be discharging at HIGH energy demand).',
        value = 0.05,
        max_value = 1.0,
        min_value = 0.0)  # slider widget    
    
    _charge_tolerance = st.sidebar.slider(
        label = r'CO2 intensity charging tolerance',
        help = 'This value determines the difference between current co2 intensity and daily minimum; to trigger battery chartging.',
        value = 0.05,
        max_value = 0.500,
        min_value = 0.001)  # slider widget
    
    _discharge_tolerance = st.sidebar.slider(
        label = r'CO2 intencity discharging tolerance',
        help = 'This value determines the difference between current co2 intensity and daily maximum; to trigger battery dischartging.',
        value = 0.05,
        max_value = 0.500,
        min_value = 0.001)     
    
    
    
    # Functions for loading files 
    csv_URL = r'https://raw.githubusercontent.com/islamrihan/Battery_App/main/Utilities/gCO2_' + str(Year) + '.csv' 
    # @st.cache
        
    @st.experimental_memo
    def csv_to_df(URL):
        return pd.read_csv(csv_URL)
    
    # @st.cache
    @st.experimental_memo
    def excel_to_df(URL):
        return pd.read_excel(excel_URL)
    
    
    # Define variables in use
    TIMESTAMP = 'timestamp'
    CARBONINTENSITY = 'carbon_intensity'
    EU_BLDG = 'EU_kWh_BLDG' # Enegry use of the building in kWh

    # Load csv carbon data as a dataframe
    GridCO2 = csv_to_df(csv_URL)
    GridCO2 = GridCO2.rename(columns={GridCO2.columns[0]:TIMESTAMP})
    GridCO2 = GridCO2.rename(columns={GridCO2.columns[1]:CARBONINTENSITY})
    
    # Convert the timestamp column to a datetime data type
    GridCO2[TIMESTAMP] = pd.to_datetime(GridCO2[TIMESTAMP])

    # Extract the date from each timestamp
    GridCO2['date'] = GridCO2[TIMESTAMP].dt.date

    # Find the lowest, hightest, and mean carbon intensity value for each day
    GridCO2['min_intensity'] = GridCO2.groupby(GridCO2['date'])[CARBONINTENSITY].transform('min')
    GridCO2['max_intensity'] = GridCO2.groupby(GridCO2['date'])[CARBONINTENSITY].transform('max')
    GridCO2['mean_intensity'] = GridCO2.groupby(GridCO2['date'])[CARBONINTENSITY].transform('mean')

    # Set timestamp values as index column
    GridCO2 = GridCO2.set_index(TIMESTAMP)

    # Get energy use data from excel   
    EnergyUse = excel_to_df(excel_URL)['Unnamed: 61']
    EnergyUse = EnergyUse.drop(index = [0,1]).reset_index().iloc[:, [1]].rename(columns = {'Unnamed: 61':EU_BLDG})
    
    # Set timestamp column as index
    EnergyUse['DATETIME'] = GridCO2.index
    EnergyUse = EnergyUse.set_index(['DATETIME'])
    


    GridCO2_vs_EU = pd.concat([GridCO2, EnergyUse], axis=1)

    

       
    EU_low = GridCO2_vs_EU[EU_BLDG].quantile(_EU_limit_low)
    EU_high = GridCO2_vs_EU[EU_BLDG].quantile(_EU_limit_high)
    
    # Function for battery action logic
    def determine_battery_action(carbon_intensity, min_intensity, max_intensity, energy_demand):
        """
        Determines whether the battery should be charged or discharged based on the carbon intensity of the grid
        and the energy demand of the building.
        """
        
        if math.isclose(carbon_intensity, min_intensity, rel_tol=_charge_tolerance) and energy_demand < EU_low:
            # Charge the battery when the carbon intensity is low and the energy demand is low
            return "charge"
        elif math.isclose(carbon_intensity, max_intensity, rel_tol=_discharge_tolerance) and energy_demand > EU_high:
            # Discharge the battery when the carbon intensity is high and the energy demand is high
            return "discharge"
        else:
            # Do not charge or discharge the battery in other cases
            return "hold"
    
    # Function for creating dataframe from battery action data
    def optimize_battery(carbon_intensity_data, min_intensity_data, max_intensity_data, energy_demand_data):
        """
        Optimizes the charging and discharging of the battery based on the carbon intensity of the grid
        and the energy demand of the building.
        """
        # Load the carbon intensity and energy demand data into a Pandas DataFrame
        df = pd.DataFrame({"carbon_intensity": carbon_intensity_data, "min_intensity": min_intensity_data, "max_intensity": max_intensity_data, "energy_demand": energy_demand_data})

        # Determine the action for the battery for each hour
        df["battery_action"] = df.apply(lambda row: determine_battery_action(row["carbon_intensity"], row["min_intensity"], row["max_intensity"], row["energy_demand"]), axis=1)

        return df

    #dataframe for battery action logic
    b_dataframe = optimize_battery(GridCO2[CARBONINTENSITY], GridCO2['min_intensity'], GridCO2['max_intensity'], GridCO2_vs_EU[EU_BLDG])




    # Add battery action column to the dataframe 
    GridCO2_vs_EU['battery_action'] = b_dataframe['battery_action']
    
    # Applying battery action logic on battery hourly profile
    battery_size = round(_max_load_quantile/100 * EnergyUse.resample('D').sum().values.max(), 2)
    charging_rate = battery_size/_battery_charging_rate
    battery_charge = 0
    EU_battery = 0
    EU_BnB = 0
    
    battery_kWh = []
    EU_kWh_Battery = []
    EU_BldgAndBatt = []

    for i, j in zip(b_dataframe['battery_action'],range(len(GridCO2_vs_EU))):
        
        # Battery is CHARGING; (LOW CO2 intensity from the grid)
        if i == 'charge': 
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
        elif i == 'discharge': 
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
        
        # Battery is ON HOLD (neither charging nor discharging);    
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
    GridCO2_vs_EU['kgCO2_BldgAndBatt'] = GridCO2_vs_EU['EU_BldgAndBatt'] * GridCO2_vs_EU[CARBONINTENSITY] / 1000


    compare = pd.DataFrame()
    compare['kWh_Opt0'] = GridCO2_vs_EU[EU_BLDG].apply(lambda x: round(x, 2))
    compare['kWh_Opt0_cumsum'] = compare['kWh_Opt0'].cumsum(axis=0, skipna=True)

    compare['kgCO2_Opt0'] = compare['kWh_Opt0'] * GridCO2_vs_EU[CARBONINTENSITY] / 1000 
    compare['kgCO2_Opt0_cumsum'] = compare['kgCO2_Opt0'].cumsum(axis=0, skipna=True)

    compare['kWh_Opt1'] = GridCO2_vs_EU['EU_BldgAndBatt']
    compare['kWh_Opt1_cumsum'] = compare['kWh_Opt1'].cumsum(axis=0, skipna=True)

    compare['kgCO2_Opt1'] = GridCO2_vs_EU['kgCO2_BldgAndBatt'] 
    compare['kgCO2_Opt1_cumsum'] = compare['kgCO2_Opt1'].cumsum(axis=0, skipna=True)


    compare = compare.round(decimals=2)



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

    
    
    with tab2:
        with t2_col2:
            st.markdown('<p style="color:Grey; font-size: 22px;">Energy Use', unsafe_allow_html=True)
            st.caption('in MWh')
                       
            st.metric(label='Building Only:', value=int(old_EU))
            st.metric(label='Building with Battery:', value=int(new_EU), delta=str(-(percent_savings_EU.round(1)))+"%", delta_color="off" )
            
        with t2_col1:
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
    with tab1:
        
        # Add side bar highlighted values
        with t1_col2:
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
        with t1_col1:
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
        st.dataframe(GridCO2_vs_EU.drop(columns=['date']))
        
        CO2_figure = px.line(b_dataframe.drop(columns=['battery_action','energy_demand']).astype(float),markers=False)
        CO2_figure.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ).update_layout(
                    legend=dict(x=0.75, y=1,traceorder="normal"),
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True
                )
        
        st.plotly_chart(CO2_figure, theme = 'streamlit')
        st.line_chart(compare.drop(columns=['kgCO2_Opt0_cumsum','kgCO2_Opt1_cumsum','kWh_Opt0_cumsum','kWh_Opt1_cumsum']))
         

    
    
            
        
        
    with tab4:
        st.dataframe(GridCO2_vs_EU.drop(columns=['date', 'min_intensity', 'max_intensity','mean_intensity']))
        st.line_chart(GridCO2_vs_EU[['battery_charge']])




if __name__ == "__main__":
    main()