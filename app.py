import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from recommenders.models.lightfm.lightfm_utils import similar_items


########## Title for the Web App ##########
st.set_page_config(layout="wide")
st.title("Recommended for You!")
st.markdown("This application is for recommending restaurants to visit for users in Singapore Central 🍔🍕🍹🍺")
st.sidebar.title("Singapore Central Restaurant Recommendation Platform")
st.sidebar.subheader("DSI Capstone Project")
st.sidebar.markdown("By: Zavier Soon")

buildings = pd.read_pickle('../code/app_files/buildings.pkl')
restaurant = pd.read_pickle('../code/app_files/restaurant.pkl')

buildings_list = list(buildings['address'].sort_values(ascending = True))
restaurant_list = list(restaurant['name'].sort_values(ascending = True))

lightfm_model = pickle.load(open('../code/app_files/lightfm_model.pkl', 'rb'))
item_features = pickle.load(open('../code/app_files/item_feature.pkl', 'rb'))

clustering_model = pickle.load(open('../code/app_files/clustering_model.pkl', 'rb'))

def main():
    model_type = st.sidebar.selectbox('Recommendation Type', ['','Location-Based Recommender (New User)', 'Similar-Restaurant Recommender (New User/Existing User)']
                                      , format_func = lambda x: 'Select an option' if x == '' else x)
    
    if model_type == 'Location-Based Recommender (New User)':
        st.subheader('Location-Based Recommendation System')
        Location = st.selectbox('📍Please enter your current location', [''] + buildings_list, format_func = lambda x: 'Input current location' if x == '' else x)
        PriceLevel = st.selectbox('🏷️ Price',  ('','Inexpensive', 'Moderate', 'Expensive'), format_func = lambda x: 'Select an option' if x == '' else x)
        
        if st.button("Recommend Restaurant"):
                current_location = buildings[buildings['address'] == Location]
                cluster = clustering_model.predict(np.array([current_location['lat'], current_location['long']]).reshape(1,-1))[0]
                results = restaurant[(restaurant['cluster'] == cluster) & (restaurant['price'] == PriceLevel)].reset_index().head(10)
                
                col1, col2 = st.columns((1,1))
                # Geographical Plot ofTop 10 Recommended Restaurants
                with col1:
                    fig_map = go.Figure(data = [go.Scattermapbox(lat = results['lat'], lon = results['long'], 
                                                         mode = 'markers+text', name = 'Recommended Restaurant',
                                                         marker = dict(size = 12, color = 'blue', opacity = 0.8), hoverinfo = "lon+lat+text",
                                                         text = [results['name'][i] + '<br>' + results['address'][i] for i in range(len(results))]),
                                               go.Scattermapbox(lat = current_location['lat'], lon = current_location['long'], 
                                                         mode = 'markers+text', name = 'Current Location',
                                                         marker = dict(size = 12, color = 'red', opacity = 0.8), hoverinfo = "lon+lat+text",
                                                         text = current_location['address']),])
                                                          
            
                    fig_map.update_layout(mapbox_style="open-street-map", width = 800, height = 600, margin = {"r":0,"t":0,"l":0,"b":0}, 
                                          mapbox = dict(center = go.layout.mapbox.Center(lat = 1.357107, lon = 103.8194992), zoom = 10),
                                         legend = dict(yanchor = 'top', y= 0.99, xanchor = 'left', x = 0.01)) 
            
                    st.plotly_chart(fig_map, use_container_width=True)
               
                # Table of Top 10 Recommended Restaurants
                with col2:
                    fig_table = go.Figure(data = [go.Table(columnorder = [1,2], columnwidth = [200,400],
                                                           header = dict(values = list(['<b>Name</b>', '<b>Address</b>']), fill_color = 'royalblue', line_color='darkslategray',
                                                                         align = ['left', 'center'], font = dict(color = 'black', size = 15), height=40),
                                                           cells = dict(values = [results['name'], results['address']], fill = dict(color=['paleturquoise', 'white']),
                                                                        line_color = 'darkslategray', align = 'left', font = dict(color='black', size = 13), height = 30))])
            
                    fig_table.update_layout(width = 800, height = 600, margin = {"r":0,"t":0,"l":0,"b":0})
     
                    st.plotly_chart(fig_table, use_container_width=True)
            
            
    if model_type == 'Similar-Restaurant Recommender (New User/Existing User)':
        st.subheader('Similar-Restaurant Recommendation System')
        st.markdown("Please select a restaurant similar to the one you'd like to visit")
        Restaurant = st.selectbox('🍴Restaurant Name', [''] + restaurant_list, format_func = lambda x: 'Select an option' if x == '' else x)
        PriceLevel = st.selectbox('🏷️ Price',  ('','Inexpensive', 'Moderate', 'Expensive'), format_func = lambda x: 'Select an option' if x == '' else x)
        
        if st.button("Recommend Restaurant"):
            item_id = restaurant[restaurant['name'] == Restaurant]['itemID'].values[0]
            results = similar_items(item_id = item_id, item_features = item_features, model = lightfm_model, N = len(restaurant)-1)
            results = results.merge(restaurant, how = 'inner', on = 'itemID') 
            results = results[results['price'] == PriceLevel].reset_index().head(10)
            
            col1, col2 = st.columns((1,1))
            # Geographical Plot ofTop 10 Recommended Restaurants
            with col1:
                fig_map = go.Figure(go.Scattermapbox(lat = results['lat'], lon = results['long'], 
                                                     mode = 'markers+text',
                                                     marker = dict(size = 12, color = 'blue', opacity = 0.8),
                                                     text = [results['name'][i] + '<br>' + results['address'][i] for i in range(len(results))]))
                                                          
            
                fig_map.update_layout(mapbox_style="open-street-map", width = 800, height = 600, margin = {"r":0,"t":0,"l":0,"b":0}, 
                                      mapbox = dict(center = go.layout.mapbox.Center(lat = 1.357107, lon = 103.8194992), zoom= 10)) 
            
                st.plotly_chart(fig_map, use_container_width=True)
               
            # Table of Top 10 Recommended Restaurants
            with col2:
                fig_table = go.Figure(data = [go.Table(columnorder = [1,2], columnwidth = [200,400],
                                                       header = dict(values = list(['<b>Name</b>', '<b>Address</b>']), fill_color = 'royalblue', line_color='darkslategray',
                                                                     align = ['left', 'center'], font = dict(color='black', size = 15), height=40),
                                                       cells = dict(values = [results['name'], results['address']], fill = dict(color=['paleturquoise', 'white']),
                                                                    line_color = 'darkslategray', align = 'left', font = dict(color = 'black', size = 13), height = 30))])
            
                fig_table.update_layout(width = 800, height = 600, margin = {"r":0,"t":0,"l":0,"b":0})
     
                st.plotly_chart(fig_table, use_container_width=True)
            
recommendation = main()
    
    
    

    