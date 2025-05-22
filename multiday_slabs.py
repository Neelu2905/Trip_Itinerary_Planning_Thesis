from gurobipy import Model, GRB, quicksum
import pandas as pd
import json
import streamlit as st
import math
import datetime
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import re
import numpy as np 

# Custom styling
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .box {
        border-radius: 10px;
        padding: 10px;
        background-color: #f4f4f4;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def convert_to_minutes(time_obj):
            return time_obj.hour * 60 + time_obj.minute

# Title
st.title("🚀 Trip Planner")

# Dropdown for City Selection
st.subheader(" Select Your City")
cities = ["Select a City","Budapest", "Delhi", "Osaka", "Glasgow", "Vienna", "Perth", "Edinburgh"]
city = st.selectbox("", cities)

st.markdown(f"<p class='big-font'>📍 City Selected: {city}</p>", unsafe_allow_html=True)

if city!="Select a City":
    # Paths to Excel files
    utility_data_path = "Updated Travel Data.xlsx"
    cost_data_path = "Cost Data.xlsx"

    utility = pd.read_excel(utility_data_path, sheet_name=city)
    cost_data = pd.read_excel(cost_data_path, sheet_name=city)

# Show interesting places box once city is selected
if city!="Select a City":
    # Wrap everything (title + table) inside a properly aligned gray box
    st.markdown("""
        <div style='background-color:#f4f4f4; padding:20px; border-radius:10px; 
                    width:100%; margin:auto; text-align:center;'>
            <h3>Interesting Places to Visit in {}</h3>
            <table style='width:100%; border-collapse: collapse; text-align: left;'>
                <thead>
                    <tr style='background-color: #bbb;'>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>POI ID</th>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>POI Name</th>
                        <th style='padding: 10px; border-bottom: 2px solid #000;'>Theme</th>
                    </tr>
                </thead>
                <tbody>
                    {}
                </tbody>
            </table>
        </div>
    """.format(
        city,
        ''.join(
            f"<tr><td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.poiID}</td>"
            f"<td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.poiName}</td>"
            f"<td style='padding: 10px; border-bottom: 1px solid #ccc;'>{row.theme}</td></tr>"
            for _, row in utility.iterrows()
        )
    ), unsafe_allow_html=True)

    st.markdown("#")

    # ✅ Select the day for planning
    selected_day = st.selectbox("Select the day for your itinerary:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    st.subheader("Select Must-See POIs")
    
    poi_ids = utility['poiID'].tolist()
    must_see_pois = st.multiselect("If you have any preference then choose the must-see POIs:", poi_ids, placeholder="Choose multiple options")

    st.subheader("Select Excluded POIs")
    
    excluded_pois = st.multiselect("If you have any preference then choose the POIs to exclude:", poi_ids, placeholder="Choose multiple options")

    ordering_constraints=[]
    # Ordering Constraints Checkbox
    st.write("")
    st.subheader("Ordering Constraints")
    # Checkbox with bigger text
    constraints = st.checkbox("I want ordering constraints", help="Check this to add ordering constraints")

    # If user selects the checkbox, show text input for constraints
    if constraints:
        ordering_constraints = st.text_input(
            "Enter the constraints in this format: (a,b),(c,d) and so on",
            placeholder="(1,2),(3,4)"
        )

        # Parsing function to convert string input into list of tuples
        def parse_constraints(input_text):
            try:
                # Extract tuples using regex
                matches = re.findall(r"\((\d+),(\d+)\)", input_text)
                parsed_constraints = [(int(a), int(b)) for a, b in matches]  # Convert to list of tuples
                return parsed_constraints
            except Exception as e:
                return str(e)  # Return error message if parsing fails

        # Process user input
        if ordering_constraints:
            ordering_constraints = parse_constraints(ordering_constraints)
            if not ordering_constraints:
                st.warning("⚠️ Invalid format! Please enter in (a,b) format.")
            # else:
            #     st.warning("⚠️ Invalid format! Please enter in (a,b) format.")

    st.subheader("Category Constraints")
    category_constraints = st.checkbox("I want category constraints", help="Check this to add category constraints")
    category_counts = utility["theme"].value_counts()

    theme_bounds = {}
   
    if category_constraints:
        # Convert your theme column counts into a DataFrame
        category_counts = utility["theme"].value_counts().reset_index()
        category_counts.columns = ["Theme", "Count"]

        # Add two new columns for lower & upper bounds (initially empty or None)
        category_counts["Lower bound"] = 0
        category_counts["Upper bound"] = category_counts["Count"]

        # Display the editable table
        # If you're on an older Streamlit version, replace `st.data_editor` with `st.experimental_data_editor`
        # Make only the last two columns editable using column_config
        edited_df = st.data_editor(
            category_counts,
            column_config={
                "Theme": st.column_config.Column(disabled=True),       # read-only
                "Count": st.column_config.Column(disabled=True),       # read-only
                "Lower bound": st.column_config.Column(disabled=False),# editable
                "Upper bound": st.column_config.Column(disabled=False) # editable
            },
            use_container_width=True
        )

        edited_df["Lower bound"] = edited_df.apply(
            lambda row: row["Lower bound"]
            if 0 <= row["Lower bound"] <= row["Count"]
            else 0,  # or e.g. row["Count"] - 1
            axis=1
        )

        edited_df["Upper bound"] = edited_df.apply(
            lambda row: row["Upper bound"]
            if row["Lower bound"] <= row["Upper bound"] <= row["Count"]
            else row["Count"],  # or e.g. row["Count"] - 1
            axis=1
        )

        col_left, col_mid, col_right = st.columns([1, 5, 1])  # Middle is wider

        with col_mid:
            st.dataframe(edited_df, use_container_width=True)
        
         # Option 1: Using dictionary comprehension + iterrows
        theme_bounds = {
            row["Theme"]: (row["Lower bound"], row["Upper bound"])
            for _, row in edited_df.iterrows()
        }

    # User input for number of days
    num_days = st.number_input("How many days do you want your trip to be?", min_value=1, max_value=30, value=1, step=1)

    # Convert to list of days for use in the model
    days = list(range(num_days))
    
    # Budget Inputs (Inline with Columns)
    st.subheader(" Travel Budget")
    # col1, col2 = st.columns(2)
    # with col1:
    #     time_budget = st.number_input("⏳ Time Budget per day (in hours)", min_value=0.0, step=0.5, format="%.2f")
    # with col2:
    #     cost_budget = st.number_input("💸 Cost Budget for trip (in INR)", min_value=0.0, step=10.0, format="%.2f")

    # time_budget_first_day= st.number_input("⏰ Enter time budget for the first day (in minutes)")
    # time_budget_last_day = st.number_input("⏰ Enter time budget for the last day (in minutes)")
    
    # start_time=600
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛬 First Day")
        first_day_start = st.time_input("Start Time (First Day)", value=datetime.time(10, 0))
        first_day_end = st.time_input("End Time (First Day)", value=datetime.time(18, 0))
        
    with col2:
        st.markdown("### 📅 Intermediate Days")
        daily_start = st.time_input("Start Time (Intermediate Days)", value=datetime.time(9, 0))
        daily_end = st.time_input("End Time (Intermediate Days)", value=datetime.time(19, 0))

    with col1:
        st.markdown("### 🛫 Last Day")
        last_day_start = st.time_input("Start Time (Last Day)", value=datetime.time(9, 0))
        last_day_end = st.time_input("End Time (Last Day)", value=datetime.time(17, 0))

    with col2:
        st.markdown("### 💸 Cost")
        cost_budget = st.number_input("Cost Budget for trip (in INR)", min_value=0.0, step=10.0, format="%.2f")

    
    time_budget_first_day = convert_to_minutes(first_day_end) - convert_to_minutes(first_day_start)
    time_budget = convert_to_minutes(daily_end) - convert_to_minutes(daily_start)
    time_budget_last_day = convert_to_minutes(last_day_end) - convert_to_minutes(last_day_start)

    start_time_first_day = convert_to_minutes(first_day_start)
    start_time = convert_to_minutes(daily_start)
    start_time_last_day = convert_to_minutes(last_day_start)
    
    # Coordinates Input (Formatted Display)
    st.subheader(" Coordinates")

    # st.markdown("### 🏁 Source Location")
    # col1, col2 = st.columns(2)
    # with col1:
    #     source_lat = st.number_input("📍 Source Latitude", format="%.6f", key="source_lat")
    # with col2:
    #     source_lon = st.number_input("📍 Source Longitude", format="%.6f", key="source_lon")

    # st.markdown("### 🏨 Hotel Location")
    # col3, col4 = st.columns(2)
    # with col3:
    #     hotel_lat = st.number_input("📍 Hotel Latitude", format="%.6f", key="hotel_lat")
    # with col4:
    #     hotel_lon = st.number_input("📍 Hotel Longitude", format="%.6f", key="hotel_lon")

    # st.markdown("### 🎯 Destination Location")
    # col5, col6 = st.columns(2)
    # with col5:
    #     dest_lat = st.number_input("📍 Destination Latitude", format="%.6f", key="dest_lat")
    # with col6:
    #     dest_lon = st.number_input("📍 Destination Longitude", format="%.6f", key="dest_lon")
    
    source_lat=34.655
    source_lon=135.43
    hotel_lat=34.696
    hotel_lon=135.51
    dest_lat=34.645
    dest_lon=135.504
    st.write(source_lat)
    st.write(source_lon)
    st.write(hotel_lat)
    st.write(hotel_lon)
    st.write(dest_lat)
    st.write(dest_lon)

    st.subheader("Trip Flow Summary")

    trip_summary_html = """
    <div style='background-color: #f5f5dc; padding: 20px; border-radius: 10px; 
                border: 2px solid #d2b48c; color: #5c4033; font-size: 17px;'>
        <ul style='margin: 0; padding-left: 20px;'>
            <li><strong>First Day:</strong> Start your journey from <b>Source</b> and travel to your <b>Hotel</b>.</li>
            <li><strong>Intermediate Days:</strong> Daily trips start and end at the <b>Hotel</b>.</li>
            <li><strong>Last Day:</strong> Depart from the <b>Hotel</b> and head to your final <b>Destination</b>.</li>
        </ul>
    </div>
    """

    st.markdown(trip_summary_html, unsafe_allow_html=True)

    # Display Selected Data (Styled)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📌 Selected Travel Details")

    st.markdown(
        f"""
        <div class='box'>
        <p><b>City:</b> {city}</p>
        <p><b>Number of days in the trip:</b> {num_days} days</p>
        <p><b>Time Budget of first day:</b> {time_budget_first_day} Minutes</p>
        <p><b>Time Budget on other days:</b> {time_budget} Minutes</p>
        <p><b>Time Budget of last day:</b> {time_budget_last_day} Minutes</p>
        <p><b>Cost Budget:</b> {cost_budget} Rupees</p>
        <p><b>Source Coordinates:</b> ({source_lat}, {source_lon})</p>
        <p><b>Hotel Coordinates:</b> ({hotel_lat}, {hotel_lon})</p>
        <p><b>Destination Coordinates:</b> ({dest_lat}, {dest_lon})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Example Haversine function returning distance in meters
    def haversine_distance_meters(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points (lat1, lon1) and
        (lat2, lon2) on Earth in meters.
        """
        # Radius of Earth in meters
        R = 6371000  # ~6,371 km in meters

        # Convert degrees to radians
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        # Haversine formula
        a = (math.sin(d_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        # Distance in meters
        distance_meters = R * c
        return distance_meters



    # Add distance columns (in meters) to the DataFrame
    utility["distance from source"] = utility.apply(
        lambda row: haversine_distance_meters(source_lat, source_lon, row["lat"], row["long"]),
        axis=1
    )

    utility["distance from destination"] = utility.apply(
        lambda row: haversine_distance_meters(dest_lat, dest_lon, row["lat"], row["long"]),
        axis=1
    )
    
    utility["distance from hotel"] = utility.apply(
        lambda row: haversine_distance_meters(hotel_lat, hotel_lon, row["lat"], row["long"]),
        axis=1
    )

    # Extract relevant data
    poi_ids = utility['poiID'].tolist()
    #poi_ids = [0] + poi_ids + [poi_ids[-1] + 1]
    poi_ids = [0] + poi_ids + [poi_ids[-1] + 1, poi_ids[-1] + 2]

    # ## APPENDING SOURCE COST DATA IN COST_DATA DATAFRAME

    # In[427]:
    new_df = utility[['poiID', 'distance from source']].copy()
    new_df.insert(0,'from',0)
    new_df.rename(columns={'poiID': 'to', 'distance from source': 'cost'}, inplace=True)
    poi_dict = dict(zip(utility["poiID"], utility["theme"]))
    profit_dict = dict(zip(utility["poiID"], utility["Utility Score"]))
    new_df['profit'] = new_df['to'].map(profit_dict)
    new_df['category'] = new_df['to'].map(poi_dict)
    # Step 1: Make a copy of new_df
    reversed_df = new_df.copy()
    # Step 2: Swap the 'from' and 'to' columns
    reversed_df['from'], reversed_df['to'] = reversed_df['to'], reversed_df['from']
    # Step 3: Set profit to 0 and category to 'hotel'
    reversed_df['profit'] = 0
    reversed_df['category'] = 'hotel'
    # Step 4: Append reversed_df to new_df
    new_df = pd.concat([new_df, reversed_df], ignore_index=True)
    # Now new_df contains both (i -> j) and (j -> i) rows.


    # ## APPENDING DESTINATION COST DATA IN COST_DATA DATAFRAME

    new_df_dest = utility[['poiID', 'distance from destination']].copy()
    new_df_dest.insert(0, 'from', poi_ids[-1])
    new_df_dest.rename(columns={'poiID': 'to', 'distance from destination': 'cost'}, inplace=True)
    new_df_dest['profit'] = new_df_dest['to'].map(profit_dict)
    new_df_dest['category'] = new_df_dest['to'].map(poi_dict)
    # Step 1: Make a copy of new_df_dest
    reversed_df_dest = new_df_dest.copy()
    # Step 2: Swap the 'from' and 'to' columns
    reversed_df_dest['from'], reversed_df_dest['to'] = reversed_df_dest['to'], reversed_df_dest['from']
    # Step 3: Set profit to 0 and category to 'hotel'
    reversed_df_dest['profit'] = 0
    reversed_df_dest['category'] = 'hotel'
    # Step 4: Append reversed_df_dest to new_df_dest
    new_df_dest = pd.concat([new_df_dest, reversed_df_dest], ignore_index=True)
    # Now new_df_dest contains both (i -> j) and (j -> i) rows.
    cost_data = pd.concat([cost_data, new_df, new_df_dest], ignore_index=True)

    # ## APPENDING HOTEL COST DATA IN COST_DATA DATAFRAME

    new_df_hotel = utility[['poiID', 'distance from hotel']].copy()
    new_df_hotel.insert(0, 'from', poi_ids[-2])  # hotel node ID
    new_df_hotel.rename(columns={'poiID': 'to', 'distance from hotel': 'cost'}, inplace=True)
    new_df_hotel['profit'] = new_df_hotel['to'].map(profit_dict)
    new_df_hotel['category'] = new_df_hotel['to'].map(poi_dict)

    # Step 1: Make a copy of new_df_hotel
    reversed_df_hotel = new_df_hotel.copy()
    # Step 2: Swap the 'from' and 'to' columns
    reversed_df_hotel['from'], reversed_df_hotel['to'] = reversed_df_hotel['to'], reversed_df_hotel['from']
    # Step 3: Set profit to 0 and category to 'hotel'
    reversed_df_hotel['profit'] = 0
    reversed_df_hotel['category'] = 'hotel'

    # Step 4: Append reversed_df_hotel to new_df_hotel
    new_df_hotel = pd.concat([new_df_hotel, reversed_df_hotel], ignore_index=True)

    # Final step: add to cost_data
    cost_data = pd.concat([cost_data, new_df_hotel], ignore_index=True)

    # Calculate required distances using haversine
    distance_source_to_dest = haversine_distance_meters(source_lat, source_lon, dest_lat, dest_lon)
    distance_source_to_hotel = haversine_distance_meters(source_lat, source_lon, hotel_lat, hotel_lon)
    distance_hotel_to_source = distance_source_to_hotel  # symmetric
    distance_hotel_to_dest = haversine_distance_meters(hotel_lat, hotel_lon, dest_lat, dest_lon)
    distance_dest_to_hotel = distance_hotel_to_dest  # symmetric

    # Append all pairs as dictionaries
    rows_to_add = [
        {'from': 0, 'to': poi_ids[-1], 'cost': distance_source_to_dest, 'profit': 0, 'category': 'hotel'},
        {'from': poi_ids[-1], 'to': 0, 'cost': distance_source_to_dest, 'profit': 0, 'category': 'hotel'},

        {'from': 0, 'to': poi_ids[-2], 'cost': distance_source_to_hotel, 'profit': 0, 'category': 'hotel'},
        {'from': poi_ids[-2], 'to': 0, 'cost': distance_hotel_to_source, 'profit': 0, 'category': 'hotel'},

        {'from': poi_ids[-2], 'to': poi_ids[-1], 'cost': distance_hotel_to_dest, 'profit': 0, 'category': 'hotel'},
        {'from': poi_ids[-1], 'to': poi_ids[-2], 'cost': distance_dest_to_hotel, 'profit': 0, 'category': 'hotel'},
    ]
    
    # 3. Convert that list into a small DataFrame
    new_rows_df = pd.DataFrame(rows_to_add)
    
    # 4. Concatenate with your existing cost_data
    cost_data = pd.concat([cost_data, new_rows_df], ignore_index=True)
    
    utility.drop(columns=['distance from source', 'distance from destination', 'distance from hotel'], inplace=True)

    cost_data.sort_values(by=['from', 'to'], ascending=[True, True], inplace=True)

    # ## APPENDING SOURCE AND DESTINATION ROWS IN UTILITY DATAFRAME
    rows_to_add = [
        {
            'poiID': 0,
            'poiName': 'source',
            'lat': source_lat,
            'long': source_lon,
            'theme': 'hotel',
            'Avg Visiting TIme': 0,
            'Utility Score': 0,
            'fees': 0,
            'opening time': datetime.time(hour=0, minute=0, second=0),
            'closing time': datetime.time(hour=23, minute=59, second=59),
            'Monday': 1,
            'Tuesday': 1,
            'Wednesday': 1,
            'Thursday': 1,
            'Friday': 1,
            'Saturday': 1,
            'Sunday': 1,
        },
        {
            'poiID': poi_ids[-1],
            'poiName': 'destination',
            'lat': dest_lat,
            'long': dest_lon,
            'theme': 'hotel',
            'Avg Visiting TIme': 0,
            'Utility Score': 0,
            'fees': 0,
            'opening time': datetime.time(hour=0, minute=0, second=0),
            'closing time': datetime.time(hour=23, minute=59, second=59),
            'Monday': 1,
            'Tuesday': 1,
            'Wednesday': 1,
            'Thursday': 1,
            'Friday': 1,
            'Saturday': 1,
            'Sunday': 1,
        },
        {
            'poiID': poi_ids[-2],
            'poiName': 'hotel',
            'lat': hotel_lat,
            'long': hotel_lon,
            'theme': 'hotel',
            'Avg Visiting TIme': 0,
            'Utility Score': 0,
            'fees': 0,
            'opening time': datetime.time(hour=0, minute=0, second=0),
            'closing time': datetime.time(hour=23, minute=59, second=59),
            'Monday': 1,
            'Tuesday': 1,
            'Wednesday': 1,
            'Thursday': 1,
            'Friday': 1,
            'Saturday': 1,
            'Sunday': 1,
        }
    ]

    # 3. Convert that list into a small DataFrame
    new_rows_df = pd.DataFrame(rows_to_add)
    
    # 4. Concatenate with your existing cost_data
    utility = pd.concat([utility, new_rows_df], ignore_index=True)
    
    utility.sort_values(by = 'poiID', inplace = True)

    # st.dataframe(utility)
    # st.dataframe(cost_data)

    # ## CREATING REQUIRED DATA STRUCTURES

    visit_times = dict(zip(utility['poiID'], utility['Avg Visiting TIme']))
    utility_scores = dict(zip(utility['poiID'], utility['Utility Score']))
    poi_lat = dict(zip(utility['poiID'], utility['lat']))
    poi_long = dict(zip(utility['poiID'], utility['long']))

    # st.write(poi_lat)
    # st.write(poi_long)


    # Create a dictionary for travel times in minutes (cost in meters converted to km then multiplied by 15)
    travel_times_walking = {(row['from'], row['to']): (row['cost'] / 1000) * 15 for _, row in cost_data.iterrows()} # 4kmph
    travel_times_taxi = {(row['from'], row['to']): (row['cost'] / 1000) * 2 for _, row in cost_data.iterrows()} #30kmph

    # Extract opening and closing times into dictionaries
    opening_times = utility.set_index("poiID")["opening time"].to_dict()
    closing_times = utility.set_index("poiID")["closing time"].to_dict()

    # Dictionary storing taxi cost per km

    taxi_cost_per_km = {
        "Delhi": 45,
        "Budapest": 90,
        "Vienna": 345,
        "Osaka": 300,
        "Edinburgh": 135,
        "Glasgow": 115,
        "Perth": 100,
        "Toronto": 260
    }

    taxi_cost_per_meter = taxi_cost_per_km[city] / 1000  

    st.markdown("""
        <style>
        div.stButton > button {
            background-color: red;  /* Blue background */
            color: white;               /* White text */
            padding: 10px 20px;         /* Padding for a better look */
            border: none;               /* Remove border */
            border-radius: 8px;         /* Rounded corners */
            font-size: 16px;            /* Increase font size */
            font-weight: bold;          /* Bold text */
            cursor: pointer;            /* Pointer on hover */
            transition: background-color 0.3s ease; /* Smooth transition */
            margin-left:37%;
        }
        div.stButton > button:hover {
            background-color: #faad9c;  /* Darker blue on hover */
        }
        </style>
        """, unsafe_allow_html=True)


    # Button to Generate Output
    if st.button("GENERATE ITINERARY"):
        
        # ## MODEL DECLARATION, VARIABLES & OBJECTIVE FUNCTION
        model = Model("ILP_Model_1")
        
        

        # Decision variables: y[i] = 1 if POI i is included in the itinerary, 0 otherwise
        y = model.addVars(poi_ids,days, vtype=GRB.BINARY, name="y")
        # Introduce new binary variables for travel between POIs
        z = model.addVars(poi_ids, poi_ids, days, vtype=GRB.BINARY, name="z")
        # start_time = model.addVar(vtype=GRB.CONTINUOUS, name="start_time")
        arrival_time = model.addVars(poi_ids, days, vtype=GRB.CONTINUOUS, name="arrival_time")
        ppoi = model.addVars(poi_ids,days, vtype=GRB.CONTINUOUS, lb=0, ub=1.0, name="ppoi")
        N = len(poi_ids)  # Total number of POIs
        w = model.addVars(poi_ids, poi_ids, days, vtype=GRB.BINARY, name="w")  # Walking
        x = model.addVars(poi_ids, poi_ids, days, vtype=GRB.BINARY, name="x")  # Taxi
        #tracks each poi is visited on which day
        day_visit = model.addVars(poi_ids, vtype=GRB.INTEGER, name="day_visit")
        s1 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s1")  # [0.5, 0.6)
        s2 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s2")  # [0.6, 0.7)
        s3 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s3")  # [0.7, 0.8)
        s4 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s4")  # [0.8, 0.9)
        s5 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s5")  # [0.9, 1.0)
        s6 = model.addVars(poi_ids, days, vtype=GRB.BINARY, name="s6")  # [1.0]
        effective_utility = model.addVars(poi_ids, days, vtype=GRB.CONTINUOUS, name="effective_utility")

        for i in poi_ids:
            for d in days:
                model.addConstr(
                    effective_utility[i,d] == utility_scores[i] * (0.5 * s1[i,d] + 0.6 * s2[i,d] + 0.7 * s3[i,d] + 0.8 * s4[i,d] + 0.9 * s5[i,d] + 1.0 * s6[i,d]),
                    name=f"utility_mapping_{i}"
                )
        
        # Objective: Maximize the sum of utility scores for selected POIs
        model.setObjective(quicksum(effective_utility[i,d] for i in poi_ids for d in days), GRB.MAXIMIZE) 
        
        # only one of the slabs can be picked
        for d in days:
            for i in poi_ids:
                model.addConstr(s1[i,d] + s2[i,d] + s3[i,d] +s4[i,d] + s5[i,d] + s6[i,d] <= 1, name=f"slab_onehot_{i}_day{d}")

        # Bind ppoi[i] to the respective slab range {relation between ppoi and s_i}
        for d in days:
            for i in poi_ids:
                model.addConstr(ppoi[i,d] >= 0.5 * s1[i,d] + 0.6 * s2[i,d] + 0.7 * s3[i,d] + 0.8 * s4[i,d] + 0.9 * s5[i,d] + 1*s6[i,d], name=f"ppoi_min_slab_{i}_day{d}")
                model.addConstr(ppoi[i,d] <= (0.6 - 1e-4) * s1[i,d] + (0.7 - 1e-4) * s2[i,d] + (0.8 - 1e-4) * s3[i,d] + (0.9 - 1e-4) * s4[i,d] + (1 - 1e-4) * s5[i,d] +1*s6[i,d], name=f"ppoi_max_slab_{i}_day{d}")
        
        #z should be 1 only on one of the days
        for i in poi_ids:
            for j in poi_ids:
                if i != j:
                    model.addConstr(
                        quicksum(z[i, j, d] for d in days) <= 1,
                        name=f"EdgeUsedOnce_{i}_{j}"
                    )
            
        for i in poi_ids:
            if i != poi_ids[-2]:  # Skip the hotel
                model.addConstr(quicksum(y[i, d] for d in days) <= 1, name=f"VisitOnce_{i}")

          
        for i in poi_ids:
            model.addConstr(
                day_visit[i] == quicksum(d * y[i, d] for d in days),
                name=f"DayAssignment_{i}"
            )
        # Ensure only one mode is chosen for travel between i and j
        model.addConstrs((z[i, j, d] == w[i, j, d] + x[i, j, d] for i in poi_ids for j in poi_ids if i != j for d in days), name="MultimodalChoice")
        model.addConstrs((z[i, j, d] <= 1 for i in poi_ids for j in poi_ids if i != j for d in days), name="SingleModeLimit")

        ## Logical connection between y and ppoi
        
        epsilon = 1e-4

        for d in days:
            for i in poi_ids:
                # (1) If y[i] == 1 → ppoi[i] ≥ 0.5
                model.addConstr(ppoi[i,d] >= 0.5 * y[i,d], name=f"ppoi_min_if_y1_{i}")

                # (2) If ppoi[i] ≥ 0.5 → y[i] = 1
                model.addConstr(ppoi[i,d] <= 0.5 - epsilon + y[i,d], name=f"y1_if_ppoi_high_{i}")

                # (3) If y[i] == 0 → ppoi[i] == 0
                model.addConstr(ppoi[i,d] <= y[i,d], name=f"ppoi_zero_if_y0_{i}")

                # (4) If ppoi[i] < 0.5 → y[i] = 0
                model.addConstr(ppoi[i,d] >= (0.5 - epsilon) - (1 - y[i,d]), name=f"y0_if_ppoi_low_{i}")

        
        # ## LOGICAL CONNECTION BETWEEN Y AND Z

        # Additional constraint: Ensure logical connection between y[i] and z[i, j]
        for d in days:
            for i in poi_ids:
                for j in poi_ids:
                    if i != j:
                        model.addConstr(z[i, j, d] <= y[i,d], name=f"TravelStartsFrom_{i}_{j}")
                        model.addConstr(z[i, j, d] <= y[j,d], name=f"TravelEndsAt_{i}_{j}")

        ## MUST SEE POIS CONSTRAINT 
        
        for poi in must_see_pois:
            model.addConstr(quicksum(y[poi,d] for d in days) == 1, name=f"MustSee_{poi}")

        ## EXCLUDED POIS CONSTRAINT 
        
        for poi in excluded_pois:
            model.addConstr(quicksum(y[poi,d] for d in days) == 0, name=f"excluded_{poi}")
        

        # ## TIME CONSTRAINT
        for d in days:
            if d == days[0]:
                day_budget = time_budget_first_day
                day_start_time = start_time_first_day
            elif d == days[-1]:
                day_budget = time_budget_last_day
                day_start_time = start_time_last_day
            else:
                day_budget = time_budget
                day_start_time = start_time

            model.addConstr(
                quicksum(travel_times_walking.get((i, j), 0) * w[i, j, d] for i in poi_ids for j in poi_ids if i != j) +
                quicksum(travel_times_taxi.get((i, j), 0) * x[i, j, d] for i in poi_ids for j in poi_ids if i != j) +
                quicksum(visit_times[i] * ppoi[i,d] for i in poi_ids) <= day_budget,
                name="TimeConstraint"
            )
        
            M=10000
            
            for i in poi_ids:
                model.addConstr(
                    arrival_time[i, d] <= day_start_time + day_budget +(1-y[i,d])*M, 
                    name=f"ArrivalWithinTimeBudget_{i}_day{d}"
                )

        for theme, (lower_bound, upper_bound) in theme_bounds.items():
            theme_count = quicksum(
                y[i, d] for i in poi_ids for d in days
                if i - 1 < len(utility) and utility.iloc[i - 1]["theme"] == theme
            )
            model.addConstr(theme_count >= lower_bound, name=f"Min_{theme}")
            model.addConstr(theme_count <= upper_bound, name=f"Max_{theme}")

        for (a, b) in ordering_constraints:
            model.addConstr(
                day_visit[a] <= day_visit[b],
                name=f"Ordering_{a}_before_{b}_daywise"
            )
        
        M=100000
        for (a, b) in ordering_constraints:
            for d in days:
                model.addConstr(
                    arrival_time[a, d] + visit_times[a]*ppoi[a,d] + (
                        travel_times_walking.get((a, b), 0) * w[a, b, d] + 
                        travel_times_taxi.get((a, b), 0) * x[a, b, d]
                    ) <= arrival_time[b, d] + M * (1 - y[a, d]) + M * (1 - y[b, d]),
                    name=f"SameDayArrivalOrdering_{a}_before_{b}_day{d}"
                )


        # ## STARTING AND ENDING CONSTRAINT (ALWAYS INCLUDE THEM IN ITINERARY)
        starting_poi = poi_ids[0]
        hotel_poi = poi_ids[-2]
        ending_poi = poi_ids[-1]
        
        # for d in days:
        #     if d != days[-1]:  # For all days except last
        #         # Start from starting_poi
        #         model.addConstr(quicksum(z[starting_poi, j, d] for j in poi_ids if j != starting_poi) == 1, name=f"StartConstraint_day{d}")
        #         # End at starting_poi
        #         model.addConstr(quicksum(z[i, starting_poi, d] for i in poi_ids if i != starting_poi) == 1, name=f"EndConstraint_day{d}")

        #     else:  # For last day
        #         # Start from starting_poi
        #         model.addConstr(quicksum(z[starting_poi, j, d] for j in poi_ids if j != starting_poi) == 1, name=f"StartConstraint_day{d}")
        #         # End at poi_ids[-1]
        #         model.addConstr(quicksum(z[i, ending_poi, d] for i in poi_ids if i != ending_poi) == 1, name=f"EndConstraint_day{d}")
        #         # Ensure no outgoing edge from end node
        #         model.addConstr(quicksum(z[ending_poi, j, d] for j in poi_ids if j != ending_poi) == 0, name=f"NoOutgoingFromEnd_day{d}")
        
        for d in days:
            if d == days[0]:
                model.addConstr(quicksum(z[starting_poi, j, d] for j in poi_ids if j != starting_poi) == 1, name=f"Start_day{d}")
                model.addConstr(quicksum(z[i, hotel_poi, d] for i in poi_ids if i != hotel_poi) == 1, name=f"End_day{d}")
                model.addConstr(quicksum(z[hotel_poi, j, d] for j in poi_ids if j != hotel_poi) == 0, name=f"NoOutFromHotel_day{d}")
            elif d == days[-1]:
                model.addConstr(quicksum(z[hotel_poi, j, d] for j in poi_ids if j != hotel_poi) == 1, name=f"Start_day{d}")
                model.addConstr(quicksum(z[i, ending_poi, d] for i in poi_ids if i != ending_poi) == 1, name=f"End_day{d}")
                model.addConstr(quicksum(z[ending_poi, j, d] for j in poi_ids if j != ending_poi) == 0, name=f"NoOutFromDest_day{d}")
            else:
                model.addConstr(quicksum(z[hotel_poi, j, d] for j in poi_ids if j != hotel_poi) == 1, name=f"Start_day{d}")
                model.addConstr(quicksum(z[i, hotel_poi, d] for i in poi_ids if i != hotel_poi) == 1, name=f"End_day{d}")

         
        # # ## CONNECTIVITY CONSTRAINT
        # for d in days:
        #     for k in poi_ids:
        #         if (d != days[-1] and k != starting_poi) or (d == days[-1] and k not in [starting_poi, ending_poi]):
        #             model.addConstr(quicksum(z[i, k, d] for i in poi_ids if i != k) == y[k, d], name=f"FlowIn_{k}_day{d}")
        #             model.addConstr(quicksum(z[k, j, d] for j in poi_ids if j != k) == y[k, d], name=f"FlowOut_{k}_day{d}")
        
        for d in days:
            for k in poi_ids:
                # Skip flow constraint for source (only starting)
                # Skip flow constraint for destination (only ending)
                # Hotel is handled separately
                if k in [starting_poi, ending_poi]:
                    continue
                if d == days[0] and k == hotel_poi:
                    continue
                if d == days[-1] and k == hotel_poi:
                    continue
                if d != days[0] and d != days[-1] and k == hotel_poi:
                    continue  # hotel is not visited as POI; it's the start/end node

                # Apply flow constraint for all true POIs
                model.addConstr(quicksum(z[i, k, d] for i in poi_ids if i != k) == y[k, d], name=f"FlowIn_{k}_day{d}")
                model.addConstr(quicksum(z[k, j, d] for j in poi_ids if j != k) == y[k, d], name=f"FlowOut_{k}_day{d}")


        # ## COST BUDGET CONSTRAINT

        # Extract valid (i, j) pairs from cost_data
        valid_edges = set(zip(cost_data["from"], cost_data["to"]))

        # Create a dictionary mapping POI to its entrance fee from utility
        fees_dict = utility.set_index("poiID")["fees"].to_dict()

        # Add constraint: Total taxi cost (across all days) + entrance fee cost (across all days) ≤ cost_budget
        model.addConstr(
            quicksum(
                cost_data.loc[(cost_data["from"] == i) & (cost_data["to"] == j), "cost"].values[0] * taxi_cost_per_meter * x[i, j, d]
                for i, j in valid_edges for d in days
            ) +
            quicksum(
                fees_dict.get(i, 0) * y[i, d]
                for i in poi_ids for d in days
            ) <= cost_budget,
            name="TotalTripCostBudget"
        )

        ## OPENING CLOSING TIME CONSTRAINT

        opening_times = {i: convert_to_minutes(t) for i, t in opening_times.items()}
        closing_times = {i: convert_to_minutes(t) for i, t in closing_times.items()}

        for d in days:
            # model.addConstr(start_time == 600, name="StartAfter10AM")
            if d == days[0]:
                day_start_time = start_time_first_day
            elif d == days[-1]:
                day_start_time = start_time_last_day
            else:
                day_start_time = start_time
                
            start_node = starting_poi if d == days[0] else hotel_poi
            model.addConstr(arrival_time[start_node,d] == day_start_time, name="StartTimeAtSource")

        # for d in days:
        #     for i in poi_ids:
        #         model.addConstr(arrival_time[i, d] >= 0, name=f"ArrivalLB_{i}_day{d}")
        
        # for d in days:
        #     for i in poi_ids:
        #         for j in poi_ids:
        #             if i != j:
        #                 model.addConstr(z[i, j, d] + z[j, i, d] <= 1, name=f"No2Cycle_{i}_{j}_day{d}")
          
        for i in poi_ids:
            for d in days:
                model.addConstr(z[i, i, d] == 0, name=f"NoSelfLoop_{i}_day{d}")

        # for d in days:
        #     if d == days[0]:
        #         start_node = starting_poi
        #     else:
        #         start_node = hotel_poi
                
        #     if d == days[0]:
        #         day_budget = time_budget_first_day
        #         day_start_time = start_time_first_day
        #     elif d == days[-1]:
        #         day_budget = time_budget_last_day
        #         day_start_time = start_time_last_day
        #     else:
        #         day_budget = time_budget
        #         day_start_time = start_time
                
        #     for i in poi_ids:
        #         if i != start_node:
                    
        #             travel_time = (
        #                 travel_times_walking.get((start_node, i), 0) * w[start_node, i, d] +
        #                 travel_times_taxi.get((start_node, i), 0) * x[start_node, i, d]
        #             )
        #             model.addConstr(
        #                 arrival_time[i,d] >= day_start_time + travel_time - day_budget * (1 - z[start_node, i, d]),
        #                 name=f"StartPropagation_to_{i}_day{d}"
        #             )
        
        # --- time-ordering with correct Big-M per arc ---
        for d in days:
            if d == days[0]:
                start_node = starting_poi
            else:
                start_node = hotel_poi

            if d == days[0]:
                day_budget = time_budget_first_day
            elif d == days[-1]:
                day_budget = time_budget_last_day
            else:
                day_budget = time_budget
                
            for i in poi_ids:
                for j in poi_ids:
                    if i != j and i != start_node:
                            
                        # travel time if that mode is chosen
                        walk_tt = travel_times_walking.get((j, i), 0) * w[j, i, d]
                        taxi_tt = travel_times_taxi.get((j, i), 0) * x[j, i, d]
                        tt = walk_tt + taxi_tt

                        # BIG-M large enough to cover any arrival_j + visit_j + tt
                        Mji = day_budget + visit_times[j]*ppoi[j,d] + (travel_times_walking.get((j,i),0) + travel_times_taxi.get((j,i),0))

                        model.addConstr(
                            arrival_time[i, d]
                            >= arrival_time[j, d] + visit_times[j]*ppoi[j,d] + tt
                                - Mji * (1 - z[j, i, d]),
                            name=f"TimeOrdering_{j}_to_{i}_day{d}"
                        )

        for d in days:
            for i in poi_ids:
                model.addConstr(arrival_time[i, d] >= opening_times[i], name=f"OpeningTime_{i}")
                model.addConstr(arrival_time[i, d] <= closing_times[i] - visit_times[i]*ppoi[i,d], name=f"ClosingTime_{i}")


        # ## OPENING CLOSING DAY CONSTRAINT

        #Create a dictionary for day availability
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Compute the weekday for each trip day based on selected_day
        start_index = days_of_week.index(selected_day)
        trip_weekdays = [days_of_week[(start_index + d) % 7] for d in days]  # days = list(range(num_days))
        
        day_availability = {day: {} for day in days_of_week}

        # Populate the dictionary
        for i, row in utility.iterrows():
            poi_id = row['poiID']
            for day in days_of_week:
                day_availability[day][poi_id] = row[day]

        #Add constraints to only allow visiting POIs on open days
        for d in days:
            weekday = trip_weekdays[d]  # Actual weekday (e.g., Wednesday)
            for i in poi_ids:
                if day_availability[weekday][i] == 0:  # POI is closed on that weekday
                    model.addConstr(y[i, d] == 0, name=f"Closed_POI_{i}_on_day{d}_{weekday}")
                    
        # ## MODEL OPTIMIZATION STARTS

        begin_time = time.time()  # Record start time
        #model.setParam("FeasibilityTol", 1e-5)
        model.optimize()
        end_time = time.time()  # Record end time
        optimization_time = end_time - begin_time
        st.write(f"Optimization completed in {optimization_time:.4f} seconds.")

        # ## RESULTS
        
        if model.status == GRB.INFEASIBLE:
            st.error("Model is infeasible. Identifying conflicting constraints...")
            
            model.computeIIS()
            
            for constr in model.getConstrs():
                if constr.IISConstr:
                    print(f"Infeasible Constraint: {constr.ConstrName}")
                    st.write(f"❌ Infeasible Constraint: {constr.ConstrName}")
            
            for var in model.getVars():
                if var.IISLB:
                    st.write(f"❌ Infeasible Lower Bound: {var.VarName}")
                if var.IISUB:
                    st.write(f"❌ Infeasible Upper Bound: {var.VarName}")


        if model.status == GRB.OPTIMAL and model.objVal > 0:
            selected_pois = [i for i in poi_ids if y[i,d].X > 0.5]
            selected_edges = [(i, j) for i in poi_ids for j in poi_ids if i != j and z[i, j, d].X > 0.5]

            # Extract latitude and longitude for selected POIs
            poi_locations = dict(zip(utility["poiID"], zip(utility["lat"], utility["long"])))
            
            
            destination_poi = selected_pois[-1]  # Last POI as destination
            
            fig, ax = plt.subplots(figsize=(8, 6))

            st.title("Multi-Day Itinerary Graph")

            fig, ax = plt.subplots(figsize=(10, 8))
            # ax.set_title("All Days - Combined Itinerary", fontsize=15, fontweight='bold')

            # Define color palette for days
            day_colors = ['deeppink', 'green', 'purple', 'orange', 'gray', 'red', 'brown', 'yellow']
            all_selected_pois = set()
            edges_drawn = set()

            # === Plot arrows for each day's trip ===
            for d in days:
                selected_pois = [i for i in poi_ids if y[i, d].X > 0.5]
                selected_edges = [(i, j) for i in poi_ids for j in poi_ids if i != j and z[i, j, d].X > 0.5]

                all_selected_pois.update(selected_pois)
                day_color = day_colors[d % len(day_colors)]

                for i, j in selected_edges:
                    if (i, j, d) in edges_drawn:
                        continue
                    edges_drawn.add((i, j, d))

                    lat1, lon1 = poi_locations[i]
                    lat2, lon2 = poi_locations[j]

                    # Mode and travel time
                    mode = "Walking" if w[i, j, d].X > 0.5 else "Taxi"
                    travel_time = travel_times_walking.get((i, j), 0) if mode == "Walking" else travel_times_taxi.get((i, j), 0)

                    # Direction vector
                    vector = np.array([lon2 - lon1, lat2 - lat1])
                    norm = np.linalg.norm(vector)
                    if norm != 0:
                        vector /= norm
                    offset = 0.002
                    new_lon1 = lon1 + offset * vector[0]
                    new_lat1 = lat1 + offset * vector[1]
                    new_lon2 = lon2 - offset * vector[0]
                    new_lat2 = lat2 - offset * vector[1]

                    arrow = FancyArrowPatch(
                        posA=(new_lon1, new_lat1),
                        posB=(new_lon2, new_lat2),
                        connectionstyle="arc3",
                        arrowstyle='->',
                        color=day_color,
                        linewidth=2,
                        linestyle='--' if mode == "Taxi" else '-',
                        mutation_scale=15
                    )
                    ax.add_patch(arrow)

                    mid_lat = (lat1 + lat2) / 2
                    mid_lon = (lon1 + lon2) / 2
                    ax.text(mid_lon, mid_lat, f"{travel_time:.1f} min", fontsize=9, ha='center', va='center')

            # === Plot POIs ===
            for poi in all_selected_pois:
                lat, lon = poi_locations[poi]
                if poi == starting_poi:
                    color = 'green'
                    label = 'Source'
                elif poi == hotel_poi:
                    color = '#8b4513'  # brown for hotel
                    label = 'Hotel'
                elif poi == destination_poi:
                    color = 'red'
                    label = 'Destination'
                else:
                    color = '#5cadbd'
                    label = 'POI'

                existing_labels = ax.get_legend_handles_labels()[1]
                ax.scatter(lon, lat, c=color, s=450, edgecolors='white', linewidth=2, zorder=2,
                        label=label if label not in existing_labels else "")
                ax.text(lon, lat, str(poi), fontsize=12, ha='center', va='center', color='white', fontweight='bold')

            walking_patch = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='Walking Edge')
            taxi_patch = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Taxi Edge')

            # === Legends ===
            poi_legend = [
                mlines.Line2D([], [], color='green', marker='o', markersize=10, linestyle='None', label='Source'),
                mlines.Line2D([], [], color='red', marker='o', markersize=10, linestyle='None', label='Destination'),
                mlines.Line2D([], [], color='#8b4513', marker='o', markersize=10, linestyle='None', label='Hotel'),
                mlines.Line2D([], [], color='#5cadbd', marker='o', markersize=10, linestyle='None', label='POI'),
                walking_patch,
                taxi_patch
            ]


            # Add day-based color patches to legend
            for d in days:
                day_color = day_colors[d % len(day_colors)]
                poi_legend.append(mlines.Line2D([], [], color=day_color, lw=3, label=f"Day {d + 1} Route"))

            ax.legend(handles=poi_legend, loc='upper left', fontsize=9, bbox_to_anchor=(1.05, 1))

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True)

            st.pyplot(fig)                

            #printing user followable itinerary

            itinerary_html = "<div style='background-color:#f0f0f0; padding:20px; border-radius:10px; width:120%; margin:auto; margin-left: -10%;'>"
            itinerary_html += "<h3 style='text-align:center; color:#333;'>User-Followable Itinerary with Accurate Arrival Times</h3><br>"

            # Initialize totals
            total_taxi_cost_of_all_days = 0
            total_visit_cost_of_all_days = 0
            total_trip_cost_all_days = 0
            total_travel_time_all_days = 0
            total_visit_time_all_days = 0
            total_time_all_days = 0

            for d in days:
                itinerary_html += f"<h2 style='color:#154360;'>Day {d + 1}</h2>"

                selected_pois = [i for i in poi_ids if y[i, d].x > 0.5]
                day_utility = sum(utility_scores[i]*ppoi[i,d].x for i in selected_pois)
                edges = [(i, j) for i in poi_ids for j in poi_ids if z[i, j, d].X > 0.5]

                total_taxi_cost_on_this_day = sum(
                    cost_data.loc[(cost_data["from"] == i) & (cost_data["to"] == j), "cost"].values[0] * taxi_cost_per_meter * x[i, j, d].X
                    for i, j in valid_edges if x[i, j, d].X > 0.5
                )

                total_entrance_fee_cost_on_this_day = sum(fees_dict.get(i, 0) for i in selected_pois)
                total_trip_cost_of_this_day = total_taxi_cost_on_this_day + total_entrance_fee_cost_on_this_day

                total_taxi_cost_of_all_days += total_taxi_cost_on_this_day
                total_visit_cost_of_all_days += total_entrance_fee_cost_on_this_day
                total_trip_cost_all_days += total_trip_cost_of_this_day

                total_travel_time_on_this_day = sum(
                    travel_times_walking.get((i, j), 0) * w[i, j, d].x +
                    travel_times_taxi.get((i, j), 0) * x[i, j, d].x
                    for i in poi_ids for j in poi_ids if i != j
                )

                total_visit_time_on_this_day = sum(visit_times[i] * ppoi[i, d].X for i in poi_ids)
                total_time_taken_on_this_day = total_travel_time_on_this_day + total_visit_time_on_this_day

                total_travel_time_all_days += total_travel_time_on_this_day
                total_visit_time_all_days += total_visit_time_on_this_day
                total_time_all_days += total_time_taken_on_this_day

                itinerary_html += f"""
                <div style='border: 10px solid gray; border-radius: 15px; padding: 15px; background-color: #f4faff; margin-top: 10px; margin-bottom: 20px; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);'>
                    <p style='font-size:20px; color:#154360; font-weight:bold;'>Day Summary</p>
                    <p style='font-size:20px; color:#145A32;'><b>Total Utility:</b> {day_utility:.2f}</p>
                    <p style='font-size:18px; color:#154360;'><b>Selected POIs:</b> {selected_pois}</p>
                    <p style='font-size:18px; color:#154360;'><b>Selected Edges:</b> {edges}</p>
                    <p style='font-size:18px; color:#154360;'><b>Entrance Fee Cost:</b> ₹{total_entrance_fee_cost_on_this_day:.2f}</p>
                    <p style='font-size:18px; color:#154360;'><b>Taxi Cost:</b> ₹{total_taxi_cost_on_this_day:.2f}</p>
                    <p style='font-size:18px; color:#154360;'><b>Total Cost:</b> ₹{total_trip_cost_of_this_day:.2f}</p>
                    <p style='font-size:18px; color:#154360;'><b>Visit Time:</b> {total_visit_time_on_this_day:.2f} min</p>
                    <p style='font-size:18px; color:#154360;'><b>Travel Time:</b> {total_travel_time_on_this_day:.2f} min</p>
                </div>
                """

                # Start node
                if d == days[0]:
                    current_poi = starting_poi
                    day_start_time = start_time_first_day
                    end_poi = hotel_poi
                elif d == days[-1]:
                    current_poi = hotel_poi
                    day_start_time = start_time_last_day
                    end_poi = destination_poi
                else:
                    current_poi = hotel_poi
                    day_start_time = start_time
                    end_poi = hotel_poi

                current_time = day_start_time
                max_steps = len(poi_ids)
                steps = 0
                visited_today = False

                while current_poi is not None and steps < max_steps:
                    if steps!=0 and current_poi == end_poi:
                        break
                    
                    steps += 1
                    visit_time = visit_times[current_poi]*ppoi[current_poi,d].x
                    next_poi = None
                    travel_time = 0

                    for j in poi_ids:
                        if z[current_poi, j, d].x > 0.5:
                            next_poi = j
                            travel_time = (
                                travel_times_walking.get((current_poi, j), 0) * w[current_poi, j, d].x +
                                travel_times_taxi.get((current_poi, j), 0) * x[current_poi, j, d].x
                            )
                            mode = "Walking" if w[current_poi, j, d].x > 0.5 else "Taxi"
                            break

                    if next_poi is not None:
                        visited_today = True
                        itinerary_html += (
                            f"<p style='font-size:18px; font-weight:bold; color:#000;'>"
                            f"POI({current_poi}), "
                            f"<span style='color:#922b21;'>Arrival Time: {current_time:.2f} min</span>, "
                            f"Visiting Time: {visit_time:.2f} min, "
                            f"<span style='color:#922b21;'>Travel Time to POI({next_poi}): {travel_time:.2f} min</span> via <b>{mode}</b></p>"
                        )
                        current_time += visit_time + travel_time
                    else:
                        break

                    current_poi = next_poi

                # Add final POI
                if current_poi == end_poi:
                    itinerary_html += (
                        f"<p style='font-size:18px; font-weight:bold; color:#000;'>"
                        f"POI({end_poi}), <span style='color:#922b21;'>Arrival Time: {current_time:.2f} min</span>, "
                        f"Visiting Time: {visit_times[end_poi]} min</p>"
                    )

                if not visited_today:
                    itinerary_html += "<p style='font-size:16px; color:gray;'>No POIs visited on this day.</p>"

            # Close main div
            itinerary_html += "</div>"
            st.markdown(itinerary_html, unsafe_allow_html=True)
        else:
            st.error("❌ No optimal solution found.")

        # Apply Custom Styling
        st.markdown("""
        <style>
        .result-box {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            line-height: 1.6;
            width: 120%;  /* Keeping width 120% as per your request */
            max-width: 1200px;  /* Ensures it doesn’t go too wide on big screens */
            margin-left: -10%;  /* Shifts it left to keep it centered */
            text-align: left;  /* Keeps text alignment clean */
            display: block;
        }
        .result-box p {
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        if model.status == GRB.OPTIMAL and model.objVal>0:
            optimized_utility_score = model.objVal

            # Print results
            print(f"Optimized Utility Score: {optimized_utility_score}")
            print(f"Selected POIs: {selected_pois}")
            print()
            print(f"Total Taxi Cost: {total_taxi_cost_of_all_days:.2f} rupees")
            print(f"Total Entrance Fee Cost: {total_visit_cost_of_all_days:.2f} rupees")
            print(f"Total Trip Cost: {total_trip_cost_all_days:.2f} rupees")

        else:
            print("No optimal solution found.")

        print()

        if model.status == GRB.OPTIMAL and model.objVal>0:
            
            st.markdown(f"""
            <div class="result-box">
                <p>📊 <b>Optimized Utility Score:</b> {optimized_utility_score:.2f}</p>
                <hr>
                <p>🚕 <b>Total Taxi Cost:</b> ₹{total_taxi_cost_of_all_days:.2f}</p>
                <p>🎟️ <b>Total Entrance Fee Cost:</b> ₹{total_visit_cost_of_all_days:.2f}</p>
                <p>🛎️ <b>Total Trip Cost:</b> ₹{total_trip_cost_all_days:.2f}</p>
                <hr>
                <p>🚶 <b>Total Travel Time:</b> {total_travel_time_all_days:.2f} minutes</p>
                <p>🕒 <b>Total Visit Time:</b> {total_visit_time_all_days:.2f} minutes</p>
                <p>⏱️ <b>Total Time Taken:</b> {total_time_all_days:.2f} minutes</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('#')
# %%
