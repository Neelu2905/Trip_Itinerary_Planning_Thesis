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



# Title
st.title("🚀 Trip Planner")

# Dropdown for City Selection
st.subheader(" Select Your City")
cities = ["Select a City","Budapest", "Delhi", "Osaka", "Glasgow", "Vienna", "Perth", "Edinburgh"]
city = st.selectbox("", cities)

st.markdown(f"<p class='big-font'>📍 City Selected: {city}</p>", unsafe_allow_html=True)

if city!="Select a City":
    # Paths to Excel files
    utility_data_path = "C:/Users/Neelu/Desktop/Thesis/data/Updated Travel Data.xlsx"
    cost_data_path = "C:/Users/Neelu/Desktop/Thesis/data/Cost Data.xlsx"

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
    
    st.subheader("Select Must-See POIs")
    
    poi_ids = utility['poiID'].tolist()
    must_see_pois = st.multiselect("If you have any preference then choose the must-see POIs:", poi_ids, placeholder="Choose multiple options")

    # Budget Inputs (Inline with Columns)
    st.subheader(" Travel Budget")
    col1, col2 = st.columns(2)
    with col1:
        time_budget = st.number_input("⏳ Time Budget (in hours)", min_value=0.0, step=0.5, format="%.1f")

    # Coordinates Input (Formatted Display)
    st.subheader(" Coordinates")
    col1, col2 = st.columns(2)

    # with col1:
    #     source_lat = st.number_input("📍 Source Latitude", format="%.6f")
    #     source_lon = st.number_input("📍 Source Longitude", format="%.6f")

    # with col2:
    #     dest_lat = st.number_input("📍 Destination Latitude", format="%.6f")
    #     dest_lon = st.number_input("📍 Destination Longitude", format="%.6f")
    
    
    source_lat=34.655
    source_lon=135.43
    dest_lat=34.645
    dest_lon=135.504
    st.write(source_lat)
    st.write(source_lon)
    st.write(dest_lat)
    st.write(dest_lon)

    # Display Selected Data (Styled)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📌 Selected Travel Details")

    st.markdown(
        f"""
        <div class='box'>
        <p><b>City:</b> {city}</p>
        <p><b>Time Budget:</b> {time_budget} Minutes</p>
        <p><b>Source Coordinates:</b> ({source_lat}, {source_lon})</p>
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

    # Extract relevant data
    poi_ids = utility['poiID'].tolist()
    poi_ids = [0] + poi_ids + [poi_ids[-1] + 1]

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


    # ## ADDING SOURCE POI TO DESTINATION POI AND VICE VERSA TO COST_DATA
    # 1. Calculate distances using your Haversine function
    distance_0_end_poi = haversine_distance_meters(source_lat, source_lon, dest_lat, dest_lon)
    # 2. Create new rows as a list of dictionaries
    rows_to_add = [
        {
            'from': 0,
            'to': poi_ids[-1],
            'cost': distance_0_end_poi,
            'profit': 0,
            'category': 'hotel'
        },
        {
            'from': poi_ids[-1],
            'to': 0,
            'cost': distance_0_end_poi,
            'profit': 0,
            'category': 'hotel'
        }
    ]
    # 3. Convert that list into a small DataFrame
    new_rows_df = pd.DataFrame(rows_to_add)
    # 4. Concatenate with your existing cost_data
    cost_data = pd.concat([cost_data, new_rows_df], ignore_index=True)
    utility.drop(columns=['distance from source','distance from destination'], inplace = True)
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
    travel_times_walking = {(row['from'], row['to']): (row['cost'] / 1000) * 15 for _, row in cost_data.iterrows()}

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
        y = model.addVars(poi_ids, vtype=GRB.BINARY, name="y")
        N = len(poi_ids)  # Total number of POIs
        # Create continuous variables for the position of each POI in the sequence
        p = model.addVars(poi_ids, vtype=GRB.CONTINUOUS, lb=2, ub=N, name="p")
        w = model.addVars(poi_ids, poi_ids, vtype=GRB.BINARY, name="w")  # Walking
        # Objective: Maximize the sum of utility scores for selected POIs
        model.setObjective(quicksum(utility_scores[i] * y[i] for i in poi_ids), GRB.MAXIMIZE)
        # # Ensure only one mode is chosen for travel between i and j
        # model.addConstrs((z[i, j] <= 1 for i in poi_ids for j in poi_ids if i != j), name="SingleModeLimit")

        # ## LOGICAL CONNECTION BETWEEN Y AND Z

        # Additional constraint: Ensure logical connection between y[i] and z[i, j]
        # for i in poi_ids:
        #     for j in poi_ids:
        #         if i != j:
        #             model.addConstr(w[i, j] <= y[i], name=f"TravelStartsFrom_{i}_{j}")
        #             model.addConstr(w[i, j] <= y[j], name=f"TravelEndsAt_{i}_{j}")
        
        ## MUST SEE POIS CONSTRAINT 
        for poi in must_see_pois:
            model.addConstr(y[poi] == 1, name=f"MustSee_{poi}")
        

        # ## TIME CONSTRAINT

        model.addConstr(
            quicksum(travel_times_walking.get((i, j), 0) * w[i, j] for i in poi_ids for j in poi_ids if i != j) +
            quicksum(visit_times[i] * y[i] for i in poi_ids) <= time_budget,
            name="TimeConstraint"
        )

        # ## STARTING AND ENDING CONSTRAINT (ALWAYS INCLUDE THEM IN ITINERARY)
        starting_poi = poi_ids[0]
        ending_poi = poi_ids[-1]
        
        model.addConstr(w[starting_poi,ending_poi]==0,name="NoDirectEdgefromStartToEnd")
        
        model.addConstr(y[starting_poi]==1,name="startPOIincluded")
        model.addConstr(y[ending_poi]==1,name="endPOIincluded")

        # Start point: Exactly one outgoing edge from node SOURCE
        model.addConstr(quicksum(w[starting_poi, j] for j in poi_ids if j != starting_poi) == 1, name="StartConstraint")

        # End point: Exactly one incoming edge to node DESTINATION
        model.addConstr(quicksum(w[i, ending_poi] for i in poi_ids if i != ending_poi) == 1, name="EndConstraint")

        # ## CONNECTIVITY CONSTRAINT

        for k in poi_ids:
            if k not in [starting_poi, ending_poi]:  # Skip start and end points
                model.addConstr(quicksum(w[i, k] for i in poi_ids if i != k) == y[k], name=f"FlowIn_{k}")
                model.addConstr(quicksum(w[k, j] for j in poi_ids if j != k) == y[k], name=f"FlowOut_{k}")


        # ## SUBTOUR ELIMINATION CONSTRAINT
        for i in poi_ids:
            for j in poi_ids:
                if i != j and i > starting_poi and j > starting_poi:  # Skip the start and end POIs
                    model.addConstr(
                        p[i] - p[j] + 1 <= (N - 1) * (1 - w[i, j]),
                        name=f"SubtourElimination_{i}_{j}"
                    )

        # ## NO OUTGOING EDGE FROM ENDING POI CONSTRAINT

        # No outgoing edges from POI 39 (end POI)
        model.addConstr(quicksum(w[ending_poi, j] for j in poi_ids if j != ending_poi) == 0, name="NoOutgoingFromEnd")

        # Extract valid (i, j) pairs from cost_data
        valid_edges = set(zip(cost_data["from"], cost_data["to"]))        

        def convert_to_minutes(time_obj):
            return time_obj.hour * 60 + time_obj.minute

        # ## MODEL OPTIMIZATION STARTS

        begin_time = time.time()  # Record start time
        model.optimize()  # Run Gurobi optimization
        end_time = time.time()  # Record end time
        optimization_time = end_time - begin_time
        st.write(f"Optimization completed in {optimization_time:.4f} seconds.")

        # ## RESULTS

        if model.status == GRB.OPTIMAL and model.objVal > 0:
            selected_pois = [i for i in poi_ids if y[i].X > 0.5]
            selected_edges = [(i, j) for i in poi_ids for j in poi_ids if i != j and w[i, j].X > 0.5]

            # Extract latitude and longitude for selected POIs
            poi_locations = dict(zip(utility["poiID"], zip(utility["lat"], utility["long"])))
            
            
            destination_poi = selected_pois[-1]  # Last POI as destination
            
            fig, ax = plt.subplots(figsize=(8, 6))

            st.title("Travel Itinerary Graph")

            # Plot edges first
            # Plot directed edges with arrows
            for edge in selected_edges:
                poi1, poi2 = edge
                lat1, lon1 = poi_locations[poi1]
                lat2, lon2 = poi_locations[poi2]

                # Determine edge color and travel time based on mode of travel
                if w[(poi1, poi2)].x == 1:
                    edge_color = 'gray'  # Walking
                    travel_time = travel_times_walking.get((poi1, poi2), 0)

                # Compute direction vector and normalize
                vector = np.array([lon2 - lon1, lat2 - lat1])
                norm = np.linalg.norm(vector)
                if norm != 0:
                    vector = vector / norm  # Normalize the vector

                # Offset both ends slightly to make arrow appear to touch the POIs
                arrow_offset = 0.00015  # Smaller offset for tighter connection

                new_lon1 = lon1 + arrow_offset * vector[0]
                new_lat1 = lat1 + arrow_offset * vector[1]
                new_lon2 = lon2 - arrow_offset * vector[0]
                new_lat2 = lat2 - arrow_offset * vector[1]

                ax.annotate("",
                xy=(new_lon2, new_lat2), xycoords='data',
                xytext=(new_lon1, new_lat1), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=edge_color, lw=2, mutation_scale=15))

                # Compute midpoint for label placement
                mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2

                # Add travel time as edge label
                ax.text(mid_lon, mid_lat, f"{travel_time:.1f} min", fontsize=10, ha='center', va='center', color='black', zorder=3)


            # Plot POIs with white edge to separate them from edges
            for poi in selected_pois:
                lat, lon = poi_locations[poi]
                if poi == starting_poi:
                    color, label = 'green', 'Source POI'
                elif poi == destination_poi:
                    color, label = 'red', 'Destination POI'
                else:
                    color, label = 'blue', 'POI'

                # Bigger POI circles with white edge
                ax.scatter(lon, lat, c=color, s=450, edgecolors='white', linewidth=2, zorder=2, label=label if label not in ax.get_legend_handles_labels()[1] else "")

                # Add labels inside the circles
                ax.text(lon, lat, str(poi), fontsize=12, ha='center', va='center', color='white', fontweight='bold', zorder=3)

            # Create legend handles for POIs (vertices)
            source_patch = mlines.Line2D([], [], color='green', marker='o', markersize=10, linestyle='None', label='Source POI')
            destination_patch = mlines.Line2D([], [], color='red', marker='o', markersize=10, linestyle='None', label='Destination POI')
            poi_patch = mlines.Line2D([], [], color='blue', marker='o', markersize=10, linestyle='None', label='POI')

            # Create legend handles for edges
            walking_patch = mpatches.Patch(color='gray', label='Walking Edge')

            # Combine legends and add with spacing
            ax.legend(handles=[source_patch, destination_patch, poi_patch, walking_patch], 
                    loc='upper left', fontsize=10, labelspacing=1.5,
                    bbox_to_anchor=(1.05, 1))  # Move legend box outside the Cartesian plane

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("POI Graph")
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True)

            # Display in Streamlit
            st.pyplot(fig)
        
            # 🔧 Build directed edge map
            edge_map = {i: j for i, j in selected_edges}

            # 🔁 Reconstruct path starting from 0
            path = [0]
            while path[-1] != 30:
                current = path[-1]
                next_poi = edge_map.get(current)
                if next_poi is None:
                    st.error(f"🚨 Path broken at POI {current}")
                    st.stop()
                path.append(next_poi)

            st.title("🗺️ Optimized Walking Itinerary")

            for i in range(len(path)):
                poi = path[i]
                st.subheader(f"📍 POI {poi}")

                if i < len(path) - 1:
                    next_poi = path[i + 1]
                    walk_time = travel_times_walking.get((poi, next_poi), 0)
                    st.write(f"➡️ Walk to POI {next_poi} — approx **{walk_time:.2f} min**")

            st.success("🎉 You’ve reached POI 30 — Trip Complete!")
        
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

        else:
            print("No optimal solution found.")

        print()

        if model.status == GRB.OPTIMAL and model.objVal>0:
            # Calculate the LHS of the time constraint after the solution
            total_travel_time = sum(
                                    travel_times_walking.get((i, j), 0) * w[i, j].x
                                    for i in poi_ids for j in poi_ids if i != j
                                )

            total_visit_time = sum(visit_times[i] * y[i].X for i in poi_ids)
            total_time_taken = total_travel_time + total_visit_time
            # st.write(y)
            st.markdown(f"""
            <div class="result-box">
                <p>📊 <b>Optimized Utility Score:</b> {optimized_utility_score:.2f}</p>
                <hr>
                <p>🚶 <b>Total Travel Time:</b> {total_travel_time:.2f} minutes</p>
                <p>🕒 <b>Total Visit Time:</b> {total_visit_time:.2f} minutes</p>
                <p>⏱️ <b>Total Time Taken:</b> {total_time_taken:.2f} minutes</p>
            </div>
            """, unsafe_allow_html=True)


            st.markdown('#')



