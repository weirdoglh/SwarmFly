import cupy as cp
import pandas as pd
from scipy.spatial import cKDTree

# Assuming your data is already converted to CuPy arrays
cupy_data = [...]

def get_first_person_data_vectorized(cupy_object_data, observation_range):
    first_person_data = []

    for timestamp_index, timestamp in enumerate(cupy_data):
        # Create a KDTree from object positions
        pos_tree = cKDTree(cp.asnumpy([obj['position'] for obj in timestamp]))

        # Initialize a DataFrame to store relative data
        columns = ['id', 'relative_position', 'relative_momentum']
        df = pd.DataFrame(columns=columns)

        for obj_index, obj in enumerate(timestamp):
            # Query for objects within the observation range
            distances, indices = pos_tree.query_ball_point(obj['position'].get(), r=observation_range)

            # Calculate relative positions and momenta for each observed object using CuPy
            rel_pos = cp.subtract(timestamp[indices]['position'], obj['position'][cp.newaxis, :])
            rel_mom = cp.subtract(timestamp[indices]['momentum'], obj['momentum'][cp.newaxis, :])

            # Convert to NumPy arrays and stack them horizontally
            rel_pos = rel_pos.get().tolist()
            rel_mom = rel_mom.get().tolist()
            rel_data = [[idx, rp, rm] for idx, rp, rm in zip(indices, rel_pos, rel_mom)]

            # Append relative data to the DataFrame
            df = df.append(pd.DataFrame(rel_data, columns=columns))

            # Create the first-person data for the current object
            fp_data = {
                'id': obj['id'],
                'timestamp': timestamp_index,
                'observed_objects': df[df['id'] != obj['id']].to_dict(orient='records')
            }

            # Append the first-person data for this object to the final list
            first_person_data.append(fp_data)

    return first_person_data

# Define the observation range (e.g., in meters)
observation_range = 5.0

# Generate first-person perspective data using vectorized computation
first_person_perspective_vectorized = get_first_person_data_vectorized(cupy_data, observation_range)