import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# Constants
WINDOW_SIZE = 40  # Window size for moving average
LOWER_BOUND = -20  # Lower limit on vertical axis
UPPER_BOUND = 20   # Upper limit on vertical axis
DISPLAY_DURATION = 40  # Duration to stay on screen

# Initialize real-time data processing variables
window = deque(maxlen=WINDOW_SIZE)
mean = 0
std_dev = 1

def update_mean_std(new_value):
    """
    Update mean and standard deviation using Welford's algorithm.
    
    Args:
        new_value (float): The new data point to include in calculations
        
    The algorithm provides numerically stable updates for mean and standard deviation.
    """
    global mean, std_dev, window
    n = len(window)
    if n == 0:
        mean = new_value
        std_dev = 0
    else:
        old_mean = mean
        mean += (new_value - old_mean) / (n + 1)
        std_dev += (new_value - old_mean) * (new_value - mean)
        std_dev = np.sqrt(std_dev / n) if n > 1 else 1

def generate_data_point(t):
    """
    Generate a data point with seasonality, noise, and occasional deviations.
    
    Args:
        t (int): Time step for generating seasonal component
        
    Returns:
        tuple: (generated_value, is_anomaly_flag)
    """
    try:
        seasonal_component = 10 * np.sin(2 * np.pi * t / 50)
        noise = random.gauss(0, 1)
        
        # Add high deviation with 5% probability
        if random.random() < 0.05:
            outlier = random.choice([random.gauss(25, 5), random.gauss(-25, 5)])
            return seasonal_component + noise + outlier, True
        
        return seasonal_component + noise, False
    except Exception as e:
        print(f"Error in data generation: {e}")
        return 0, False

def detect_anomaly(new_value, previous_value, is_generated_anomaly):
    """
    Enhanced anomaly detection function.
    Only detects generated anomalies or values truly outside the normal range.
    
    Args:
        new_value (float): Current data point to check
        previous_value (float): Previous data point
        is_generated_anomaly (bool): Flag indicating if the point was generated as an anomaly
        
    Returns:
        bool: True if the point is an anomaly, False otherwise
    """
    try:
        if is_generated_anomaly:
            return True
        
        # Check absolute bounds
        if new_value < LOWER_BOUND or new_value > UPPER_BOUND:
            return True
        
        return False
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return False

def clean_old_data(data_x, data_y, current_time):
    """
    Clean up data points that are no longer visible on screen.
    
    Args:
        data_x (deque): X-coordinates of data points
        data_y (deque): Y-coordinates of data points
        current_time (int): Current timestamp
        
    Returns:
        tuple: (cleaned_data_x, cleaned_data_y)
    """
    try:
        if not data_x:
            return data_x, data_y
        
        while data_x and data_x[0] < current_time - WINDOW_SIZE:
            data_x.popleft()
            data_y.popleft()
        
        return data_x, data_y
    except Exception as e:
        print(f"Error in data cleaning: {e}")
        return deque(), deque()

def data_stream():
    """
    Real-time data stream simulation function.
    
    Yields:
        tuple: (new_value, is_anomaly)
    """
    t = 0
    while True:
        new_value, is_anomaly = generate_data_point(t)
        yield new_value, is_anomaly
        t += 1

def initialize_plot():
    """
    Initialize the real-time visualization settings.
    
    Returns:
        tuple: (figure, axis, data_line, anomalies_line)
    """
    plt.ion()
    fig, ax = plt.subplots()
    data, = ax.plot([], [], 'b-', label="Data Stream")
    anomalies, = ax.plot([], [], 'ro', label="Anomalies")
    
    ax.set_ylim(-50, 50)
    ax.set_xlim(0, WINDOW_SIZE)
    plt.legend()
    
    return fig, ax, data, anomalies

def run_anomaly_detection():
    """
    Main function for anomaly detection and visualization.
    Handles real-time data processing, anomaly detection, and visualization updates.
    """
    try:
        # Initialize visualization
        fig, ax, data, anomalies = initialize_plot()
        
        # Initialize data structures
        data_x = deque(maxlen=DISPLAY_DURATION)
        data_y = deque(maxlen=DISPLAY_DURATION)
        anomaly_x = deque(maxlen=DISPLAY_DURATION)
        anomaly_y = deque(maxlen=DISPLAY_DURATION)
        anomaly_labels = deque(maxlen=DISPLAY_DURATION)
        previous_value = 0
        annotations = []
        current_time = 0

        for new_value, is_generated_anomaly in data_stream():
            current_time += 1
            
            # Update statistics
            window.append(new_value)
            update_mean_std(new_value)

            # Check for anomalies
            if detect_anomaly(new_value, previous_value, is_generated_anomaly):
                anomaly_x.append(current_time)
                anomaly_y.append(new_value)
                anomaly_labels.append(f"{new_value:.2f}")
            
            # Update data points
            data_x.append(current_time)
            data_y.append(new_value)

            # Clean old data
            data_x, data_y = clean_old_data(data_x, data_y, current_time)
            
            # Clean old anomalies
            while anomaly_x and anomaly_x[0] < current_time - WINDOW_SIZE:
                anomaly_x.popleft()
                anomaly_y.popleft()
                anomaly_labels.popleft()

            # Update visualization
            data.set_data(list(data_x), list(data_y))
            anomalies.set_data(list(anomaly_x), list(anomaly_y))
            ax.set_xlim(current_time - WINDOW_SIZE, current_time)

            # Update annotations
            for annotation in annotations:
                annotation.remove()
            annotations.clear()

            for i, txt in enumerate(anomaly_labels):
                annotation = ax.annotate(txt, (anomaly_x[i], anomaly_y[i]), 
                                       textcoords="offset points", 
                                       xytext=(0, 10), 
                                       ha='center')
                annotations.append(annotation)

            previous_value = new_value

            plt.pause(0.01)
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    run_anomaly_detection()