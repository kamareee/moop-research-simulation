# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import pickle
import numpy as np


def generate_data(num_vehicles=20, time_slots=96, num_stations=5):
    """
    N: Number of vehicles, T: Time slots (15 min), F: Public stations
    """
    dt = 0.25  # 15 minutes in hours

    # 1. Depot Pricing (VIC/AU style ToU) [cite: 1464]
    pi_grid = np.zeros(time_slots)
    pi_grid[0:28] = 0.15  # Off-peak (12am-7am)
    pi_grid[28:60] = 0.25  # Shoulder
    pi_grid[60:84] = 0.35  # Peak (3pm-9pm)
    pi_grid[84:96] = 0.20  # Shoulder

    # 2. PV Generation and Base Load [cite: 1462]
    t = np.arange(time_slots)
    p_pv = 50 * np.maximum(0, np.sin(np.pi * (t - 24) / 48))  # Peak at noon
    p_base = 10 + 5 * np.random.rand(time_slots)

    # 3. Public Station Parameters [cite: 1464, 1471]
    stations = []
    for f in range(num_stations):
        stations.append(
            {
                "pi_pub": {"L2": 0.20, "DC": 0.50},
                "power": {"L2": 11.0, "DC": 100.0},
                "wait_profile": 5
                + 10 * np.sin(np.pi * t / 96) ** 2,  # Busy midday [cite: 1471, 1669]
            }
        )

    # 4. Vehicle/Fleet Schedules [cite: 1455, 1736]
    fleet = []
    for i in range(num_vehicles):
        trip_start = [32, 68]  # 8am, 5pm
        trip_duration = 12  # 3 hours
        dwell_windows = [[0, 32], [44, 68], [80, 96]]  # Depot, Route, Depot

        fleet.append(
            {
                "bat_max": 75.0,
                "soc_init": 0.4,  # 40% [cite: 1455]
                "e_trip": [25.0, 30.0],  # Energy needed per trip [cite: 1455]
                "dwells": dwell_windows,
                "detour": np.random.randint(
                    5, 15, size=num_stations
                ),  # Detour min per station [cite: 1464]
            }
        )

    data = {
        "N": num_vehicles,
        "T": time_slots,
        "F": num_stations,
        "dt": dt,
        "pi_grid": pi_grid,
        "p_pv": p_pv,
        "p_base": p_base,
        "stations": stations,
        "fleet": fleet,
        "p_site_max": 150.0,
        "evse_m": 10,
        "c_d_t": 0.5,  # $0.5/min detour [cite: 1464]
    }

    with open("sim_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Synthetic data generated and saved to sim_data.pkl")


if __name__ == "__main__":
    generate_data()
