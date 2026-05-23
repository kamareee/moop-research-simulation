# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, wrong-import-order

import numpy as np
import pickle


def generate_data(N=45, T=96, F=3):
    """
    N: Number of vehicles, T: Time slots (15 min), F: Public stations
    """
    dt = 0.25

    # 1. Depot Pricing - Aggressive Peak Pricing to discourage depot use
    pi_grid = np.zeros(T)
    pi_grid[0:28] = 0.12  # Cheap Off-peak
    pi_grid[28:60] = 0.22  # Shoulder
    pi_grid[60:88] = 0.45  # EXTREME Peak (3pm-10pm) to force public DC use
    pi_grid[88:96] = 0.18

    # 2. Lower PV and Higher Base Load
    t = np.arange(T)
    p_pv = 30 * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
    p_base = 20 + 10 * np.random.rand(T)  # Constant higher base load

    # 3. Public Station Parameters - Higher Wait Times to penalize public use
    stations = []
    for f in range(F):
        stations.append(
            {
                "pi_pub": {"L2": 0.25, "DC": 0.60},  # Premium DC pricing
                "power": {"L2": 7.0, "DC": 150.0},  # Faster DC
                "wait_profile": 15
                + 20 * np.sin(np.pi * t / 96) ** 2,  # High wait (15-35 min)
            }
        )

    # 4. Fleet Energy Intensity (alpha) and Dwell Compression (beta)
    # We lower initial SoC to force vehicles to NEED energy [cite: 1652-1654].
    fleet = []
    for i in range(N):
        trip_start = [30, 70]
        dwell_windows = [[0, 30], [42, 70], [82, 96]]

        fleet.append(
            {
                "bat_max": 75.0,
                "soc_init": 0.25,  # Lowered to 25% to force urgency
                "e_trip": [35.0, 35.0],  # Higher energy demand per trip
                "dwells": dwell_windows,
                "detour": np.random.randint(10, 20, size=F),  # Longer detours
            }
        )

    data = {
        "N": N,
        "T": T,
        "F": F,
        "dt": dt,
        "pi_grid": pi_grid,
        "p_pv": p_pv,
        "p_base": p_base,
        "stations": stations,
        "fleet": fleet,
        "p_site_max": 100.0,  # Lowered site limit [cite: 1550]
        "evse_m": 12,  # Limited chargers [cite: 1552]
        "c_d_t": 1.0,  # Higher driver cost to make the choice "harder"
    }

    with open("sim_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Data Generated: N={N}, SiteLimit={data['p_site_max']}kW")


if __name__ == "__main__":
    generate_data()
