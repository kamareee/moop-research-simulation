# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring, wrong-import-order

import numpy as np
import pickle


def generate_data(N=45, T=96, F=3):
    """
    N: 45 vehicles to ensure contention.
    T: 96 time slots (15-min intervals)[cite: 573, 1334, 1762].
    """
    dt = 0.25  # 15 minutes = 0.25 hours

    # 1. Australian Commercial TOU Tariff Implementation
    # Values converted from c/kWh to $/kWh
    off_peak = 0.1781
    shoulder = 0.2087
    peak = 0.3121

    pi_grid = np.zeros(T)
    # 00:00 - 07:00 (Slots 0 to 28): Off-peak
    pi_grid[0:28] = off_peak
    # 07:00 - 15:00 (Slots 28 to 60): Shoulder
    pi_grid[28:60] = shoulder
    # 15:00 - 21:00 (Slots 60 to 84): Peak
    pi_grid[60:84] = peak
    # 21:00 - 22:00 (Slots 84 to 88): Shoulder
    pi_grid[84:88] = shoulder
    # 22:00 - 24:00 (Slots 88 to 96): Off-peak
    pi_grid[88:96] = off_peak

    # 2. Site Parameters (Tuned for Contention)
    t = np.arange(T)
    p_pv = 30 * np.maximum(0, np.sin(np.pi * (t - 24) / 48))
    p_base = 25 + 10 * np.random.rand(T)  # Fixed higher base load

    # 3. Public Station Parameters
    stations = []
    for f in range(F):
        stations.append(
            {
                "pi_pub": {"L2": 0.25, "DC": 0.40},
                "power": {"L2": 7.0, "DC": 150.0},
                "wait_profile": 10 + 15 * np.sin(np.pi * t / 96) ** 2,
            }
        )

    # 4. Fleet Parameters (High Intensity / Low Initial SoC)
    fleet = []
    for i in range(N):
        # Operational windows
        dwell_windows = [[0, 30], [42, 70], [82, 96]]

        fleet.append(
            {
                "bat_max": 75.0,  # [cite: 1018, 1362, 1455]
                "soc_init": 0.20,  # 20% Initial SoC to force energy demand
                "e_trip": [35.0, 35.0],  # 70kWh total daily demand
                "dwells": dwell_windows,
                "detour": np.random.randint(10, 20, size=F),
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
        "p_site_max": 200.0,
        "evse_m": 15,
        "c_d_t": 0.30,
    }

    with open("sim_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(
        f"Data Generated with AU Commercial Tariff: P_site_max={data['p_site_max']}kW"
    )


if __name__ == "__main__":
    generate_data()
