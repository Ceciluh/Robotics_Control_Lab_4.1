import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/PD/best_PD5_actual_deg.csv"
OUT_PATH = "figs/PD5_joint_positions.png"

def main():
    df = pd.read_csv(CSV_PATH)

    t = df["time_s"]
    cols = [c for c in df.columns if c.startswith("q_") and c.endswith("_deg")]

    plt.figure()
    for c in cols:
        plt.plot(t, df[c], label=c.replace("q_", "").replace("_deg", ""))

    plt.xlabel("time (s)")
    plt.ylabel("joint position (deg)")
    plt.title("PD5 - Joint positions vs time (MuJoCo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
