# ECG Signal Viewer

I made a Python tool that analyzes heart signals (ECG) and detects heartbeats automatically.

## How:
- Loads patient ECG data
- Finds each heartbeat  
- Calculates heart rate
- Compares to expert cardiologist annotations ('atr')
- Shows results

## How to:

1. Install Python libraries:
```
pip install numpy matplotlib scipy wfdb
```

2. Download ECG data files from PhysioNet:
   - Go to: https://physionet.org/content/mitdb/1.0.0/
   - Download `100.dat` and `100.hea`
   - Put them in the same folder as the Python file

3. Run the program:
```
python ecg_viewer_advanced.py
```

## You'll see:
A graph showing the heartbeat pattern with detected beats marked in red and expert annotations in green!

<img src="example_output.png" alt="Example Output" width="800"/>

---

Created as a biomedical engineering learning project.
```


5. **Save and close** the file


