import numpy as np
import matplotlib.pyplot as plt
import sir_erlang_steps as model

if __name__ == "__main__":
    #import traceback
    try:
        model.main()
    except:
        print(f"GGA CRASHED {1e20}")
        #traceback.print.exc()

