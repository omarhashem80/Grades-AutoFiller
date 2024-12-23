import tracemalloc
import joblib

tracemalloc.start()
try:
    load_digit_model = joblib.load("Module1/HOG_Model_DIGITS.npy")
except MemoryError as e:
    print("MemoryError encountered:", e)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024**2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
finally:
    tracemalloc.stop()
