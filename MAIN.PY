import sys
import os

# Define the build folder path
BUILD_DIR = ".build"

def run_script(script_path, output_dir):
    print(f"Running {script_path} ...")
    # Pass the output_dir as an argument to the script
    exit_code = os.system(f"{sys.executable} \"{script_path}\" \"{output_dir}\"")
    if exit_code != 0:
        print(f"Error running {script_path}")
        sys.exit(exit_code)

if __name__ == "__main__":
    # Create build folder if it doesn't exist
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
        print(f"Created build directory: {BUILD_DIR}")

    # Step 1: Data Preprocessing
    run_script(os.path.join("files", "DATA PREPROCESSING.py"), BUILD_DIR)
    
    # Step 2: Data Visualization
    run_script(os.path.join("files", "DATA VISUALIZATION.py"), BUILD_DIR)
    
    # Step 3: All Models (Forecasting)
    run_script(os.path.join("files", "ALL MODELS.py"), BUILD_DIR)

    print("✅ Project pipeline completed. All outputs are in the '.build' folder.")
