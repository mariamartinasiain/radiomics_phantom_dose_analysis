import re
import os

def check_nan_and_negative(report_file_path):
    nan_icc_entries = []
    negative_icc_entries = []
    msb_less_than_mse_entries = []
    
    with open(report_file_path, 'r') as file:
        report_lines = file.readlines()

    current_block = []
    msb_pattern = re.compile(r"MSB:\s*([-+]?\d*\.\d+|\d+)")
    mse_pattern = re.compile(r"MSE:\s*([-+]?\d*\.\d+|\d+)")

    # Process the report line by line
    for i, line in enumerate(report_lines):
        if "------------------------------------------------------------" in line:
            if current_block:
                # Check conditions inside the block
                block_text = '\n'.join(current_block)
                
                # Check for ICC: nan
                if any("ICC: nan" in l for l in current_block):
                    nan_icc_entries.append(block_text)

                # Check for ICC < 0
                if any("ICC: -" in l for l in current_block):
                    negative_icc_entries.append(block_text)

                # Check for MSB < MSE
                msb_match = msb_pattern.search(block_text)
                mse_match = mse_pattern.search(block_text)
                if msb_match and mse_match:
                    msb = float(msb_match.group(1))
                    mse = float(mse_match.group(1))
                    if msb < mse:
                        msb_less_than_mse_entries.append(block_text)
            
            # Start a new block
            current_block = []
        else:
            current_block.append(line.strip())

    return nan_icc_entries, negative_icc_entries, msb_less_than_mse_entries

def save_results_to_file(nan_icc_entries, negative_icc_entries, msb_less_than_mse_entries, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save NaN and Negative ICC results
    nan_negative_output_path = os.path.join(output_dir, 'nan_and_negative_icc.txt')
    with open(nan_negative_output_path, 'w') as output_file:
        if nan_icc_entries:
            output_file.write("Entries with ICC: nan:\n")
            output_file.write("=" * 50 + "\n")
            for entry in nan_icc_entries:
                output_file.write(entry + "\n")
                output_file.write("=" * 50 + "\n")

        if negative_icc_entries:
            output_file.write("Entries with Negative ICC:\n")
            output_file.write("=" * 50 + "\n")
            for entry in negative_icc_entries:
                output_file.write(entry + "\n")
                output_file.write("=" * 50 + "\n")
    
    print(f"NaN and Negative ICC results saved to {nan_negative_output_path}")

    # Save MSB < MSE results
    msb_mse_output_path = os.path.join(output_dir, 'msb_less_than_mse.txt')
    with open(msb_mse_output_path, 'w') as output_file:
        if msb_less_than_mse_entries:
            output_file.write("Entries where MSB < MSE:\n")
            output_file.write("=" * 50 + "\n")
            for entry in msb_less_than_mse_entries:
                output_file.write(entry + "\n")
                output_file.write("=" * 50 + "\n")

    print(f"MSB < MSE results saved to {msb_mse_output_path}")

# File paths
report_file_path = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_computation_log.txt'
output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose'

# Process the file
nan_icc_entries, negative_icc_entries, msb_less_than_mse_entries = check_nan_and_negative(report_file_path)

# Save the results
save_results_to_file(nan_icc_entries, negative_icc_entries, msb_less_than_mse_entries, output_dir)



