import re

def check_nan_and_negative_icc(report_file_path, output_dir):
    nan_icc_entries = []
    negative_icc_entries = []
    
    with open(report_file_path, 'r') as file:
        report_lines = file.readlines()

    # Process the report line by line
    for i, line in enumerate(report_lines):
        # Check if the line contains ICC value and it's either nan or negative
        if "ICC: nan" in line or "ICC: -" in line:
            # Collect the 3 lines as required (current line + two previous lines)
            paragraph = []
            
            # Add the previous two lines if available
            if i > 1:
                paragraph.append(report_lines[i-2].strip())  # Second previous line
            if i > 0:
                paragraph.append(report_lines[i-1].strip())  # First previous line
            paragraph.append(line.strip())  # Current line

            # Add the paragraph to the respective list based on the ICC value
            if "ICC: nan" in line:
                nan_icc_entries.append('\n'.join(paragraph))
            else:
                negative_icc_entries.append('\n'.join(paragraph))

    return nan_icc_entries, negative_icc_entries

def save_results_to_file(nan_icc_entries, negative_icc_entries, output_dir, output_filename='nan_and_negative_icc.txt'):
    output_path = f"{output_dir}/{output_filename}"
    with open(output_path, 'w') as output_file:
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
    
    print(f"Results saved to {output_path}")

# Example usage
report_file_path = '/mnt/nas7/data/maria/final_features/icc_results_dose/icc_computation_log.txt'
output_dir = '/mnt/nas7/data/maria/final_features/icc_results_dose'  # Change this as needed

nan_icc_entries, negative_icc_entries = check_nan_and_negative_icc(report_file_path, output_dir)
save_results_to_file(nan_icc_entries, negative_icc_entries, output_dir)




