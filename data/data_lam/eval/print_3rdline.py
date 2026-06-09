def print_third_line(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for index, line in enumerate(file, start=1):
                if index == 3:
                    # rstrip('\n') removes the trailing newline character
                    print(line.rstrip('\n'))
                    return
            print("The file has fewer than 3 lines.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage with your currently active document:
file_path = "/home/lam/Downloads/hcmue_benchmark_review.md"
print_third_line(file_path)
