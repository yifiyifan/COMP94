import re 

def clean_string(text):
    # Define the regex pattern to match all characters except alphanumeric and regular punctuations
    pattern = r'[^a-zA-Z0-9.,!?;:\'\"()\[\]{}<>@#$%^&*+=\-_~`\s]'
    # Substitute all characters matching the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def format_execution_time(start_time, end_time):
    elapsed_time = end_time - start_time

    # Format the elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_text = f"{seconds:.2f} seconds"
    if minutes > 0:
        time_text = f"{int(minutes)} minute " + time_text
    if hours > 0:
        time_text = f"{int(hours)} hours " + time_text
    # Print the formatted elapsed time
    return time_text