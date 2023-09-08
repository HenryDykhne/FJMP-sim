import re
import os
import matplotlib.pyplot as plt

# Define an array of file names
file_names = ['logs/fjmp_with_new_heads/log_initial_training_schedule','logs/fjmp_with_new_heads/log_new_training_schedule', 'logs/fjmp_with_new_heads/log']

# Define a list of colors to use for each file
colors = ['black', 'red', 'orange', 'green', 'blue', 'purple']

# Use regex to find all the FDE values in each file and store them in a dictionary
values_dict = {}
for i, file_name in enumerate(file_names):
    with open(file_name, 'r') as file:
        contents = file.read()
        matches = re.findall(r'  FDE: ([+-]?([0-9]*[.])?[0-9]+)', contents)
        print(len(matches))
        values = [float(match[0]) for match in matches]
        values_dict[file_name] = values

# Plot the FDE values for each file
for i, file_name in enumerate(file_names):
    plt.plot(values_dict[file_name], color=colors[i%len(colors)], label=file_name.replace('log', ''))

# Set the x and y axis labels and the title of the plot
plt.xlabel('Epoch')
plt.ylabel('minFDE')
plt.title('FDE Training Runs and Ablations')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()

# Save the plot to a file in the same directory as the script
plt.savefig('img/fde_training_runs.png')