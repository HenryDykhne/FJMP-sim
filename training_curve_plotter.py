import re
import os
import matplotlib.pyplot as plt

# Define an array of file names
file_names = ['logs/original_M1/log', 'logs/config_argo_3/log' ]
take_last = [36, 36]

# Define a list of colors to use for each file
colors = ['black', 'red', 'orange', 'green', 'blue', 'purple']

items = ['FDE', 'ADE']
regex = [r'  FDE: ([+-]?([0-9]*[.])?[0-9]+)', r'\tADE: ([+-]?([0-9]*[.])?[0-9]+)']
for e, item in enumerate(items):
    # Use regex to find all the FDE values in each file and store them in a dictionary
    values_dict = {}
    for i, file_name in enumerate(file_names):
        with open(file_name, 'r') as file:
            contents = file.read()
            matches = re.findall(regex[e], contents)
            print(len(matches))
            values = [float(match[0]) for match in matches[-take_last[i]:]]
            values_dict[file_name] = values


    # Plot the FDE values for each file
    for i, file_name in enumerate(file_names):
        plt.plot(values_dict[file_name], color=colors[i%len(colors)], label=file_name.replace('log', ''))

    # Set the x and y axis labels and the title of the plot
    plt.xlabel('Epoch')
    plt.ylabel('min'+item+'')
    plt.title(''+item+' Training Runs and Ablations')

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    #plt.show()

    # Save the plot to a file in the same directory as the script
    plt.savefig('img/'+item+'_training_runs.png')

    plt.close()