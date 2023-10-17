import torch

num_devices = torch.cuda.device_count()
print(f'Number of cuda devices availible for computing: {num_devices}')
for i in range(num_devices):
    device = torch.cuda.device(i)
    print(f'Device {i} is: {torch.cuda.get_device_name(device)}', f'Device {i} has the following properties:\n    {torch.cuda.get_device_properties(device)}', sep='\n')