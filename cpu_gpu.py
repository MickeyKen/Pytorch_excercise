import torch

if torch.cuda. is_available():
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    data1 = torch.zeros((2,2) , device = gpu)
    data_cpu = data1.to(cpu)
    data_gpu = data_cpu.to(gpu)
    print(data_cpu)
    print(data_gpu)

    # gpu = torch.device("cuda")
    data2 = torch.ones((3,2))
    print(data2)
    data2_gpu = data2.to(gpu)
    print(data2_gpu)
else:
    print("GPUが利用できる環境が整っていません")
