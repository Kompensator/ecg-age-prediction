from re import A
import torch
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append(r"C:\Users\dingyi.zhang\Documents\DeepLearningECG\dataset")
from CLSA_dataset import CLSA
from resnet import ResNet1d
from scipy.signal import resample

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    device = torch.device("cuda:0")
    model = ResNet1d(input_dim=(12, 4096),
                    blocks_dim=list(zip([64, 128, 196, 256, 320], [4096, 1024, 256, 64, 16])),
                    n_classes=1,
                    kernel_size=17,
                    dropout_rate=0.8,
                    mlp_output=0)
    model.load_state_dict(torch.load(r'C:\Users\dingyi.zhang\Documents\ecg-age-prediction\checkpoints\FI39_new_weights.pth')['model'])
    model = model.to(device)
    valid_dataset = CLSA(start=25000, end=27500)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
    mse = []
    pairs = []
    for i, batch in enumerate(valid_loader):
        model.eval()
        x = batch[0].to(device)
        y = batch[8].to(device)
        w = batch[7].to(device)
        ages = batch[6]
        sexes = batch[9]
        tabular = torch.stack((ages, sexes), 1)
        tabular = tabular.to(device)

        with torch.no_grad():
            output = model(x)
        
        output = np.squeeze(output.detach().cpu().numpy())
        y = np.squeeze(y.detach().cpu().numpy())
        for error in (output - y) ** 2:
            mse.append(error)
        for i, j in zip(y, output):
            pairs.append([i, j])
        
    pairs = sorted(pairs, key=lambda x: x[0])
    actual = np.array([i[0] for i in pairs])
    predicted = np.array([i[1] for i in pairs])
    predicted[predicted < 0] = 0
    predicted = moving_average(predicted , 15)
    predicted = resample(predicted, len(actual))
    
    plt.title("Predicted vs. Actual FI39 in CLSA Cohort")
    plt.plot(actual, color='red', label='Actual FI39')
    plt.plot(predicted, color='blue', label='Predicted FI39')
    plt.xlabel("Test Set Patients Sorted By Ascending Actual 39-Item Frailty Index (FI39)")
    plt.ylabel("39-Item Frailty Index (FI39)")
    plt.legend()
    plt.grid()
    plt.show()

    print("MSE validation: {}".format(sum(mse)/len(mse)))


if __name__ == '__main__':
    main()