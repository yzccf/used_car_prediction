from torch.nn.functional import relu
from tqdm import tqdm
import csv

out_csv = 'submission.csv'


def predict(dataloader, net, log_dir, price_mean, price_std, batch_size):
    with open(log_dir + out_csv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SaleID", "price"])
        i = 0
        for X in tqdm(dataloader):
            X = X[0]
            y = net(X).view(-1)
            prediction = relu(y * price_std + price_mean)
            prediction = prediction.detach().numpy().tolist()
            writer.writerows(list(zip([_ for _ in range(i, i + batch_size)], prediction)))
            i += batch_size
