import os
import logging
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import net
import utils
from data_loader import GetDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--model_dir", default="model")

# Helper
def Predicts(net, test_set):
    """ Generator of the predictions of test_set """
    for x_in in test_set:
        x_in = x_in.unsqueeze(0)    # Add one more dim to send it into net
        output = net(x_in.float())
        predict = torch.max(output, dim=1)[1]
        yield predict

# Main functions
def train(data_loader, model, optimizer, loss_fn, metrics, params):
    # Enter training mode
    model.train()

    summary = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for i, (x_batch, y_batch) in enumerate(data_loader):
            # Forward
            outputs = model(x_batch)
            # Cross entropy loss doesn't support one-hot vector
            loss = loss_fn(outputs, y_batch)
            loss_avg.update(loss.item())
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()
            # Calculate metrics
            if i % params.save_summary_steps == 0:
                summary_batch = {metric: metrics[metric](outputs, y_batch)
                                 for metric in metrics
                                 }
                summary_batch["loss"] = loss.item()
                summary.append(summary_batch)
            # Update loss
            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()
    metric_mean = {
        metric: np.mean([x[metric] for x in summary])
        for metric in summary[0]
    }
    metric_string = " ; ".join("{}: {:05.3f}".format(k, v)
                               for k, v in metric_mean.items())
    logging.info("-Train metrics:" + metric_string)

def evaluate(model, data_loader, loss_fn, metrics):
    model.eval()
    summary = []
    # 注意这里并没有enumerate
    for x_batch, y_batch in data_loader:
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)

        summary_batch = {
            metric: metrics[metric](outputs, y_batch)
            for metric in metrics
        }
        summary_batch["loss"] = loss.item()
        summary.append(summary_batch)
    metrics_mean = {
        metric: np.mean([batch[metric] for batch in summary])
        for metric in summary[0]
    }
    metrics_string = ";".join("{}:{:05.3f}".format(k, v)
                              for k, v in metrics_mean.items())
    logging.info("-Eval metrics:" + metrics_string)
    return metrics_mean

def train_and_evaluate(model, train_dl, val_dl, optimizer, loss, metrics, params, model_dir):
    best_val_accuracy = 0
    for i in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(i, params.num_epochs))
        train(train_dl, model, optimizer, loss, metrics, params)
        val_metrics = evaluate(model, val_dl, loss, metrics)

        # 保存神经网络权重
        # 这里展示了两种保存的方法
        is_better = val_metrics["accuracy"] > best_val_accuracy
        utils.SaveCheckPoint({"Epoch": i + 1,
                              "state_dict": model.state_dict(),
                              "optim_dict": optimizer.state_dict()},
                             checkpoint=model_dir,
                             is_best=is_better)

        # 保存metrics(旧的和新的)
        if is_better:
            # 更新accuracy
            logging.info("Find new best accuracy!")
            best_val_accuracy = val_metrics["accuracy"]

            better_metrics = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.SaveDictToJson(val_metrics, better_metrics)
        last_metrics = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.SaveDictToJson(val_metrics, last_metrics)

if __name__ == "__main__":
    # 1. Get parser
    args = parser.parse_args()
    params_path = os.path.join(args.model_dir, "params.json")
    assert os.path.exists(params_path), "File not exist!"
    params = utils.Params(params_path)
    # 2. Get logger
    utils.SetLogger(args.model_dir)
    torch.manual_seed(0)
    # 3. Get data loader(train and cv)
    logging.info("Load dataset!")
    data_dir = args.data_dir
    data_loader = GetDataLoader(data_dir, ["train", "val"], params)
    train_dl = data_loader["train"]
    val_dl = data_loader["val"]
    logging.info("Done!")
    # 4. Define net, optimizers etc
    model = net.Net()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=params.learning_rate, alpha=0.9, weight_decay=0.0)
    # 这里的weight decay原来就是增加正则项啊
    loss = nn.CrossEntropyLoss()
    metrics = net.metrics
    # 5. Start train and evaluation
    logging.info("Start training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss, metrics, params, args.model_dir)
    # # 6. Get result
    # with open("cnn_digits_result.csv", "w") as f:
    #     f.write("ImageId,Label\n")
    #     for index, result in enumerate(Predicts(net, data_set["test"])):
    #         f.write(str(index + 1) + "," + str(int(result)))
    #         f.write("\n")
