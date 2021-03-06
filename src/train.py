import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from tensorboardX import SummaryWriter
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import resume
from .data_handler.data_processor import FlightDataset, InfoPrefix
from .model.embedding_network import FTN
from .model.knn import WeightedKNNPredictor
from .model.prediction_network import PredNetwork
from .model.focal_loss import FocalLoss

import argparse
import os
import shutil

def get_args():
    parser = argparse.ArgumentParser("train of flight delay forecasting model")
    parser.add_argument("--model_version", type=str, help="model version for the train")
    parser.add_argument("--metric_learning", type=bool, default=False, help="model version for the train")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of images per batch")
    parser.add_argument("--num_worker", type=int, default=4, help="The number of worker for dataloader")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.00004)
    parser.add_argument('--smoothing_factor', type=float, default=0.05)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--random_projection_dim', type=int, default=256)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--glip_threshold', type=float, default=0.1)
    parser.add_argument('--num_nearest_neighbors', type=int, default=30)
    parser.add_argument('--gaussian_kernel_k', type=float, default=0.2)
    parser.add_argument('--SSOt', type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--test_interval", type=int, default=2, help="Number of epoches between testing phases")
    parser.add_argument("--eval_interval", type=int, default=20, help="Number of epoches between eval phases")
    parser.add_argument("--tensorboard_path", type=str, default="tensorboard")
    parser.add_argument("--checkpoint_root_dir", type=str, default="checkpoint_dir", help="checkpoint directory path")
    parser.add_argument("--resume", type=str, default=None, help="resume checkpoint path")

    args = parser.parse_args()
    return args

def train(args):
    metric_learning = args.metric_learning
    df = pd.read_csv(os.path.join(os.path.dirname(__file__).replace(os.path.basename(os.path.dirname(__file__)), ''),'data/FlightSchedule.csv'))
    train_data_df, test_data_df = train_test_split(df, test_size=0.25)
    info_prefix = InfoPrefix(df)
    route_dict = info_prefix.extract_route()
    aircraft_type_dict = info_prefix.extract_aircraft_type()
    aircraft_name_dict = info_prefix.extract_aircraft_name()
    max_min_seating_capacity_dict = info_prefix.extract_max_min_seating_capacity()
    max_min_baggage_weight_dict = info_prefix.extract_max_min_baggage_weight()

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(123)
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
        device = 'cpu'
        torch.manual_seed(123)

    train_dataset = FlightDataset(
        train_data_df,
        route_dict,
        aircraft_type_dict,
        aircraft_name_dict,
        max_min_seating_capacity_dict,
        max_min_baggage_weight_dict
    )
    test_dataset = FlightDataset(
        test_data_df,
        route_dict,
        aircraft_type_dict,
        aircraft_name_dict,
        max_min_seating_capacity_dict,
        max_min_baggage_weight_dict
    )
    train_params = {
        "batch_size": args.batch_size * num_gpus,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_worker,
    }

    test_params = {
        "batch_size": args.batch_size * num_gpus,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_worker
    }

    train_generator = DataLoader(train_dataset, **train_params)
    test_generator = DataLoader(test_dataset, **test_params)

    if metric_learning:
        model = FTN(input_dim=train_dataset.input_dim(), embedding_dim=args.embedding_dim, dropout_p=args.dropout_p)
        random_projection_layer = torch.normal(mean=0, std=1.0, size=[train_dataset.input_dim(), args.random_projection_dim])
        knn_predictor = WeightedKNNPredictor(args.num_nearest_neighbors, args.gaussian_kernel_k, args.SSOt, 2, args.sample_size)
    else:
        model = PredNetwork(input_dim=train_dataset.input_dim(), embedding_dim=args.embedding_dim, dropout_p=args.dropout_p)
        focal_loss = FocalLoss(args.smoothing_factor)

    if torch.cuda.is_available():
        model.cuda()
        if metric_learning:
            random_projection_layer = random_projection_layer.cuda()

    if args.resume is not None:
        _ = resume(model, device, args.resume)

    model = torch.nn.DataParallel(model)

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
        os.makedirs(args.tensorboard_path)
    else:
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.checkpoint_root_dir):
        os.makedirs(args.checkpoint_root_dir)

    writer = SummaryWriter(args.tensorboard_path)

    params = list(model.parameters())
    base_optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = base_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(train_generator.__len__() * 50),
        T_mult=1,
        eta_min=0.00001
    )

    model.train()
    num_iter_per_epoch = len(train_generator)
    train_iter = 0
    for epoch in range(args.num_epochs):
        epoch_loss = []
        progress_bar = tqdm(train_generator)
        for iter, data in enumerate(progress_bar):
            scheduler.step(epoch + iter / num_iter_per_epoch)
            optimizer.zero_grad()
            data_feature_1 = data['rep']
            if metric_learning:
                data_feature_2 = data['rep'][torch.randperm(data['rep'].shape[0])]

            if torch.cuda.is_available():
                if metric_learning:
                    reconstructed_inputs_1, dense_representation_1 = model(data_feature_1.cuda().float())
                    reconstructed_inputs_2, dense_representation_2 = model(data_feature_2.cuda().float())
                    random_projected_output_1 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_1.cuda().float(), dim=1, p=2),random_projection_layer))
                    random_projected_output_2 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_2.cuda().float(), dim=1, p=2),random_projection_layer))
                    rp_inner_product = (random_projected_output_1 * random_projected_output_2).sum(dim=1) / (random_projected_output_2.shape[1] ** (1 / 2))
                    dr_inner_product = (dense_representation_1 * dense_representation_2).sum(dim=1) / (dense_representation_2.shape[1] ** (1 / 2))
                    reconstructed_loss_1 = F.mse_loss(reconstructed_inputs_1, data_feature_1.cuda().float())
                    reconstructed_loss_2 = F.mse_loss(reconstructed_inputs_2, data_feature_2.cuda().float())
                else:
                    output = model(data_feature_1.cuda().float())
                    label = data['label'].cuda()
            else:
                if metric_learning:
                    reconstructed_inputs_1, dense_representation_1 = model(data_feature_1.float())
                    reconstructed_inputs_2, dense_representation_2 = model(data_feature_2.float())
                    random_projected_output_1 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_1.float(), dim=1, p=2),random_projection_layer))
                    random_projected_output_2 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_2.float(), dim=1, p=2),random_projection_layer))
                    rp_inner_product = (random_projected_output_1 * random_projected_output_2).sum(dim=1) / (random_projected_output_2.shape[1] ** (1 / 2))
                    dr_inner_product = (dense_representation_1 * dense_representation_2).sum(dim=1) / (dense_representation_2.shape[1] ** (1 / 2))
                    reconstructed_loss_1 = F.mse_loss(reconstructed_inputs_1, data_feature_1.float())
                    reconstructed_loss_2 = F.mse_loss(reconstructed_inputs_2, data_feature_2.float())
                else:
                    output = model(data_feature_1.float())
                    label = data['label'].cuda()

            if metric_learning:
                reconstruction_loss = reconstructed_loss_1 + reconstructed_loss_2
                random_projection_loss = F.mse_loss(dr_inner_product, rp_inner_product)
                loss = random_projection_loss + reconstruction_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.glip_threshold)
                optimizer.step()
                epoch_loss.append(float(loss))
            else:
                loss = focal_loss(output,label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.glip_threshold)
                optimizer.step()
                epoch_loss.append(float(loss))

            total_loss = np.mean(epoch_loss)
            train_iter += 1

            if metric_learning:
                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, args.num_epochs, iter + 1, num_iter_per_epoch, loss, total_loss
                    )
                )
            else:
                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, args.num_epochs, iter + 1, num_iter_per_epoch, loss, total_loss
                    )
                )

            writer.add_scalars('total_loss', {'train': total_loss}, train_iter)
            scheduler.step()

        if (epoch + 1) % args.test_interval == 0 and epoch + 1 >= 0:
            epoch_loss = []
            model.eval()

            if metric_learning:
                data_bank = []
                rdp_loss_list = []
                reconstruction_loss_list = []

            num_correct_pred = 0
            num_pred = 0
            with torch.no_grad():
                if metric_learning:
                    for iter, data in enumerate(tqdm(train_generator)):
                        if torch.cuda.is_available():
                            reconstructed_inputs, dense_representation = model(data['rep'].cuda().float())
                            label = data['label'].cuda()
                        else:
                            reconstructed_inputs, dense_representation = model(data['rep'].float())
                            label = data['label'].cuda()

                        for dense_feature_iter in range(dense_representation.shape[0]):
                            data_bank_representation = torch.cat([dense_representation[dense_feature_iter, :],label[dense_feature_iter].float().unsqueeze(0)], dim=0)
                            data_bank.append(data_bank_representation)

                    data_bank = torch.stack(data_bank, dim=0)
                    test_data_embedding_list = []

                for iter, data in enumerate(tqdm(test_generator)):
                    data_feature_1 = data['rep']
                    if metric_learning:
                        data_feature_2 = data['rep'][torch.randperm(data['rep'].shape[0])]
                    if torch.cuda.is_available():
                        if metric_learning:
                            reconstructed_inputs_1, dense_representation_1 = model(data_feature_1.cuda().float())
                            reconstructed_inputs_2, dense_representation_2 = model(data_feature_2.cuda().float())
                            test_data_embedding_list.append(dense_representation_1)
                            prob = knn_predictor(
                                dense_representation_1.float(),
                                data_bank
                            )
                            random_projected_output_1 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_1.cuda().float(), dim=1, p=2),random_projection_layer))
                            random_projected_output_2 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_2.cuda().float(), dim=1, p=2),random_projection_layer))
                            rp_inner_product = (random_projected_output_1 * random_projected_output_2).sum(dim=1) / (random_projected_output_2.shape[1] ** (1 / 2))
                            dr_inner_product = (dense_representation_1 * dense_representation_2).sum(dim=1) / (dense_representation_2.shape[1] ** (1 / 2))
                            reconstructed_loss_1 = F.mse_loss(reconstructed_inputs_1, data_feature_1.cuda().float())
                            reconstructed_loss_2 = F.mse_loss(reconstructed_inputs_2, data_feature_2.cuda().float())
                        else:
                            logit = model(data_feature_1.cuda().float())
                            prob = torch.softmax(logit, dim=1)
                            label = data['label'].cuda()

                        pred = torch.argmax(prob, dim=1)
                        correct_pred = (pred == data['label'].cuda())


                    else:
                        if metric_learning:
                            reconstructed_inputs_1, dense_representation_1 = model(data_feature_1.float())
                            reconstructed_inputs_2, dense_representation_2 = model(data_feature_2.float())
                            test_data_embedding_list.append(dense_representation_1)
                            prob = knn_predictor(
                                dense_representation_1.float(),
                                data_bank,
                            )
                            random_projected_output_1 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_1.float(), dim=1, p=2),random_projection_layer))
                            random_projected_output_2 = torch.cos(torch.matmul(torch.nn.functional.normalize(data_feature_2.float(), dim=1, p=2),random_projection_layer))
                            rp_inner_product = (random_projected_output_1 * random_projected_output_2).sum(dim=1) / (random_projected_output_2.shape[1] ** (1 / 2))
                            dr_inner_product = (dense_representation_1 * dense_representation_2).sum(dim=1) / (dense_representation_2.shape[1] ** (1 / 2))
                            reconstructed_loss_1 = F.mse_loss(reconstructed_inputs_1, data_feature_1.float())
                            reconstructed_loss_2 = F.mse_loss(reconstructed_inputs_2, data_feature_2.float())
                        else:
                            logit = model(data_feature_1.float())
                            prob = torch.softmax(logit, dim=1)
                            label = data['label']

                        pred = torch.argmax(prob, dim=1)
                        correct_pred = (pred == data['label'])

                    if metric_learning:
                        reconstruction_loss = reconstructed_loss_1 + reconstructed_loss_2
                        random_projection_loss = F.mse_loss(dr_inner_product, rp_inner_product)
                        loss = reconstruction_loss + random_projection_loss
                        rdp_loss_list.append(random_projection_loss.item())
                        reconstruction_loss_list.append(reconstruction_loss.item())
                    else:
                        loss = focal_loss(logit, label)

                    epoch_loss.append(loss.item())
                    num_correct_pred += correct_pred.sum(0).item()
                    num_pred += data['rep'].shape[0]

            total_loss = np.mean(epoch_loss)
            accuracy = num_correct_pred / num_pred

            writer.add_scalars('total_loss', {'test': total_loss}, train_iter)
            writer.add_scalar('accuracy', accuracy, train_iter)

            if metric_learning:
                print('rdp loss: {}'.format(np.mean(rdp_loss_list)))
                print('reconstruction loss: {}'.format(np.mean(reconstruction_loss_list)))

            print('Epoch: {}/{}. Total loss: {:1.5f}'.format(epoch + 1, args.num_epochs, total_loss))
            print('Epoch: {}/{}. accuracy : {:1.5f}'.format(epoch + 1, args.num_epochs, accuracy))

            if (epoch + 1) % args.eval_interval == 0 and epoch + 1 >= 0:
                if torch.cuda.device_count() > 1:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.module.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            args.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}.pth'.format('model_v1', epoch + 1)
                        )
                    )
                else:
                    checkpoint_dict = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
                    torch.save(
                        checkpoint_dict,
                        os.path.join(
                            args.checkpoint_root_dir,
                            '{}_checpoint_epoch_{}.pth'.format('model_v1', epoch + 1)
                        )
                    )

            model.train()
            print('---------------------------------------------------------------------------------------------------')

    writer.close()

if __name__ == "__main__":
    args = get_args()

    args.tensorboard_path = os.path.join(args.tensorboard_path, args.model_version)
    args.checkpoint_root_dir = os.path.join(args.checkpoint_root_dir, args.model_version)
    train(args)