# Script for SageMaker Endpoint Deployment

import json
import os
import traceback

import numpy as np
import torch

import os
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
# sys.path.append('../../')
sys.path.append('../')
sys.path.append('../model_zoo/SimpleX/src/')
from matchbox import datasets
from datetime import datetime
from matchbox.utils import load_config, set_logger, print_to_json, print_to_list
from matchbox.features import FeatureMap, FeatureEncoder
# import model_zoo.SimpleX.src as models
from SimpleX import SimpleX
from matchbox.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import os
from pathlib import Path
import torch


def load_data_and_model(model_file, data_path):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    experiment_id = 'SimpleX_yelp18_m1'
    params = load_config('../model_zoo/SimpleX/config/SimpleX_yelp18_m1', experiment_id)
    params['gpu'] = -1

    seed_everything(seed=params['seed'])
    dataset = params['dataset_id'].split('_')[0].lower()
    params['data_root'] = '../data/Yelp18/'
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_encoder = FeatureEncoder(**params)
    feature_encoder.feature_map.load(feature_encoder.json_file)
    feature_map = feature_encoder.feature_map
    params["train_data"] = os.path.join(data_dir, 'train*.h5')
    params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
    if "test_data" in params:
        params["test_data"] = os.path.join(data_dir, 'test*.h5')
    params["item_corpus"] = os.path.join(data_dir, 'item_corpus.h5')

    model = SimpleX(feature_map, **params)
    # model.load_weights(model_file)
    return model


def model_fn(model_dir):
    print("## model_fn ##")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        print("device", device)

        model_file_path = os.path.join(model_dir, "model.pt")
        print(f"model_file_path: :  {model_file_path}")

        # Initialize the pytorch model
        model = load_data_and_model(
            model_file=model_file_path, data_path=model_dir
        )
        # model.dataset = dataset
        print("--> model network is loaded")

        return model

    except Exception:
        print("########## Failure loading a Model #######")
        print(traceback.format_exc())
        return "model error:" + traceback.format_exc()


def input_fn(input_data, content_type):
    # item_id sequence와 candidates를 동시에 인풋으로 받아야?
    # 일단 item_id seq만 생각해보자
    print("Deserializing the input data.")
    # dataset = global_variables["dataset"]
    # max_seq_length = global_variables["max_seq_length"]

    if content_type == "application/json":
        input_data = json.loads(input_data)

        sessions = input_data["sessions"]
        candidates = input_data["candidates"]

        # pad_len = max_seq_length - len(sessions)
        # transformed_sessions = dataset.token2id(dataset.iid_field, sessions)
        # transformed_sessions = [0] * pad_len + transformed_sessions.tolist()
        #
        # transformed_sessions = transformed_sessions[-max_seq_length:]
        # # truncate for long session sequences
        # transformed_candidates = dataset.token2id(dataset.iid_field, candidates)

        data = {}
        data["sessions"] = sessions
        data["sessions_len"] = len(sessions)
        data["candidates"] = candidates
        data["k"] = min(input_data["k"], len(candidates))

        return data

    raise Exception(
        "Requested unsupported ContentType in content_type: " + content_type
    )


def predict_fn(data, model):
    """Predict Function."""
    print("#### predict_fn starting ######")

    try:
        sessions = data["sessions"]
        candidates = data["candidates"]
        pad_len = model.max_seq_length - len(sessions)
        dataset = model.dataset
        transformed_sessions = dataset.token2id(dataset.iid_field, sessions)
        transformed_sessions = [0] * pad_len + transformed_sessions.tolist()

        transformed_sessions = transformed_sessions[-model.max_seq_length :]
        # truncate for long session sequences
        transformed_candidates = dataset.token2id(dataset.iid_field, candidates)

        k = data["k"]
        sessions_len = min(model.max_seq_length, len(sessions))

        # to make 2-dim tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sessions = torch.from_numpy(np.asarray([transformed_sessions])).to(device)
        sessions_len = torch.from_numpy(np.asarray([sessions_len])).to(device)
        candidates = torch.from_numpy(np.asarray([transformed_candidates])).to(device)

        print("predict_fn sessions", sessions)

        with torch.no_grad():
            # Forward
            user_dict = {
                'user_history': sessions,
                'user_id': torch.tensor([0,1])
            }
            item_dict = {
                'item_id': candidates
            }

            inputs = [user_dict, item_dict]
            predictions = model(inputs, training=False)['y_pred']

            # seq_out = model(sessions, sessions_len)  # [B, H] batch, hidden

            predictions = torch.gather(predictions, dim=1, index=candidates)
            scores, indices = torch.topk(input=predictions, k=k, dim=1)
            recommended_item_ids = torch.gather(
                input=candidates, dim=1, index=indices
            ).squeeze(0)
            scores = scores.squeeze(0).detach().cpu().numpy()

        prediction = recommended_item_ids.detach().cpu().numpy()
        prediction_token = dataset.id2token(dataset.iid_field, prediction)
        prediction_items = []
        for i, pi in enumerate(prediction):
            if pi == 0:
                continue
            prediction_items.append((prediction_token[i], scores[i]))

        return prediction_items

    except Exception:
        print(traceback.format_exc())
        return [("predict errorL:" + traceback.format_exc(), 0.0)]


def output_fn(prediction, content_type):
    if content_type == "application/json":
        ret = []
        # prediction = list(filter((0).__ne__, prediction))
        # prediction = dataset.id2token(dataset.iid_field, prediction)
        for x in prediction:
            item_payload = {}
            item_payload["id"] = x[0]
            item_payload["score"] = round(float(x[1]), 6)
            ret.append(item_payload)

        recommends = json.dumps({"predictions": ret})
        return recommends

    raise Exception(
        "Requested unsupported ContentType in content_type: " + content_type
    )


if __name__ == "__main__":
    context = {}
    content_type = "application/json"
    model = model_fn("../")
    raw_input_data = {
        "sessions": ["242", "302"],
        "candidates": ["242", "302", "210", "224"],
        "k": 3,
    }
    raw_input_str = json.dumps(raw_input_data)
    in_data = input_fn(raw_input_str, content_type)
    predictions = predict_fn(in_data, model)
    output = output_fn(predictions, content_type)
    print(output)
