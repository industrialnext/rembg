import hashlib
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Type

import gdown
import onnxruntime as ort

from .session_base import BaseSession
from .session_cloth import ClothSession
from .session_simple import SimpleSession

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config


def new_session(model_name: str) -> BaseSession:
    session_class: Type[BaseSession]

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = "https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR"
        session_class = SimpleSession
    elif model_name == "u2net":
        md5 = "60024c5c889badc19c04ad937298a77b"
        url = "https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab"
        session_class = SimpleSession
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j"
        session_class = SimpleSession
    elif model_name == "u2net_cloth_seg":
        md5 = "2434d1f3cb744e0e49386c906e5a08bb"
        url = "https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz"
        session_class = ClothSession
    elif model_name == "u2net_fender":
        md5 = "cbc3fd37740be9d4d86b531165b1a66f"
        s3_bucket = "industrialnext-public"
        url = "vision/model/fender_2022-12-10_320x320v2_u2net_bce_itr_112000_train_0.529639_tar_0.067323.pth.onnx"
        session_class = SimpleSession
    else:
        assert AssertionError(
            "Choose between u2net, u2netp, u2net_human_seg, u2net_cloth_seg, u2net_fender"
        )

    home = os.getenv(
        "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    )
    path = Path(home).expanduser() / f"{model_name}.onnx"
    path.parents[0].mkdir(parents=True, exist_ok=True)

    download = False
    if not path.exists():
        download = True
    else:
        hashing = hashlib.new("md5", path.read_bytes(), usedforsecurity=False)
        if hashing.hexdigest() != md5:
            download = True

    if download:
        with redirect_stdout(sys.stderr):
            if "drive.google.com" in url:
                gdown.download(url, str(path), use_cookies=False)
            else:
                s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

                try:
                    s3.Bucket(s3_bucket).download_file(url, str(path))
                except botocore.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        print("The object does not exist.")
                    else:
                        raise

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(
        model_name,
        ort.InferenceSession(
            str(path), providers=ort.get_available_providers(), sess_options=sess_opts
        ),
    )
