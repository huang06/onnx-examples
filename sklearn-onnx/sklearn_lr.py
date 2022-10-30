from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as rt
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression


def linear_regression(df: pd.DataFrame) -> LinearRegression:
    x_train = df[["x1", "x2", "x3"]]
    y_train = df["y"]
    clr = LinearRegression().fit(x_train, y_train)
    return clr


def convert_to_onnx(model: Any, verbose: int = 0) -> Any:
    features = ["x1", "x2", "x3"]
    initial_types = [("input", FloatTensorType([None, len(features)]))]
    model_onnx = convert_sklearn(model, initial_types=initial_types, verbose=verbose)
    return model_onnx


def save_onnx_model(model: Any, path: str | Path) -> None:
    Path(path).write_bytes(model.SerializeToString())


if __name__ == "__main__":
    training_data = pd.DataFrame(np.random.rand(100, 4), columns=["x1", "x2", "x3", "y"])
    testing_data = pd.DataFrame(np.random.rand(5, 4), columns=["x1", "x2", "x3", "y"])
    x_test = testing_data[["x1", "x2", "x3"]]
    y_test = testing_data["y"]
    skl_model = linear_regression(training_data)
    skl_pred = skl_model.predict(x_test)
    print(f"{skl_pred=}")

    # Convert Sklearn model to ONNX
    onnx_model = convert_to_onnx(skl_model)
    onnx_path = Path(__file__).resolve().parent / "linear_regression.onnx"
    save_onnx_model(onnx_model, onnx_path)

    onnx_input = x_test.to_numpy().astype("float32")
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onnx_pred = sess.run([label_name], {input_name: onnx_input})[0][:, 0]
    print(f"{onnx_pred=}")
