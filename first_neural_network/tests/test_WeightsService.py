import unittest
from NeuralNetwork.WeightsService import WeightsService


def test_get_bias():
    ws = WeightsService(-0.7,0.7)
    bias = ws.get_bias()
    assert bias >= -0.7 and bias <= 0.7

def test_get_weights():
    ws = WeightsService(-0.7,0.7)
    weights = ws.get_weights(4)

    assert len(weights) == 4
    for w in weights:
        assert w >= -0.7 and w <= 0.7