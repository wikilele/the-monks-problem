import unittest
from dataset import DataSet


def test_encode1():
    ts = DataSet('data/monks-1.test')
    result = ts.encode_1ofk([2,1,1,2,3,1])
    
    assert len(result) == 17
    
    correct_list = [0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0]

    for i in range(17):
        assert result[i] == correct_list[i]