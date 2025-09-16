import torch
from src.models.retriever.two_tower import TwoTowerModel


def test_twotower_shapes():
    model = TwoTowerModel(n_users=10, n_items=20, d_model=16, dropout=0.0)
    u = torch.randint(0, 10, (4,))
    i = torch.randint(0, 20, (4,))
    pos = model.user_tower(u)
    neg = model.item_tower(i)
    assert pos.shape == (4, 16)
    assert neg.shape == (4, 16)


