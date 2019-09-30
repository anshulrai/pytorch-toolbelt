import pytest
import torch

from pytorch_toolbelt.modules import encoders as E, ABN
from torch import nn


def get_supported_devices():
    devices=["cpu"]
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


@pytest.mark.parametrize('device', get_supported_devices())
def test_inplace_abn(device):
    from inplace_abn import ABN as OutOfPlaceABN
    from inplace_abn import InPlaceABN

    x: torch.Tensor = torch.randn((4, 3, 224, 224), requires_grad=True).to(device)

    conv1 = nn.Conv2d(3, 16, kernel_size=3,padding=1)
    conv2 = nn.Conv2d(16, 16, kernel_size=3,padding=1)
    conv3 = nn.Conv2d(16, 16, kernel_size=3,padding=1)

    net1 = nn.Sequential(conv1,
                         OutOfPlaceABN(16),
                         conv2,
                         OutOfPlaceABN(16),
                         conv3,
                         OutOfPlaceABN(16)).eval()

    net2 = nn.Sequential(conv1,
                         InPlaceABN(16),
                         conv2,
                         InPlaceABN(16),
                         conv3,
                         InPlaceABN(16)).eval()

    y1 = net1(x)
    y2 = net2(x)
    assert torch.allclose(y1,y2)


@pytest.mark.parametrize('device', get_supported_devices())
def test_inplace_abn(device):
    abn = ABN

    try:
        from inplace_abn import ABN as OutOfPlaceABN
        from inplace_abn import InPlaceABN
        abn = OutOfPlaceABN

    except ImportError:
        return
    torch.manual_seed(42)

    net_classic_bn = E.SEResNeXt50Encoder(abn_block=abn, abn_params={'activation':'leaky_relu'}).to(device).eval()
    net_inplace_abn = E.SEResNeXt50Encoder(abn_block=InPlaceABN, abn_params={'activation':'leaky_relu'}).to(device).eval()

    x: torch.Tensor = torch.randn((4, 3, 224, 224), requires_grad=True).to(device)
    y1: torch.Tensor = net_classic_bn(x)
    y2: torch.Tensor = net_inplace_abn(x)

    n_outputs = len(y1)
    for i in range(n_outputs):
        diff_y = y1[i] - y2[i]
        # diff_y_grad = y1[i].grad - y2[i].grad

        max_diff = diff_y.max()
        # max_grad_diff = diff_y_grad.max()
        print(max_diff)

        # assert torch.allclose(y1[i], y2[i])
        # assert torch.allclose(y1[i].grad, y2[i].grad)
