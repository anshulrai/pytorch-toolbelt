import pytest
import torch

from pytorch_toolbelt.modules import encoders as E, ABN
from torch import nn


def get_supported_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def has_inplace_abn():
    try:
        from inplace_abn import InPlaceABN

        return True
    except ImportError:
        return False


@pytest.mark.parametrize('device', get_supported_devices())
@pytest.mark.skipif(not has_inplace_abn(), reason='inplace_abn package is not installed')
def test_inplace_abn_simple(device):
    from inplace_abn import InPlaceABN

    x: torch.Tensor = torch.randn((4, 3, 16, 16), requires_grad=True).to(device)

    conv1 = nn.Conv2d(3, 16, kernel_size=3,padding=1)
    conv2 = nn.Conv2d(16, 16, kernel_size=3,padding=1)
    conv3 = nn.Conv2d(16, 16, kernel_size=3,padding=1)

    net1 = nn.Sequential(conv1,
                         ABN(16),
                         conv2,
                         ABN(16),
                         conv3,
                         ABN(16)).to(device).eval()

    net2 = nn.Sequential(conv1,
                         InPlaceABN(16),
                         conv2,
                         InPlaceABN(16),
                         conv3,
                         InPlaceABN(16)).to(device).eval()

    y1 = net1(x)
    y2 = net2(x)
    assert torch.allclose(y1, y2, atol=1e-5)


@pytest.mark.parametrize("device", get_supported_devices())
@pytest.mark.skipif(not has_inplace_abn(), reason='inplace_abn package is not installed')
def test_inplace_abn(device):
    try:
        from inplace_abn import InPlaceABN
    except ImportError:
        pytest.skip("")

    net_classic_bn = E.SEResNeXt50Encoder(
        abn_block=ABN, abn_params={"activation": "leaky_relu"}
    ).to(device)

    net_inplace_abn = E.SEResNeXt50Encoder(
        abn_block=InPlaceABN, abn_params={"activation": "leaky_relu"}
    ).to(device)

    x: torch.Tensor = torch.randn((4, 3, 224, 224), requires_grad=True).to(device)
    y1: torch.Tensor = net_classic_bn(x)
    y2: torch.Tensor = net_inplace_abn(x)

    for i in range(len(y1)):
        assert torch.allclose(y1[i], y2[i], atol=1e-5)
        assert torch.allclose(y1[i].grad, y2[i].grad, atol=1e-5)
