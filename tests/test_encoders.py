import pytest
import torch

from pytorch_toolbelt.modules import encoders as E


def get_supported_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


@pytest.mark.parametrize('device', get_supported_devices())
def test_inplace_abn(device):
    try:
        from inplace_abn import InPlaceABN
    except ImportError:
        return

    net_classic_bn = E.SEResnet50Encoder().to(device)
    net_inplace_abn = E.SEResnet50Encoder(abn_block=InPlaceABN).to(device)

    x: torch.Tensor = torch.randn((4, 3, 224, 224), requires_grad=True).to(
        device)
    y1: torch.Tensor = net_classic_bn(x).cpu()
    y2: torch.Tensor = net_inplace_abn(x).cpu()

    assert torch.isclose(y1, y2)
    assert torch.isclose(y1.grad, y2.grad)
