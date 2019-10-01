import pytest
import torch

from pytorch_toolbelt.modules import encoders as E, ABN


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


@pytest.mark.parametrize("device", get_supported_devices())
@pytest.mark.skipif(has_inplace_abn())
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
    y1: torch.Tensor = net_classic_bn(x).cpu()
    y2: torch.Tensor = net_inplace_abn(x).cpu()

    assert torch.isclose(y1, y2)
    assert torch.isclose(y1.grad, y2.grad)
