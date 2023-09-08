from models.res_unet.unet_model import ResUNet, PADL


def build_model(net_arch,pretrained):
    """
    return models
    """
    if  net_arch == "res_unet":
        model = ResUNet(resnet='resnet34', num_classes=7, pretrained=pretrained)
    elif net_arch == "PADL":
        model = PADL(resnet='resnet34', num_classes=2, rater_num=6, pretrained=True)
    return model
