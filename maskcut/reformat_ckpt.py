import torch

if __name__ == "__main__":
    pretrained_pth = "pretrained/checkpoint_vit_base_tomo.pth"
    new_state_dict = {}
    state_dict = torch.load(pretrained_pth, map_location=torch.device('cpu'))['teacher']
    for key in state_dict.keys():
        if key.startswith('module.backbone.'):
            new_key = key.replace('module.backbone.', '')
        elif key.startswith('module.'):
            new_key = key.replace('module.', '')
        else:
            new_key = key

        new_state_dict.update({new_key : state_dict[key]})

    torch.save(new_state_dict, "pretrained/checkpoint_vit_base_tomo_2.pth")

