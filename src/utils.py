import torch

def resume(model, device, resume_checkpoint):
    print('=> loading checkpoint : "{}"'.format(resume_checkpoint))
    if str(device) == 'cpu':
        model.load_state_dict(torch.load(resume_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    else:
        model.load_state_dict(torch.load(resume_checkpoint)['state_dict'])
