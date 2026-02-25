import torch

def sample(model, diffusion, past_frames, device):

    model.eval()

    with torch.no_grad():

        x = torch.randn(
            (past_frames.size(0), 3, past_frames.size(-2), past_frames.size(-1))
        ).to(device)

        for i in reversed(range(diffusion.timesteps)):

            t = torch.full((past_frames.size(0),), i, device=device, dtype=torch.long)

            x_input = torch.cat([past_frames, x.unsqueeze(1)], dim=1)

            predicted_noise = model(x_input, t)

            alpha = diffusion.alpha[i].to(device)
            alpha_hat = diffusion.alpha_hat[i].to(device)
            beta = diffusion.beta[i].to(device)

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                1 / torch.sqrt(alpha)
                * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                + torch.sqrt(beta) * noise
            )

    model.train()
    return x