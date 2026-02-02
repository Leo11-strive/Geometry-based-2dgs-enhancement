import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch

def view(dataset, pipe, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    raise e
                    print('Viewer closed')
                    exit(0)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--port_retries', type=int, default=100, help="If port is busy, try next ports up to this many times (then fall back to ephemeral).")
    parser.add_argument('--port_strict', action='store_true', default=False, help="Do not auto-select a free port when the requested port is in use.")
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    actual_port = network_gui.init(
        args.ip,
        args.port,
        allow_fallback=(not args.port_strict),
        port_retries=args.port_retries,
        fallback_to_ephemeral=True,
    )
    if actual_port != args.port:
        print(f"[GUI] Port {args.port} unavailable; switched to {args.ip}:{actual_port}")
    else:
        print(f"[GUI] Listening on {args.ip}:{actual_port}")
    
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")
