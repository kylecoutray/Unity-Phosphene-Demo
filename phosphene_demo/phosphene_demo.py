# Phosphene front-end demo: camera/image -> retina preprocessing -> CNN feature encoder
# -> electrode grid -> Gaussian phosphene renderer -> (placeholder) cortical map -> display/UDP stream.
import argparse
import socket
import struct

import cv2
import numpy as np
from utils import grid_points, make_gaussian_field, draw_tiled, downsample_gray, normalize01
from retinotopy import visual_to_cortical

try:
    #torch/ torchvision are optional -- if unavailable fall back to raw intensity encoding
    import torch
    import torch.nn.functional as F
    from torchvision import models
    from torchvision.models.feature_extraction import create_feature_extractor
except ImportError:  # either i didn't instal torch/torchvision or i didnt activate the env
    torch = None
    F = None
    models = None
    create_feature_extractor = None

UDP_MAGIC_PHOS = b"PHOS"  # electrode grid identifier
UDP_MAGIC_VID = b"VID0"  # video stream identifier
UDP_FMT_UINT8 = 0
UDP_FMT_FLOAT32 = 1
UDP_FMT_JPEG = 10
UDP_HEADER = struct.Struct("!4sIHHIIHHBB")  # this is the shared header for chunked UDP payloads
# see obsidian notes for breakdown of fields
UDP_HEADER_SIZE = UDP_HEADER.size


class Preprocessor:
    """ center crops frames and applies optional retinal-inspired transforms before the CNN encoder """

    def __init__(
        self,
        target_size=256,
        gamma=1.0,
        clahe=False,
        dog=False,
        dog_sigma=(1.0, 2.0),
    ):
        self.target_size = target_size  # spatial resolution fed downstream
        self.gamma = gamma  # photoreceptor style nonlinearity
        self.clahe = clahe  # local contrast enhancement
        self.dog = dog  # difference of gaussians toggle (center surround)
        self.dog_sigma = dog_sigma
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    def __call__(self, frame_rgb):
        base = prepare_rgb_frame(frame_rgb, target=self.target_size)  # square crop + resize
        proc = base.astype(np.float32) / 255.0
        if self.gamma != 1.0:
            proc = np.power(np.clip(proc, 0.0, 1.0), self.gamma)  # gamma compression/expansion
        if self.clahe:
            lab = cv2.cvtColor((proc * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            proc = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0  # return normalized RGB
        if self.dog:
            gray = cv2.cvtColor((proc * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            sigma1, sigma2 = self.dog_sigma
            blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
            blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
            dog_map = normalize01(np.clip(blur1 - blur2, 0.0, None))  # keep excitatory response only
            proc = np.repeat(dog_map[..., None], 3, axis=2)  # stack into pseudo-RGB for CNN
        return (proc * 255.0).astype(np.uint8), base  # both as uint8 for downstream modules


class RawIntensityEncoder:
    """ baseline (fallback) encoder... grayscale downsample + gamma """

    def __init__(self, grid, gamma=0.8):
        self.grid = grid
        self.gamma = gamma

    def __call__(self, frame_rgb):
        return downsample_gray(frame_rgb, n=self.grid, gamma=self.gamma)


class CNNFeatureEncoder:
    """ frozen torchvision backbone tapped at mid-level layers to produce "electrode" intensities """

    def __init__(
        self,
        model_name,
        grid,
        gamma=0.8,
        input_size=160,
        device=None,
        layer_spec=None,
        mix_spec=None,
        activation="sigmoid",
    ):
        if torch is None or models is None or create_feature_extractor is None:
            raise ImportError(
                "pytorch and torchvision are required for --cnn mode "
                "check to see if I activated venv or `pip install torch torchvision`."
            )

        model_name = model_name.lower()
        self.grid = grid
        self.gamma = gamma
        self.input_size = input_size
        self.activation = activation
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.layers = self._resolve_layers(model_name, layer_spec)
        self.mix_weights = self._resolve_mix_weights(mix_spec)
        self.extractor, self.mean, self.std = self._build_extractor(model_name)
        self.extractor.to(self.device)
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)

    def _resolve_layers(self, model_name, layer_spec):
        if layer_spec:
            layers = [layer.strip() for layer in layer_spec.split(",") if layer.strip()]
            if not layers:
                raise ValueError("Empty --cnn-layers specification.")
            return layers

        if model_name in ("mobilenet", "mobilenet_v2"):
            return ["features.6", "features.12"]
        if model_name in ("vgg16", "vgg"):
            return ["features.16", "features.23"]
        raise ValueError(f"Unsupported CNN backbone '{model_name}'.")

    def _resolve_mix_weights(self, mix_spec):
        if mix_spec is None:
            return None
        weights = [float(x) for x in mix_spec.split(",") if x.strip()]
        return weights if weights else None

    def _build_extractor(self, model_name):
        if model_name in ("mobilenet", "mobilenet_v2"):
            weights = models.MobileNet_V2_Weights.DEFAULT
            backbone = models.mobilenet_v2(weights=weights)
        elif model_name in ("vgg16", "vgg"):
            weights = models.VGG16_Weights.DEFAULT
            backbone = models.vgg16(weights=weights)
        else:
            raise ValueError(f"Unsupported CNN backbone '{model_name}'.")

        available = dict(backbone.named_modules()).keys()
        missing = [layer for layer in self.layers if layer not in available]
        if missing:
            raise ValueError(
                f"Requested layers {missing} not found in {model_name}. "
                f"Available example: {sorted(list(available))[:10]}..."
            )

        return_nodes = {layer: f"feat_{i}" for i, layer in enumerate(self.layers)}
        extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        mean = torch.tensor(weights.meta.get("mean", [0.485, 0.456, 0.406]), dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor(weights.meta.get("std", [0.229, 0.224, 0.225]), dtype=torch.float32).view(
            1, 3, 1, 1
        )
        return extractor, mean.to(self.device), std.to(self.device)

    def __call__(self, frame_rgb):
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # NHWC->NCHW
        tensor = tensor.to(self.device)
        tensor = F.interpolate(
            tensor, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
        )  # resize to network input
        tensor = (tensor - self.mean) / self.std  # normalize with imagenet statistics

        with torch.no_grad():
            feats = self.extractor(tensor)

        accum = None
        for idx, feat in enumerate(feats.values()):
            pooled = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))  # map feature map -> electrode grid
            energy = torch.mean(torch.abs(pooled), dim=1, keepdim=False)  # collapse channel dimension
            if self.mix_weights and idx < len(self.mix_weights):
                energy = energy * self.mix_weights[idx]  # weighted blend of tapped layers
            accum = energy if accum is None else accum + energy

        combined = accum / max(len(feats), 1)
        combined = apply_activation(combined, self.activation)
        out = combined.squeeze(0).detach().cpu().numpy().astype(np.float32)
        out = normalize01(out)  # ensure output lies in [0,1]
        if self.gamma != 1.0:
            out = np.power(out, self.gamma)  # perceptual compression
        return out


def apply_activation(tensor, activation):
    """ applies a configurable non-linearity to "electrode" activations """
    if torch is None:
        raise RuntimeError("no torch available for CNNFeatureEncoder activation")
    if activation is None:
        return tensor
    act = activation.lower()
    if act == "identity":
        return tensor
    if act == "sigmoid":
        return torch.sigmoid(tensor)
    if act == "relu":
        return torch.relu(tensor)
    if act == "tanh":
        return torch.tanh(tensor)
    if act == "softplus":
        return F.softplus(tensor)
    if act.startswith("power"):
        parts = act.split(":")
        exponent = float(parts[1]) if len(parts) > 1 else 1.0
        return torch.pow(torch.clamp(tensor, min=0.0), exponent)
    raise ValueError(f"Unsupported activation '{activation}'.")


class PhospheneRenderer:
    """Paints each electrode as a Gaussian blob over the visual field."""

    def __init__(self, grid, sigma=0.06, vis_h=256, vis_w=256):
        self.sigma = sigma
        self.vis_h = vis_h
        self.vis_w = vis_w
        self._grid = None
        self._points = None
        self.update_grid(grid)

    def update_grid(self, grid):
        if grid != self._grid:
            self._grid = grid
            self._points = grid_points(grid)

    def __call__(self, weights):
        if weights.shape[0] != self._grid * self._grid:
            raise ValueError(
                f"Weight vector size {weights.shape[0]} incompatible with grid {self._grid}."
            )
        return make_gaussian_field(self._points, weights, self.vis_h, self.vis_w, sigma=self.sigma)


class CorticalMapper:
    """Placeholder log-polar transform from visual field to cortical sheet."""

    def __init__(self, cortex_h=256, cortex_w=384):
        self.cortex_h = cortex_h
        self.cortex_w = cortex_w

    def __call__(self, field):
        return visual_to_cortical(field, cortex_h=self.cortex_h, cortex_w=self.cortex_w)


def prepare_rgb_frame(frame_rgb, target=256):
    """center-crops the largest square from the frame and resizes to `target` pixels."""

    h, w = frame_rgb.shape[:2]
    min_side = min(h, w)
    startx = w // 2 - min_side // 2
    starty = h // 2 - min_side // 2
    cropped = frame_rgb[starty : starty + min_side, startx : startx + min_side]
    if cropped.shape[0] != target:
        cropped = cv2.resize(cropped, (target, target), interpolation=cv2.INTER_AREA)
    return cropped


def send_udp_chunks(sock, addr, magic, frame_id, dim0, dim1, fmt_code, payload, chunk_limit):
    """ streams a payload over UDP in fixed-size chunks with a simple header"""

    if sock is None or addr is None or payload is None:
        return True
    total_len = len(payload)
    if total_len == 0:
        return True
    chunk_limit = max(UDP_HEADER_SIZE + 1, min(chunk_limit, 65000))  # honour header size + UDP MTU
    chunk_payload = chunk_limit - UDP_HEADER_SIZE
    total_chunks = max(1, (total_len + chunk_payload - 1) // chunk_payload)
    mv = memoryview(payload)  # zero-copy slices
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_payload
        end = min(start + chunk_payload, total_len)
        sub = mv[start:end]
        header = UDP_HEADER.pack(
            magic,
            frame_id,
            total_chunks,
            chunk_idx,
            total_len,
            start,
            dim0,
            dim1,
            fmt_code,
            0,
        )
        try:
            sock.sendto(header + sub.tobytes(), addr)
        except OSError:
            sock.close()
            return False
    return True

def main():
    # ---- CLI configuration ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help=" webcam index: 0 is normally my mac default")
    ap.add_argument("--image", type=str, default=None, help="path to image (if using one)")
    ap.add_argument("--grid", type=int, default=32, help=" grid size (electrode count per side)")
    ap.add_argument("--sigma", type=float, default=0.06, help=" phosphene spread in normalized units")
    ap.add_argument("--gamma", type=float, default=0.8, help=" contrast compression")
    ap.add_argument("--cortex_h", type=int, default=256)
    ap.add_argument("--cortex_w", type=int, default=384)
    ap.add_argument("--visual_h", type=int, default=256, help="height of intermediate phosphene raster.")
    ap.add_argument("--visual_w", type=int, default=256, help=" width of intermediate phosphene raster.")
    ap.add_argument("--pre-size", type=int, default=256, help="target spatial size before encoding.")
    ap.add_argument("--pre-gamma", type=float, default=1.0, help="gamma applied during preprocessing.")
    ap.add_argument("--pre-clahe", action="store_true", help="apply CLAHE to boost local contrast.")
    ap.add_argument("--pre-dog", action="store_true", help="apply Difference-of-Gaussians edge emphasis.")
    ap.add_argument(
        "--pre-dog-sigma",
        type=str,
        default="1.0,2.0",
        help="Comma-separated sigmas for DoG (small,big).",
    )
    ap.add_argument("--cnn", type=str, default=None, help="enable CNN feature encoding (mobilenet or vgg16).")
    ap.add_argument("--cnn-layers", type=str, default=None, help="comma-separated layer taps for the CNN backbone.")
    ap.add_argument("--cnn-input", type=int, default=160, help="spatial size fed into the CNN encoder.")
    ap.add_argument("--cnn-mix", type=str, default=None, help="comma weights matching tapped CNN layers.")
    ap.add_argument(
        "--cnn-activation",
        type=str,
        default="sigmoid",
        help="activation applied after layer fusion (sigmoid, tanh, relu, softplus, power:x, identity).",
    )
    ap.add_argument("--device", type=str, default=None, help="torch device to run CNN on (e.g., cuda, cpu).")
    ap.add_argument("--udp", type=str, default=None, help="send electrode grid over UDP host:port.")
    ap.add_argument(
        "--udp-format",
        choices=["float32", "uint8"],
        default="float32",
        help="Numeric format for UDP payload.",
    )
    ap.add_argument(
        "--udp-chunk",
        type=int,
        default=4000,
        help="Maximum UDP payload bytes per datagram (will chunk if exceeded).",
    )
    ap.add_argument(
        "--udp-video",
        type=str,
        default=None,
        help="Send center-cropped RGB frame over UDP host:port (JPEG).",
    )
    ap.add_argument("--udp-video-quality", type=int, default=80, help="JPEG quality for --udp-video (0-100).")
    args = ap.parse_args()
    # ----------------------------

    if args.camera is None and args.image is None:
        print("Provide --camera N or --image path")
        return

    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Failed to open camera", args.camera)
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # keep upstream bandwidth low
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    try:
        dog_parts = [float(x) for x in args.pre_dog_sigma.split(",") if x.strip()]
        dog_sigma = tuple(dog_parts[:2]) if len(dog_parts) >= 2 else (1.0, 2.0)
    except ValueError:
        dog_sigma = (1.0, 2.0)

    preprocessor = Preprocessor(
        target_size=args.pre_size,
        gamma=args.pre_gamma,
        clahe=args.pre_clahe,
        dog=args.pre_dog,
        dog_sigma=dog_sigma,
    )
    encoder = RawIntensityEncoder(args.grid, gamma=args.gamma)  # default fallback encoder
    if args.cnn is not None:
        try:
            encoder = CNNFeatureEncoder(
                args.cnn,
                grid=args.grid,
                gamma=args.gamma,
                input_size=args.cnn_input,
                device=args.device,
                layer_spec=args.cnn_layers,
                mix_spec=args.cnn_mix,
                activation=args.cnn_activation,
            )
            print(f"[CNN] Using {args.cnn} backbone on {encoder.device} with layers {encoder.layers}.")
            if encoder.mix_weights:
                print(f"[CNN] Layer mix weights: {encoder.mix_weights}")
        except Exception as exc:
            print(f"Failed to initialize CNN feature encoder: {exc}")
            if cap is not None:
                cap.release()
            return

    renderer = PhospheneRenderer(args.grid, sigma=args.sigma, vis_h=args.visual_h, vis_w=args.visual_w)
    cortical = CorticalMapper(cortex_h=args.cortex_h, cortex_w=args.cortex_w)

    udp_sock = None
    udp_addr = None
    udp_frame_id = 0
    if args.udp:
        try:
            host, port = args.udp.split(":")
            udp_addr = (host, int(port))
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[UDP] Streaming electrode grid to {udp_addr} as {args.udp_format}.")
            if args.udp_chunk <= UDP_HEADER_SIZE:
                raise ValueError(f"--udp-chunk must exceed UDP header size ({UDP_HEADER_SIZE}).")
        except Exception as exc:
            print(f"Failed to configure UDP target '{args.udp}': {exc}")
            if cap is not None:
                cap.release()
            return

    udp_video_sock = None
    udp_video_addr = None
    udp_video_frame_id = 0
    if args.udp_video:
        try:
            host, port = args.udp_video.split(":")
            udp_video_addr = (host, int(port))
            udp_video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[UDP] Streaming center-crop video to {udp_video_addr} as JPEG (quality={args.udp_video_quality}).")
            if args.udp_chunk <= UDP_HEADER_SIZE:
                raise ValueError(f"--udp-chunk must exceed UDP header size ({UDP_HEADER_SIZE}).")
        except Exception as exc:
            print(f"Failed to configure UDP video target '{args.udp_video}': {exc}")
            if cap is not None:
                cap.release()
            if udp_sock:
                udp_sock.close()
            return

    if cap is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop late frames to minimize latency
    while True:
        if cap is not None:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Preprocessor handles crop/resize + optional CLAHE/DoG.
            pre_frame, display_frame = preprocessor(frame_rgb)
        else:
            frame_bgr = cv2.imread(args.image)
            if frame_bgr is None:
                print("Failed to read image", args.image)
                return
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pre_frame, display_frame = preprocessor(frame_rgb)

        n = args.grid  # electrode grid size

        small = encoder(pre_frame)  # encoder outputs n x n intensities in [0,1]
        weights = small.reshape(-1)  # flatten for renderer/UDP

        renderer.update_grid(n)
        field = renderer(weights)  # visual phosphene field (Gaussian sum)
        cortex = cortical(field)  # placeholder cortical map

        small_up = cv2.resize(small, (args.visual_w, args.visual_h), interpolation=cv2.INTER_NEAREST)

        # Stream electrode grid to Unity
        if udp_sock and udp_addr:
            fmt_code = UDP_FMT_FLOAT32 if args.udp_format == "float32" else UDP_FMT_UINT8
            if args.udp_format == "float32":
                payload = small.astype(np.float32, copy=False).tobytes(order="C")
            else:
                payload = (normalize01(small) * 255.0).astype(np.uint8, copy=False).tobytes(order="C")
            udp_frame_id = (udp_frame_id + 1) & 0xFFFFFFFF
            ok = send_udp_chunks(
                udp_sock,
                udp_addr,
                UDP_MAGIC_PHOS,
                udp_frame_id,
                n,
                n,
                fmt_code,
                payload,
                args.udp_chunk,
            )
            if not ok:
                print("[UDP] Electrode stream send failed; disabling.")
                udp_sock = None
                udp_addr = None

        # stream the reference center-crop as JPEG for side-by-side comparison.
        if udp_video_sock and udp_video_addr:
            # opencv expects BGR for JPEG encoding.
            bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            ok_enc, jpeg = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(args.udp_video_quality, 1, 100))],
            )
            if ok_enc:
                udp_video_frame_id = (udp_video_frame_id + 1) & 0xFFFFFFFF
                ok_vid = send_udp_chunks(
                    udp_video_sock,
                    udp_video_addr,
                    UDP_MAGIC_VID,
                    udp_video_frame_id,
                    display_frame.shape[1],
                    display_frame.shape[0],
                    UDP_FMT_JPEG,
                    jpeg.tobytes(),
                    args.udp_chunk,
                )
                if not ok_vid:
                    print("[UDP] Video stream send failed; disabling.")
                    udp_video_sock = None
                    udp_video_addr = None


        # compose diagnostic panels for quick visual inspection.
        tiled = draw_tiled(
            [display_frame, pre_frame, small_up, field, cortex],
            labels=[
                "center-crop",
                "preprocessed",
                f"{n}x{n} electrodes",
                "phosphenes",
                "cortical map",
            ],
            scale=0.7
        )
        cv2.imshow("Video -> Phosphene -> Cortex", tiled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if cap is None:
            cv2.waitKey(0)
            break

    if cap is not None:
        cap.release()
    if udp_sock:
        udp_sock.close()
    if udp_video_sock:
        udp_video_sock.close()
    cv2.destroyAllWindows()  # clean UI window shutdown

if __name__ == "__main__":
    main()
