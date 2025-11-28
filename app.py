# Particle Advection + Diffusion — Streamlit App
# Save this file as `particle_advection_app.py` and run:
#    pip install streamlit opencv-python-headless numpy
#    streamlit run particle_advection_app.py
# Note: use opencv-python-headless on servers; on local machines 'opencv-python' is fine.

import streamlit as st
import cv2
import numpy as np
import csv
import math
import time
import tempfile
import os
from io import BytesIO

st.set_page_config(page_title="Particle Advection + Diffusion", layout="wide")
st.title("Particle Advection + Diffusion — Upload video and run")

# ---- UI: sidebar parameters ----
st.sidebar.header("Processing parameters")
SCALE = st.sidebar.slider("Scale (fraction of original)", 0.1, 1.0, 0.45, 0.05)
NUM_PARTICLES = st.sidebar.number_input("Number of particles", min_value=1, max_value=2000, value=50, step=1)
DIFFUSION_STD = st.sidebar.slider("Diffusion std (initial)", 0.0, 10.0, 0.6, 0.1)
ARROW_SCALE = st.sidebar.slider("Arrow scale", 1, 40, 12)
MIN_AREA_FOR_MASK = st.sidebar.number_input("Min area for mask (px)", 1, 100000, 100, step=1)
HSV_H_MIN = st.sidebar.slider("HSV hue min", 0, 179, 5)
HSV_S_MIN = st.sidebar.slider("HSV sat min", 0, 255, 80)
HSV_V_MIN = st.sidebar.slider("HSV val min", 0, 255, 50)
HSV_H_MAX = st.sidebar.slider("HSV hue max", 0, 179, 25)
HSV_S_MAX = st.sidebar.slider("HSV sat max", 0, 255, 255)
HSV_V_MAX = st.sidebar.slider("HSV val max", 0, 255, 255)
FB_PYR = st.sidebar.number_input("Farneback pyr_scale", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
FB_LEVELS = st.sidebar.number_input("Farneback levels", min_value=1, max_value=8, value=2, step=1)
FB_WINSIZE = st.sidebar.number_input("Farneback winsize", min_value=4, max_value=50, value=9, step=1)
FB_ITER = st.sidebar.number_input("Farneback iterations", min_value=1, max_value=10, value=2, step=1)
MASK_BLUR = st.sidebar.slider("Mask blur (odd kernel, 0 disables)", 0, 15, 3, step=1)
KILL_IF_LEAVE_MASK = st.sidebar.checkbox("Kill particle when it leaves mask", value=False)
RESPAWN_PARTICLES = st.sidebar.checkbox("Respawn particles to maintain count", value=True)
MAX_FRAMES = st.sidebar.number_input("Max frames to process (0 = all)", min_value=0, max_value=100000, value=0, step=1)
EDGE_MARGIN = st.sidebar.number_input("Edge margin for sampling", min_value=0, max_value=50, value=2, step=1)
MAX_ARROW_LEN = st.sidebar.number_input("Max arrow length (px)", min_value=1, max_value=1000, value=40, step=1)
VERBOSE = st.sidebar.checkbox("Verbose logging", value=True)

st.sidebar.markdown("---")

# ---- Helpers (adapted from provided script) ----

def mask_from_bgr(bgr, hsv_lower, hsv_upper, mask_blur=3):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(hsv_lower, dtype=np.uint8)
    hi = np.array(hsv_upper, dtype=np.uint8)
    m = cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if mask_blur and mask_blur > 0:
        ksz = mask_blur if (mask_blur % 2 == 1) else (mask_blur + 1)
        m = cv2.GaussianBlur(m, (ksz, ksz), 0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    m = cv2.medianBlur(m, 5)
    return m


def get_largest_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    mask_big = np.zeros_like(mask)
    cv2.drawContours(mask_big, [c], -1, 255, -1)
    return mask_big, area


def sample_points_uniform_in_mask(mask, n, margin=2):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return np.zeros((0,2), dtype=np.float32)
    pts = []
    idxs = np.arange(len(xs))
    np.random.shuffle(idxs)
    h,w = mask.shape
    for ii in idxs:
        if len(pts) >= n: break
        x = int(xs[ii]); y = int(ys[ii])
        ymin = max(0, y - margin); ymax = min(h, y + margin + 1)
        xmin = max(0, x - margin); xmax = min(w, x + margin + 1)
        patch = mask[ymin:ymax, xmin:xmax]
        if patch.size == 0: continue
        if np.mean(patch/255.0) < 0.7: continue
        pts.append([float(x), float(y)])
    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] < n and pts.shape[0] > 0:
        extra_idx = np.random.choice(range(pts.shape[0]), n - pts.shape[0])
        pts = np.vstack([pts, pts[extra_idx]])
    return pts


def bilinear_sample(flow, xs_f, ys_f):
    xs = np.asarray(xs_f, dtype=np.float32)
    ys = np.asarray(ys_f, dtype=np.float32)
    was_scalar = False
    if xs.shape == () or ys.shape == ():
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        was_scalar = True
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have same shape")
    h,w = flow.shape[:2]
    x0 = np.floor(xs).astype(int); x1 = x0 + 1
    y0 = np.floor(ys).astype(int); y1 = y0 + 1
    x0 = np.clip(x0, 0, w-1); x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1); y1 = np.clip(y1, 0, h-1)
    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)
    Ia = flow[y0, x0]; Ib = flow[y0, x1]; Ic = flow[y1, x0]; Id = flow[y1, x1]
    res = (Ia * wa[:,None] + Ib * wb[:,None] + Ic * wc[:,None] + Id * wd[:,None])
    if was_scalar:
        return res[0]
    return res


def clamp_positions(pts, W, H):
    pts[:,0] = np.clip(pts[:,0], 0, W-1)
    pts[:,1] = np.clip(pts[:,1], 0, H-1)

# ---- File upload ----
st.markdown("## Upload video")
uploaded = st.file_uploader("Choose a video file (mp4, avi, mov...)", type=['mp4','avi','mov','mkv'])

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    tfile.write(uploaded.read())
    tfile.flush()
    VIDEO = tfile.name
    st.success(f"Saved upload to {VIDEO}")

    run_button = st.button("Run particle advection")

    if run_button:
        with st.spinner("Processing — this can take time depending on video length and resolution..."):
            # prepare parameters
            HSV_LOWER = (int(HSV_H_MIN), int(HSV_S_MIN), int(HSV_V_MIN))
            HSV_UPPER = (int(HSV_H_MAX), int(HSV_S_MAX), int(HSV_V_MAX))
            FB = dict(pyr_scale=float(FB_PYR), levels=int(FB_LEVELS), winsize=int(FB_WINSIZE), iterations=int(FB_ITER), poly_n=5, poly_sigma=1.1, flags=0)
            OUT_VIDEO = VIDEO + "_advected.mp4"
            OUT_CSV = VIDEO + "_advected.csv"
            OUT_IMG = VIDEO + "_advected_sample.png"

            cap = cv2.VideoCapture(VIDEO)
            if not cap.isOpened():
                st.error("Cannot open uploaded video file.")
            else:
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                W = int(orig_w * SCALE); H = int(orig_h * SCALE)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_vid = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (W,H))

                ret, f0 = cap.read()
                if not ret:
                    st.error("Cannot read first frame from video")
                else:
                    f0s = cv2.resize(f0, (W, H))
                    mask0 = mask_from_bgr(f0s, HSV_LOWER, HSV_UPPER, MASK_BLUR)
                    mask_big0, area0 = get_largest_mask(mask0)
                    if mask_big0 is None or area0 < MIN_AREA_FOR_MASK:
                        st.error(f"Mask not found or too small (area={area0}). Tune HSV settings.")
                    else:
                        particles = sample_points_uniform_in_mask(mask_big0, int(NUM_PARTICLES), margin=int(EDGE_MARGIN))
                        if particles.shape[0] == 0:
                            st.error("Failed to sample initial particles inside mask.")
                        else:
                            part_ids = list(range(len(particles)))
                            from io import StringIO
                            csv_text = StringIO()
                            cw = csv.writer(csv_text)
                            cw.writerow(["frame","id","x","y"]) 
                            prev_gray = cv2.cvtColor(f0s, cv2.COLOR_BGR2GRAY)

                            frame_idx = 0
                            saved_sample = False
                            start = time.time()

                            def diffusion_std_for_frame(i):
                                return DIFFUSION_STD * (0.98 ** i)

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            while True:
                                if int(MAX_FRAMES) > 0 and frame_idx >= int(MAX_FRAMES):
                                    break
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_s = cv2.resize(frame, (W,H))
                                gray = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
                                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FB)

                                if particles.shape[0] == 0:
                                    if VERBOSE:
                                        st.write("No particles remaining.")
                                    break

                                xs = particles[:,0]; ys = particles[:,1]
                                sampled = bilinear_sample(flow, xs, ys)  # Nx2
                                noise_scale = diffusion_std_for_frame(frame_idx)
                                noise = np.random.normal(0.0, noise_scale, sampled.shape).astype(np.float32)
                                new_particles = particles + sampled + noise
                                clamp_positions(new_particles, W, H)

                                mask_cur = mask_from_bgr(frame_s, HSV_LOWER, HSV_UPPER, MASK_BLUR)
                                mask_big_cur, area_cur = get_largest_mask(mask_cur)
                                keep_mask = np.ones(len(new_particles), dtype=bool)
                                if KILL_IF_LEAVE_MASK and mask_big_cur is not None:
                                    for i,(nx,ny) in enumerate(new_particles):
                                        xi = int(round(nx)); yi = int(round(ny))
                                        if yi < 0 or yi >= mask_big_cur.shape[0] or xi < 0 or xi >= mask_big_cur.shape[1] or mask_big_cur[yi, xi] == 0:
                                            keep_mask[i] = False

                                particles = new_particles[keep_mask]
                                part_ids = [pid for i,pid in enumerate(part_ids) if keep_mask[i]]

                                if RESPAWN_PARTICLES and mask_big_cur is not None and particles.shape[0] < int(NUM_PARTICLES):
                                    needed = int(NUM_PARTICLES) - particles.shape[0]
                                    extra = sample_points_uniform_in_mask(mask_big_cur, needed, margin=int(EDGE_MARGIN))
                                    if extra.shape[0] > 0:
                                        base = max(part_ids) + 1 if len(part_ids) > 0 else 0
                                        new_ids = list(range(base, base + extra.shape[0]))
                                        particles = np.vstack([particles, extra])
                                        part_ids.extend(new_ids)

                                for i,(x,y) in enumerate(particles):
                                    cw.writerow([frame_idx, part_ids[i], float(x), float(y)])

                                vis = frame_s.copy()
                                if mask_big_cur is not None:
                                    contours, _ = cv2.findContours(mask_big_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        cv2.drawContours(vis, contours, -1, (0,255,255), 1)

                                if particles.shape[0] > 0:
                                    vis_sampled = bilinear_sample(flow, particles[:,0], particles[:,1])
                                    for i,(x,y) in enumerate(particles):
                                        vx, vy = float(vis_sampled[i,0]), float(vis_sampled[i,1])
                                        ex_f = x + ARROW_SCALE * vx
                                        ey_f = y + ARROW_SCALE * vy
                                        dx = ex_f - x; dy = ey_f - y
                                        mag = math.hypot(dx, dy)
                                        if mag > MAX_ARROW_LEN and mag > 1e-8:
                                            factor = MAX_ARROW_LEN / mag
                                            dx *= factor; dy *= factor
                                            ex_f = x + dx; ey_f = y + dy
                                        xi = int(round(x)); yi = int(round(y))
                                        ex = int(round(ex_f)); ey = int(round(ey_f))
                                        if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                                            color = (255,255,255)
                                        else:
                                            ang = (math.degrees(math.atan2(vy, vx)) + 360) % 360
                                            hue = int(ang/2)
                                            color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue,255,200]]]), cv2.COLOR_HSV2BGR)[0,0])
                                        cv2.arrowedLine(vis, (xi,yi), (ex,ey), color, 1, tipLength=0.3)
                                        cv2.circle(vis, (xi,yi), 2, (255,255,255), -1)

                                out_vid.write(vis)
                                if not saved_sample:
                                    cv2.imwrite(OUT_IMG, vis)
                                    saved_sample = True

                                prev_gray = gray.copy()
                                frame_idx += 1

                                if total_frames > 0:
                                    progress = min(1.0, frame_idx / float(total_frames))
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processed {frame_idx}/{total_frames} frames — particles={len(particles)}")

                            # finished loop
                            csv_bytes = csv_text.getvalue().encode("utf-8")
                            csv_text.close()
                            cap.release()
                            out_vid.release()
                            elapsed = time.time() - start
                            st.success(f"DONE — processed {frame_idx} frames in {elapsed:.1f}s")

                            # show sample image and provide downloads
                            if os.path.exists(OUT_IMG):
                                st.image(OUT_IMG, caption="Sample frame with particles", use_column_width=True)

                            with open(OUT_VIDEO, 'rb') as f:
                                video_bytes = f.read()
                            st.download_button("Download advected video", data=video_bytes, file_name=os.path.basename(OUT_VIDEO), mime='video/mp4')

                            st.download_button("Download particle CSV", data=csv_bytes, file_name=os.path.basename(OUT_CSV), mime='text/csv')

                            # cleanup temp upload if any
                            try:
                                os.remove(VIDEO)
                            except Exception:
                                pass

else:
    st.info("Upload a video in the left panel to begin.")

st.markdown("---")
st.caption("Script adapted from user-provided Colab-ready code. If your video is large, consider downscaling or running locally.")
