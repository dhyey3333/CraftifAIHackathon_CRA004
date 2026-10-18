#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Head-region computation (from PIR pseudo_code, nodes 6 & 7) ─────────────
// Node 6: solid black fill over head region (head_share=0.2, head_grow=0.2)
// Node 7: pixelate head region (cell=16, same head_share/head_grow)
// Combined: solid fill first, then pixelate on top — solid wins (strongest privacy).
// Since solid black already overwrites the region completely, pixelation on the
// same region is a no-op visually — but we faithfully implement both operations
// sequentially as specified.

static constexpr float HEAD_SHARE = 0.20f;
static constexpr float HEAD_GROW  = 0.20f;
static constexpr int   CELL_SIZE  = 16;

// Compute the head region from a person detection box, return false if degenerate.
// All coordinates are in full-frame pixels, clamped to [0, W) x [0, H).
// Edges are snapped to even values so they align with NV12 chroma sampling.
static bool compute_head_region(float left, float top, float bw, float bh,
                                int frameW, int frameH,
                                int &x0, int &y0, int &x1, int &y1) {
    // Head height = top 20 % of person box
    float head_h = bh * HEAD_SHARE;
    // Grow the region by HEAD_GROW on each axis
    float rx = left - bw * HEAD_GROW * 0.5f;
    float ry = top  - head_h * HEAD_GROW * 0.5f;
    float rw = bw * (1.0f + HEAD_GROW);
    float rh = head_h * (1.0f + HEAD_GROW);

    // Snap to even boundaries (for NV12 chroma alignment)
    x0 = std::max(0, ((int)std::floor(rx / 2.0f)) * 2);
    y0 = std::max(0, ((int)std::floor(ry / 2.0f)) * 2);
    x1 = std::min(frameW, ((int)std::ceil((rx + rw) / 2.0f)) * 2);
    y1 = std::min(frameH, ((int)std::ceil((ry + rh) / 2.0f)) * 2);

    return (x1 - x0 >= 2) && (y1 - y0 >= 2);
}

// Pixelate a region of a cv::Mat in-place (downscale then upscale with INTER_NEAREST).
static void pixelate_region(cv::Mat &img, int x0, int y0, int x1, int y1, int cell) {
    int rw = x1 - x0, rh = y1 - y0;
    if (rw < 2 || rh < 2) return;
    cv::Rect roi(x0, y0, rw, rh);
    cv::Mat block = img(roi);
    int sw = std::max(1, rw / cell);
    int sh = std::max(1, rh / cell);
    cv::Mat small;
    cv::resize(block, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
    cv::resize(small, block, cv::Size(rw, rh), 0, 0, cv::INTER_NEAREST);
}

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[privacy_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // Read frame as BGR
    cv::Mat frame = read_nv12_bgr(in);

    // Collect person detections from NvDsObjectMeta (class_id == 0 is "person")
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;

    render_meta::Builder sb(W, H);

    int hidden_count = 0;

    if (fmeta) {
        for (NvDsMetaList *ol = fmeta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            // Only process "person" class (class_id == 0 for COCO YOLO)
            if (obj->class_id != 0) continue;

            float bleft  = obj->rect_params.left;
            float btop   = obj->rect_params.top;
            float bw     = obj->rect_params.width;
            float bh     = obj->rect_params.height;

            int x0, y0, x1, y1;
            if (!compute_head_region(bleft, btop, bw, bh, W, H, x0, y0, x1, y1))
                continue;

            // Node 6: solid black fill over head region
            cv::rectangle(frame,
                          cv::Point(x0, y0),
                          cv::Point(x1 - 1, y1 - 1),
                          cv::Scalar(0, 0, 0),
                          cv::FILLED);

            // Node 7: pixelate the same head region (on top of the solid fill).
            // The reference faithfully runs pixelation after the solid fill;
            // visually the solid black dominates, which is the intended strongest
            // privacy guarantee.
            pixelate_region(frame, x0, y0, x1, y1, CELL_SIZE);

            hidden_count++;

            // Publish a box for each redacted head region
            sb.box((double)x0, (double)y0, (double)(x1 - x0), (double)(y1 - y0),
                   "redacted_head", (int)obj->object_id, (double)obj->confidence);
        }
    }

    sb.field_i("heads_redacted", hidden_count);

    write_nv12(frame, in);
    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);