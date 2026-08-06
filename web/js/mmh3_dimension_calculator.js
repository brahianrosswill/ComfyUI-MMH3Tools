import { app } from "../../../scripts/app.js";

// Option lists are computed here rather than fetched, so there is no async race with
// ComfyUI restoring saved widget values on workflow load. Must stay in step with
// build_options() / native_canvas() in mmh3tools/nodes_util.py.
const GRID = 32;
const BASE_SHORT_EDGE = 768;
const MAX_PIXELS = 768 * 1344;
const MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 2.0];

const RATIOS = [
    [21, 9, "21:9 - ultrawide, cinematic", "9:21 - ultrawide portrait"],
    [16, 9, "16:9 - YouTube, HD, TV",      "9:16 - TikTok, Reels, Shorts"],
    [3,  2, "3:2 - photography, DSLR",     "2:3 - portrait photo"],
    [4,  3, "4:3 - classic TV, monitor",   "3:4 - tablet portrait"],
    [1,  1, "1:1 - square, Instagram",     "1:1 - square, Instagram"],
];

// Python's round() is banker's rounding (half to even); JS Math.round() is half-up.
// They diverge at exact .5 - e.g. 3:2 gives 352*1.5/32 = 16.5, so Python snaps to 512
// and Math.round would give 544. Match Python, since Python parses the chosen value.
function roundHalfEven(x) {
    const f = Math.floor(x);
    const diff = x - f;
    if (diff > 0.5) return f + 1;
    if (diff < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
}

const snap32 = (x) => Math.max(GRID, roundHalfEven(x / GRID) * GRID);

function nativeCanvas(rl, rs) {
    const r = rl / rs;
    let w = BASE_SHORT_EDGE * r, h = BASE_SHORT_EDGE;
    if (w * h > MAX_PIXELS) {
        const s = Math.sqrt(MAX_PIXELS / (w * h));
        w *= s; h *= s;
    }
    return [snap32(w), snap32(h)];
}

function buildOptions(rl, rs, landscape) {
    const r = rl / rs;
    const [nw, nh] = nativeCanvas(rl, rs);
    const entries = MEGAPIXELS.map((mp) => {
        const h = snap32(Math.sqrt((mp * 1e6) / r));
        return [snap32(h * r), h];
    });
    entries.push([nw, nh]);

    const uniq = new Map();
    for (const [lw, lh] of entries) uniq.set(`${lw}x${lh}`, [lw, lh]);
    const sorted = [...uniq.values()].sort((a, b) => a[0] * a[1] - b[0] * b[1]);

    const out = [];
    const seen = new Set();
    for (const [lw, lh] of sorted) {
        const w = landscape ? lw : lh;
        const h = landscape ? lh : lw;
        let tag = `${w}x${h}  ${((w * h) / 1e6).toFixed(2)}MP`;
        if (lw === nw && lh === nh) tag += "  [native]";
        if (!seen.has(tag)) { seen.add(tag); out.push(tag); }
    }
    return out;
}

// "16:9 - YouTube, HD, TV" -> "16:9";  "9:16 - TikTok..." -> "16:9"
function toRatioKey(label) {
    const raw = String(label ?? "").split("-")[0].trim();
    const [a, b] = raw.split(":").map(Number);
    if (!a || !b) return "16:9";
    return a >= b ? `${a}:${b}` : `${b}:${a}`;
}

const landscapeLabels = () => Object.fromEntries(RATIOS.map(([l, s, ll]) => [`${l}:${s}`, ll]));
const portraitLabels  = () => Object.fromEntries(RATIOS.map(([l, s, , pl]) => [`${l}:${s}`, pl]));

app.registerExtension({
    name: "mmh3.DimensionCalculator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MMH3DimensionCalculator") return;

        function refresh(node, { keepValue = true } = {}) {
            const ratioW = node.widgets?.find((w) => w.name === "ratio");
            const orientW = node.widgets?.find((w) => w.name === "orientation");
            const resW = node.widgets?.find((w) => w.name === "resolution");
            if (!ratioW || !orientW || !resW) return false;

            const isPortrait = orientW.value === "Portrait";
            const map = isPortrait ? portraitLabels() : landscapeLabels();
            const key = toRatioKey(ratioW.value);

            ratioW.options.values = Object.values(map);
            ratioW.value = map[key] ?? Object.values(map)[0];

            const [rl, rs] = key.split(":").map(Number);
            const options = buildOptions(rl, rs, !isPortrait);
            const prev = resW.value;
            resW.options.values = options;
            resW.value = (keepValue && options.includes(prev))
                ? prev
                : (options.find((o) => o.includes("[native]")) ?? options[0]);

            node.graph?.setDirtyCanvas(true, true);
            return true;
        }

        function wireCallbacks(node) {
            if (node.__mmh3Wired) return;
            for (const name of ["ratio", "orientation"]) {
                const w = node.widgets?.find((x) => x.name === name);
                if (!w) return;                 // not ready yet, try again next tick
                const orig = w.callback;
                w.callback = function () {
                    const r = orig?.apply(this, arguments);
                    refresh(node);
                    return r;
                };
            }
            node.__mmh3Wired = true;
        }

        // Retry briefly: with the V3 schema API widgets are not always built by onAdded.
        function refreshWhenReady(node, tries = 10) {
            if (refresh(node)) { wireCallbacks(node); return; }
            if (tries > 0) setTimeout(() => refreshWhenReady(node, tries - 1), 50);
        }

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            onAdded?.apply(this, arguments);
            refreshWhenReady(this);
        };

        // Runs AFTER saved widget values are restored, so the list is rebuilt around
        // the real ratio/orientation and the saved resolution survives.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            refreshWhenReady(this);
        };
    },
});
