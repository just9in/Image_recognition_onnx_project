const express = require("express");
const multer = require("multer");
const ort = require("onnxruntime-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(express.json());

const DISTANCE_THRESHOLD = Number(process.env.DISTANCE_THRESHOLD || 0.95);
const COSINE_THRESHOLD = Number(process.env.COSINE_THRESHOLD || 0.62);
const REQUIRED_VARIANT_MATCHES = Number(process.env.REQUIRED_VARIANT_MATCHES || 1);

let arcfaceSession;

/* ===============================
   Load ArcFace Model
================================= */
async function loadModels() {
    arcfaceSession = await ort.InferenceSession.create("./models/arc.onnx");
    console.log("ArcFace ONNX Model Loaded");

    console.log(arcfaceSession.inputMetadata);

}

loadModels();

/* ===============================
   Image Preprocessing
================================= */
async function preprocessTo112(imagePath) {
    return await sharp(imagePath)
        .resize(112, 112)
        .removeAlpha()
        .raw()
        .toBuffer();
}

async function getEmbeddingFromBuffer(rawBuffer) {

    const floatData = new Float32Array(rawBuffer.length);

    for (let i = 0; i < rawBuffer.length; i += 3) {

        const r = rawBuffer[i];
        const g = rawBuffer[i + 1];
        const b = rawBuffer[i + 2];

        // Convert RGB → BGR and normalize
        floatData[i]     = (b - 127.5) / 128.0;
        floatData[i + 1] = (g - 127.5) / 128.0;
        floatData[i + 2] = (r - 127.5) / 128.0;
    }

    const tensor = new ort.Tensor("float32", floatData, [1, 112, 112, 3]);

    const feeds = {};
    feeds[arcfaceSession.inputNames[0]] = tensor;

    const results = await arcfaceSession.run(feeds);
    const output = results[arcfaceSession.outputNames[0]];

    return Array.from(output.data);
}



/* ===============================
   Embedding Utilities
================================= */
function normalizeEmbedding(embedding) {
    const norm = Math.sqrt(
        embedding.reduce((sum, val) => sum + val * val, 0)
    );
    if (!norm || !Number.isFinite(norm)) return embedding;
    return embedding.map(val => val / norm);
}

function euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
}

function cosineSimilarity(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

function averageEmbeddings(embeddings) {
    const length = embeddings[0].length;
    const sum = new Array(length).fill(0);

    for (const emb of embeddings) {
        for (let i = 0; i < length; i++) {
            sum[i] += emb[i];
        }
    }

    return sum.map(v => v / embeddings.length);
}

/* ===============================
   Augmentation (Register + Find)
================================= */
async function getAugmentedEmbeddings(imagePath) {
    const image = sharp(imagePath).removeAlpha();
    const metadata = await image.metadata();

    const width = metadata.width;
    const height = metadata.height;

    const buffers = [];

    // Full image resized
    buffers.push(await preprocessTo112(imagePath));

    // Center crops (92% and 85%)
    if (width && height) {
        const square = Math.min(width, height);
        const ratios = [0.92, 0.85];

        for (const ratio of ratios) {
            const side = Math.floor(square * ratio);
            const left = Math.floor((width - side) / 2);
            const top = Math.floor((height - side) / 2);

            const cropped = await image.clone()
                .extract({ left, top, width: side, height: side })
                .resize(112, 112)
                .raw()
                .toBuffer();

            buffers.push(cropped);
        }
    }

    const embeddings = [];

    for (const buffer of buffers) {
        const embedding = await getEmbeddingFromBuffer(buffer);
        embeddings.push(normalizeEmbedding(embedding));
    }

    return embeddings;
}

/* ===============================
   Build Person Profile
================================= */
function buildPersonProfile(embeddings) {
    const centroid = normalizeEmbedding(averageEmbeddings(embeddings));

    let maxIntraDistance = 0;
    let minIntraSimilarity = Number.POSITIVE_INFINITY;

    for (const emb of embeddings) {
        const distance = euclideanDistance(emb, centroid);
        const similarity = cosineSimilarity(emb, centroid);

        if (distance > maxIntraDistance) maxIntraDistance = distance;
        if (similarity < minIntraSimilarity) minIntraSimilarity = similarity;
    }

    return {
        embeddings,
        centroid,
        maxIntraDistance,
        minIntraSimilarity
    };
}

/* ===============================
   Register Endpoint
================================= */
app.post("/register", upload.array("images", 5), async (req, res) => {

    const name = req.body.name;
    if (!name) return res.status(400).json({ message: "Name is required" });
    if (!req.files || req.files.length === 0)
        return res.status(400).json({ message: "At least one image required" });

    const embeddings = [];

    for (const file of req.files) {
        const variants = await getAugmentedEmbeddings(file.path);
        embeddings.push(...variants);
    }

    const profile = buildPersonProfile(embeddings);

    fs.writeFileSync(
        path.join("data", `${name}.json`),
        JSON.stringify(profile, null, 2)
    );

    res.json({
        message: "Person Registered",
        samplesStored: embeddings.length,
        maxIntraDistance: profile.maxIntraDistance,
        minIntraSimilarity: profile.minIntraSimilarity
    });
});

/* ===============================
   Find Endpoint
================================= */
app.post("/find", upload.single("image"), async (req, res) => {

    const name = req.body.name;
    if (!name) return res.status(400).json({ message: "Name required" });
    if (!req.file) return res.status(400).json({ message: "Image required" });

    const profilePath = path.join("data", `${name}.json`);
    if (!fs.existsSync(profilePath))
        return res.json({ message: "Person not registered" });

    const profile = JSON.parse(fs.readFileSync(profilePath));

    const queryEmbeddings = await getAugmentedEmbeddings(req.file.path);

    let minDistance = Infinity;
    let maxSimilarity = -Infinity;
    let variantMatches = 0;

    for (const query of queryEmbeddings) {
        for (const saved of profile.embeddings) {

            const distance = euclideanDistance(saved, query);
            const similarity = cosineSimilarity(saved, query);

            if (distance < minDistance) minDistance = distance;
            if (similarity > maxSimilarity) maxSimilarity = similarity;

            if (
                distance <= DISTANCE_THRESHOLD &&
                similarity >= COSINE_THRESHOLD
            ) {
                variantMatches++;
            }
        }
    }

    const centroidDistance = euclideanDistance(
        profile.centroid,
        queryEmbeddings[0]
    );

    const centroidSimilarity = cosineSimilarity(
        profile.centroid,
        queryEmbeddings[0]
    );

    const isMatch =
        minDistance <= DISTANCE_THRESHOLD &&
        maxSimilarity >= COSINE_THRESHOLD &&
        variantMatches >= REQUIRED_VARIANT_MATCHES;

    return res.json({
        message: isMatch ? "Person Found" : "Person Not Found",
        minDistance,
        maxSimilarity,
        centroidDistance,
        centroidSimilarity,
        variantMatches,
        thresholds: {
            distance: DISTANCE_THRESHOLD,
            cosine: COSINE_THRESHOLD
        }
    });
});

/* ===============================
   Start Server
================================= */
app.listen(3000, () => {
    console.log("Server running on port 3000");
});

