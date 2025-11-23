import * as tf from '@tensorflow/tfjs';

/**
 * Generates an interpretability heatmap using Batched Occlusion Sensitivity.
 * * @param {tf.GraphModel} model - The loaded TensorFlow.js model
 * @param {tf.Tensor} imgTensor - The input image [1, 224, 224, 3]
 * @param {Array|null} metaDataVector - (Optional) The [1, 3] metadata vector for the 2nd input
 * @param {number} targetClassIdx - The class index to explain (e.g., 1 for Tumor)
 * @param {number} patchSize - Size of the sliding window (e.g., 20px)
 * @param {number} stride - Step size (e.g., 10px). Lower = Higher Res but slower.
 */
export async function generateOcclusionMap(
    model, 
    imgTensor, 
    metaDataVector = null, 
    targetClassIdx = 1,
    patchSize = 24, 
    stride = 12
) {
    const [batch, height, width, channels] = imgTensor.shape;
    
    // 1. GET BASELINE CONFIDENCE
    // We need to know the score of the original unmasked image
    let baselineScore = 0;
    tf.tidy(() => {
        let inputs = imgTensor;
        if (metaDataVector) {
            // If metadata model, inputs is an array [img, meta]
            const metaTensor = tf.tensor(metaDataVector).expandDims(0);
            inputs = [imgTensor, metaTensor];
        }
        const pred = model.predict(inputs);
        baselineScore = tf.softmax(pred).dataSync()[targetClassIdx];
    });

    // 2. GENERATE MASKED BATCHES
    const maskCoords = [];
    const maskedImagesData = [];
    const originalData = await imgTensor.data(); // Float32Array

    // Loop through the image
    for (let y = 0; y <= height - patchSize; y += stride) {
        for (let x = 0; x <= width - patchSize; x += stride) {
            maskCoords.push({ x, y });
            
            // Copy original image
            const maskedData = new Float32Array(originalData);
            
            // Apply the "Grey Box" (0.0 if normalized around 0, or 127 if 0-255)
            // Assuming input is normalized (-2 to 2 range), 0 is roughly grey.
            for (let py = 0; py < patchSize; py++) {
                for (let px = 0; px < patchSize; px++) {
                    const idx = ((y + py) * width + (x + px)) * channels;
                    maskedData[idx] = 0;     // R
                    maskedData[idx + 1] = 0; // G
                    maskedData[idx + 2] = 0; // B
                }
            }
            maskedImagesData.push(maskedData);
        }
    }

    // 3. RUN BATCHED INFERENCE (The Fast Part)
    const BATCH_SIZE = 32; // Send 32 images to GPU at once
    const importanceScores = new Float32Array(maskCoords.length);

    for (let i = 0; i < maskedImagesData.length; i += BATCH_SIZE) {
        tf.tidy(() => {
            const end = Math.min(i + BATCH_SIZE, maskedImagesData.length);
            const chunk = maskedImagesData.slice(i, end);
            
            // Flatten list of arrays into one big buffer
            const batchFlat = new Float32Array(chunk.reduce((acc, val) => [...acc, ...val], []));
            const batchTensor = tf.tensor4d(batchFlat, [chunk.length, height, width, channels]);

            // Handle Inputs
            let inputs;
            if (metaDataVector) {
                // We must tile the metadata vector to match the batch size
                const singleMeta = tf.tensor(metaDataVector).expandDims(0);
                const tiledMeta = singleMeta.tile([chunk.length, 1]);
                inputs = [batchTensor, tiledMeta];
            } else {
                inputs = batchTensor;
            }

            // Predict
            const logits = model.predict(inputs);
            const probs = tf.softmax(logits);
            const scores = probs.slice([0, targetClassIdx], [-1, 1]).dataSync();

            // Calculate Importance (Baseline - Masked)
            // If Baseline was 0.99 and Masked is 0.40, Importance is 0.59 (High)
            for (let j = 0; j < scores.length; j++) {
                const drop = Math.max(0, baselineScore - scores[j]);
                importanceScores[i + j] = drop;
            }
        });
        
        // Non-blocking wait to keep UI responsive
        await new Promise(r => setTimeout(r, 0)); 
    }

    // 4. DRAW HEATMAP ONTO CANVAS
    return renderHeatmap(importanceScores, maskCoords, width, height, patchSize);
}

function renderHeatmap(scores, coords, w, h, patchSize) {
    const canvas = document.createElement('canvas');
    canvas.width = w; 
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Normalize scores (0.0 to 1.0) for visualization
    const maxScore = Math.max(...scores) || 1; 

    scores.forEach((score, i) => {
        const { x, y } = coords[i];
        const intensity = score / maxScore;
        
        // Filter out low noise
        if (intensity > 0.15) {
            // Red overlay, alpha based on importance
            ctx.fillStyle = `rgba(255, 0, 0, ${intensity * 0.8})`; 
            ctx.fillRect(x, y, patchSize, patchSize);
        }
    });

    return canvas.toDataURL(); // Returns base64 PNG
}