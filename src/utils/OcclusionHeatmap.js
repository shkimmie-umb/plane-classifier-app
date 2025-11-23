import * as tf from '@tensorflow/tfjs';

export async function generateOcclusionHeatmap(
    model, 
    imgTensor, // This assumes imgTensor is ALREADY preprocessed (0-1, Normalized)
    targetClassIdx, 
    extraInputTensor = null, 
    patchSize = 28, 
    stride = 14
) {
    // We need to work with the tensor dimensions.
    // NOTE: Check if channels are at index 1 (NCHW) or 3 (NHWC)
    const shape = imgTensor.shape; // e.g. [1, 224, 224, 3]
    const isNCHW = shape[1] === 3;
    
    const height = isNCHW ? shape[2] : shape[1];
    const width = isNCHW ? shape[3] : shape[2];
    const channels = isNCHW ? shape[1] : shape[3];

    // 1. Get Baseline
    let baselineScore;
    tf.tidy(() => {
        const inputs = extraInputTensor ? [imgTensor, extraInputTensor] : imgTensor;
        const preds = model.predict(inputs);
        const probs = tf.softmax(preds);
        baselineScore = probs.dataSync()[targetClassIdx];
    });

    // 2. Prepare Accumulation Grids
    const importanceGrid = new Float32Array(height * width).fill(0);
    const countsGrid = new Float32Array(height * width).fill(0);

    // 3. Loop and Mask
    // We do this loop logically on X/Y coordinates
    const tensorData = await imgTensor.data(); // Flattened data
    
    // Pre-calculate normalization value for "Mean" padding
    // If we mask with 0.0 in a normalized tensor, that is actually grey/black.
    // Ideally, we mask with the "Mean" value of the dataset to simulate "missing data".
    // For ImageNet normalized tensors, 0.0 is actually a valid color value.
    // The best occlusion color for normalized tensors is often -2.0 (approx Black).
    const MASK_VALUE = -2.0; 

    for (let y = 0; y <= height - patchSize; y += stride) {
        for (let x = 0; x <= width - patchSize; x += stride) {
            
            // Clone the original data
            const maskedData = new Float32Array(tensorData);
            
            // Apply Mask directly to the Float32 Buffer
            for (let py = 0; py < patchSize; py++) {
                for (let px = 0; px < patchSize; px++) {
                    // Calculate 1D index based on layout
                    if (isNCHW) {
                        // NCHW: [Channel, Row, Col]
                        for (let c = 0; c < channels; c++) {
                            const idx = c * (height * width) + (y + py) * width + (x + px);
                            maskedData[idx] = MASK_VALUE;
                        }
                    } else {
                        // NHWC: [Row, Col, Channel]
                        const idx = ((y + py) * width + (x + px)) * channels;
                        maskedData[idx] = MASK_VALUE;     // R
                        maskedData[idx + 1] = MASK_VALUE; // G
                        maskedData[idx + 2] = MASK_VALUE; // B
                    }
                }
            }

            // Run Inference
            tf.tidy(() => {
                const maskedTensor = tf.tensor(maskedData, shape);
                const inputs = extraInputTensor ? [maskedTensor, extraInputTensor] : maskedTensor;
                const logits = model.predict(inputs);
                const score = tf.softmax(logits).dataSync()[targetClassIdx];
                
                // Accumulate Importance (Drop in confidence)
                const diff = Math.max(0, baselineScore - score);
                
                for (let py = 0; py < patchSize; py++) {
                    for (let px = 0; px < patchSize; px++) {
                        const gridIdx = (y + py) * width + (x + px);
                        importanceGrid[gridIdx] += diff;
                        countsGrid[gridIdx] += 1;
                    }
                }
            });
            
            // Anti-freeze
            if (x % (stride*2) === 0) await new Promise(r => setTimeout(r, 0));
        }
    }

    // Normalize and Draw
    let maxVal = 0;
    for(let i=0; i<importanceGrid.length; i++) {
        if(countsGrid[i] > 0) importanceGrid[i] /= countsGrid[i];
        if(importanceGrid[i] > maxVal) maxVal = importanceGrid[i];
    }

    return createHeatmapOverlay(importanceGrid, width, height, maxVal);
}

function createHeatmapOverlay(grid, w, h, maxVal) {
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);

    for (let i = 0; i < grid.length; i++) {
        const val = grid[i];
        const intensity = maxVal > 0 ? val / maxVal : 0;
        const pixelIdx = i * 4;
        
        // Simple Red Heatmap
        // Only show if intensity > 20%
        if (intensity > 0.2) {
            imgData.data[pixelIdx] = 255;
            imgData.data[pixelIdx + 3] = Math.floor(intensity * 200);
        } else {
            imgData.data[pixelIdx + 3] = 0;
        }
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL();
}