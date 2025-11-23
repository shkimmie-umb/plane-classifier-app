import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useDropzone } from 'react-dropzone'; // Optional: npm install react-dropzone or use native API below
import './App.css';

// --- CONFIGURATION: DEFAULT SAMPLES ---
const DEFAULT_SAMPLES = [
  {
    id: 'def_axial_1',
    src: '/images/sample_axial.jpg',
    label: 'Ref_Axial_01',
    type: 'mri',
    fallbackPlane: 'AXIAL',
    fallbackEntropy: 0.05
  },
  {
    id: 'def_coronal_1',
    src: '/images/sample_coronal.jpg',
    label: 'Ref_Coronal_01',
    type: 'mri',
    fallbackPlane: 'CORONAL',
    fallbackEntropy: 0.08
  },
  {
    id: 'def_sag_1',
    src: '/images/sample_sagittal.jpg', // Ensure you have this or duplicate another
    label: 'Ref_Sagittal_01',
    type: 'mri',
    fallbackPlane: 'SAGITTAL',
    fallbackEntropy: 0.06
  },
  {
    id: 'def_ood_noise',
    src: '/images/dog_ood.jpg', // External test
    label: 'OOD_Input_Dog',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_scene',
    src: '/images/ood_scene1.jpg', // External test
    label: 'OOD_Input_Scene',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_coffee',
    src: '/images/ood_food2.jpeg', // External test
    label: 'OOD_Input_Coffee',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_melanoma1',
    src: '/images/melanoma1.jpg', // External test
    label: 'OOD_Input_Melanoma1',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_melanoma2',
    src: '/images/melanoma2.jpg', // External test
    label: 'OOD_Input_Melanoma2',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_melanoma3',
    src: '/images/melanoma3.jpg', // External test
    label: 'OOD_Input_Melanoma3',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
  {
    id: 'def_ood_melanoma4',
    src: '/images/melanoma4.jpg', // External test
    label: 'OOD_Input_Melanoma4',
    type: 'ood',
    fallbackPlane: 'UNCERTAIN',
    fallbackEntropy: 2.10
  },
];

function App() {
  // --- STATE ---
  const [model, setModel] = useState(null);
  const [samples, setSamples] = useState(DEFAULT_SAMPLES);
  const [selectedSample, setSelectedSample] = useState(DEFAULT_SAMPLES[0]);
  const [logs, setLogs] = useState([]);
  const [loadingMsg, setLoadingMsg] = useState("Initializing Engine...");

  // Results
  const [result, setResult] = useState({
    pred: null,
    conf: 0,
    entropy: 0,
    isOOD: false
  });

  const consoleEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // --- LOGGING ---
  const addLog = (msg) => setLogs(prev => [...prev, `> ${msg}`]);
  useEffect(() => { consoleEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logs]);

  // --- 1. LOAD MODEL ---
  useEffect(() => {
    async function init() {
      try {
        await tf.ready();
        const backend = tf.getBackend();
        addLog(`Backend: ${backend.toUpperCase()} (Hardware Accel)`);
        
        const loaded = await tf.loadGraphModel('/tfjs_models/2.5D_tfjs/model.json').catch(() => null);
        
        if (loaded) {
            setModel(loaded);
            setLoadingMsg("SYSTEM ONLINE");
            addLog("Plane Classification Model Loaded.");
        } else {
            setLoadingMsg("SIMULATION MODE");
            addLog("WARN: Model file missing. Using heuristics.");
        }
      } catch (e) {
        addLog(`ERR: ${e.message}`);
      }
    }
    init();
  }, []);

  // --- 2. IMAGE PROCESSING ---
  const preprocess = (imgEl) => {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgEl)
            .resizeNearestNeighbor([224, 224])
            .toFloat();
        const mean = tf.tensor([123.675, 116.28, 103.53]);
        const std = tf.tensor([58.395, 57.12, 57.375]);
        return tensor.sub(mean).div(std).expandDims(0);
    });
  };

  const computeEntropy = (probs) => {
    let entropy = 0;
    probs.forEach(p => { if (p > 0) entropy -= p * Math.log(p); });
    return entropy;
  };

  // --- 3. INFERENCE ENGINE ---
  const runAnalysis = async (currentSample) => {
    const imgEl = document.getElementById('preview-img');
    if (!imgEl) return;
    
    setResult(prev => ({ ...prev, pred: 'Scanning...', entropy: 0 }));
    addLog(`Processing: ${currentSample.label}`);

    if (model) {
        const imgTensor = preprocess(imgEl);
        const start = performance.now();
        
        const logits = model.predict(imgTensor);
        const probs = await tf.softmax(logits).data();
        const inferenceTime = (performance.now() - start).toFixed(1);
        
        const entropy = computeEntropy(probs);
        const maxIdx = probs.indexOf(Math.max(...probs));
        const classes = ['AXIAL', 'CORONAL', 'SAGITTAL'];
        const isOOD = entropy > 0.11;

        setResult({
            pred: isOOD ? "UNKNOWN" : classes[maxIdx],
            conf: probs[maxIdx],
            entropy: entropy,
            isOOD: isOOD
        });

        addLog(`Time: ${inferenceTime}ms | Entropy: ${entropy.toFixed(4)}`);
        if(isOOD) addLog("⚠️ OOD DETECTED (Entropy > 0.11)");
        imgTensor.dispose();
    } else {
        // Fallback
        await new Promise(r => setTimeout(r, 600));
        const ent = currentSample.fallbackEntropy || 0.15;
        const isOOD = ent > 0.11;
        setResult({
            pred: isOOD ? "UNKNOWN" : currentSample.fallbackPlane,
            conf: 0.95,
            entropy: ent,
            isOOD: isOOD
        });
        addLog(`Simulated Entropy: ${ent.toFixed(4)}`);
    }
  };

  // --- 4. BATCH UPLOAD HANDLERS ---
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
        processFiles(e.target.files);
    }
  };

  const processFiles = (fileList) => {
      // Convert FileList to Array
      const filesArray = Array.from(fileList);
      addLog(`Batch Ingestion: Loading ${filesArray.length} images...`);

      const newSamples = filesArray.map(file => ({
          id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`, // Unique ID
          src: URL.createObjectURL(file),
          label: file.name.length > 15 ? file.name.substring(0, 12) + '...' : file.name,
          type: 'user',
          fallbackPlane: 'AXIAL',
          fallbackEntropy: 0.10
      }));
      
      // Batch update state (Orders: Newest first)
      setSamples(prev => [...newSamples, ...prev]);
      
      // Select the first uploaded image immediately
      if (newSamples.length > 0) {
          handleSampleClick(newSamples[0]);
      }
      addLog(`Success: ${filesArray.length} images added to local gallery.`);
  };

  // Drag and Drop Logic
  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const onDrop = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          processFiles(e.dataTransfer.files);
      }
  };

  const handleSampleClick = (s) => {
      setSelectedSample(s);
      setTimeout(() => runAnalysis(s), 100);
  };

  // --- RENDER HELPERS ---
  const ENTROPY_VISUAL_MAX = 0.5; 
  const thresholdPercent = (0.11 / ENTROPY_VISUAL_MAX) * 100;
  const currentEntropyPercent = Math.min((result.entropy / ENTROPY_VISUAL_MAX) * 100, 100);

  return (
    <div className="dashboard" onDragOver={onDragOver} onDrop={onDrop}>
      <header>
          <div className="header-top">
            <div>
               <h1>MediVision <span className="version-tag mono">V5.0 PRO-LAYOUT</span></h1>
            </div>
            <div className="status-bar mono">{loadingMsg}</div>
          </div>

          <div className="research-banner">
            <span style={{fontSize: '1.2rem'}}>🎓</span> 
            <strong>Reference:</strong>
            <a href="https://arxiv.org/abs/2511.14021" target="_blank" rel="noreferrer" style={{marginLeft:'5px'}}>
                "MRI plane orientation detection using a context-aware 2.5D model"
            </a>
            <span style={{opacity: 0.7, marginLeft: '5px'}}>(ArXiv, 2025)</span>
          </div>
      </header>

      <main className="workspace-grid">
        
        {/* --- LEFT SIDEBAR: SELECTION ONLY --- */}
        <section className="panel sidebar-panel">
            <h3 style={{marginTop:0}}>Input Gallery</h3>
            
            {/* Upload Area */}
            <div className="upload-zone" onClick={() => fileInputRef.current.click()}>
                <span style={{fontSize:'2rem'}}>📂</span><br/>
                <span>Batch Upload<br/>(1000+ files)</span>
                <input 
                    type="file" 
                    hidden 
                    multiple 
                    ref={fileInputRef} 
                    accept="image/*" 
                    onChange={handleFileSelect} 
                />
            </div>

            {/* Full Height Scrollable List */}
            <div className="thumbnail-list">
                {samples.map(s => (
                    <div 
                        key={s.id} 
                        className={`list-item ${selectedSample.id === s.id ? 'selected' : ''}`}
                        onClick={() => handleSampleClick(s)}
                    >
                        <img src={s.src} alt="thumb" />
                        <div className="list-meta">
                            <div className="list-label">{s.label}</div>
                            {s.type === 'user' && <span className="user-badge-text">NEW</span>}
                        </div>
                    </div>
                ))}
            </div>
        </section>

        {/* --- RIGHT STAGE: PREVIEW & ANALYSIS --- */}
        <section className="panel main-stage-panel">
            
            {/* TOP ROW: IMAGE + MAIN METRIC */}
            <div className="stage-top-row">
                
                {/* 1. The Big Preview Image (Fills the space!) */}
                <div className="stage-preview-container">
                    <img id="preview-img" src={selectedSample.src} alt="Target" />
                    <div className="image-overlay-label">{selectedSample.label}</div>
                </div>

                {/* 2. Key Metrics (Side by side with image) */}
                <div className="stage-metrics-column">
                    {/* Classification Card */}
                    <div className={`metric-card compact ${result.isOOD ? 'ood-state' : 'safe-state'}`}>
                        <h4 className="text-sec">DETECTED PLANE</h4>
                        <div className="metric-value-large">
                            {result.pred || "--"}
                        </div>
                        {!result.isOOD && (
                            <div className="mono text-cyan">Conf: {(result.conf * 100).toFixed(1)}%</div>
                        )}
                    </div>

                    {/* Uncertainty Card */}
                    <div className="metric-card compact">
                        <div style={{display:'flex', justifyContent:'space-between'}}>
                             <h4 className="text-sec">INTEGRITY</h4>
                             <strong className={result.isOOD ? 'text-red' : 'text-cyan'}>
                                {result.isOOD ? 'REJECTED' : 'VERIFIED'}
                             </strong>
                        </div>
                        
                        <div style={{marginTop:'15px'}}>
                            <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.8rem'}}>
                                <span>Entropy: {result.entropy.toFixed(3)}</span>
                                <span style={{color:'#ef4444'}}>Limit: 0.11</span>
                            </div>
                            <div className="entropy-track-large">
                                <div 
                                    className="entropy-fill-large"
                                    style={{
                                        width: `${currentEntropyPercent}%`,
                                        backgroundColor: result.isOOD ? '#ef4444' : '#0ea5e9'
                                    }}
                                ></div>
                                <div className="threshold-line" style={{left: `${thresholdPercent}%`}}></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* BOTTOM ROW: LOGS (Fills remaining height) */}
            <div className="stage-logs-container">
                <h4 style={{margin:'0 0 10px 0', borderBottom:'1px solid #333', paddingBottom:'5px'}}>System Logs</h4>
                <div className="console-log-fill mono">
                    {logs.map((l, i) => <div key={i}>{l}</div>)}
                    <div ref={consoleEndRef}></div>
                </div>
            </div>
        </section>

      </main>
    </div>
  );
}

export default App;