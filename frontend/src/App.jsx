import { useState, useEffect } from "react";
import "./App.css";

export default function App() {
  const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://localhost:5000";
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(false);

  // Check server health on mount
  useEffect(() => {
    fetch(`${apiBaseUrl}/health`)
      .then((res) => {
        if (res.ok) setServerStatus(true);
      })
      .catch(() => setServerStatus(false));
  }, [apiBaseUrl]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResponse(null); // Clear previous results
      setError(null);
    }
  };

  const enviarImagen = async () => {
    if (!image) return;

    const form = new FormData();
    form.append("image", image);

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        body: form
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Error al procesar la imagen");
      }

      setResponse(data);
    } catch (err) {
      setResponse(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clearImage = (e) => {
    e.stopPropagation();
    setImage(null);
    setPreview(null);
    setResponse(null);
    setError(null);
  };

  return (
    <div className="container">
      {/* HEADER */}
      <div className="header-card">
        <div className="status-badge">
          <div className={`status-dot ${serverStatus ? "active" : "inactive"}`} 
               style={{ backgroundColor: serverStatus ? "#22c55e" : "#ef4444", boxShadow: `0 0 8px ${serverStatus ? "#22c55e" : "#ef4444"}`}}></div>
          {serverStatus ? "Servidor conectado" : "Desconectado"}
        </div>
        
        <h1 className="title">🌽 Detección de Enfermedades en Hojas de Maíz</h1>
        <p className="subtitle">Modelo de visión por computadora para clasificar hojas de maíz</p>
      </div>

      <div className="content-grid">
        {/* LEFT COMPONENT: IMAGE INPUT */}
        <div className="card-panel">
          <div className="panel-header">Imagen</div>
          
          <label className="image-area">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange}
              className="input-file"
            />
            
            {preview ? (
              <>
                <img src={preview} alt="Vista previa" className="preview-img" />
                <div className="close-btn" onClick={clearImage}>✕</div>
              </>
            ) : (
              <div className="placeholder-text">
                <span style={{fontSize: "2rem"}}>📷</span>
                <p>Haz clic para subir una imagen</p>
              </div>
            )}
          </label>

          <button onClick={enviarImagen} disabled={!image || loading} className="btn">
            {loading ? "Analizando..." : "Realizar Predicción"}
          </button>

          {error && (
            <div style={{color: "#ef4444", marginTop: "10px", textAlign: "center"}}>
              {error}
            </div>
          )}
        </div>

        {/* RIGHT COMPONENT: RESULTS */}
        <div className="card-panel">
          <div className="panel-header">Predicciones</div>
          
          <div className="prediction-area">
            {response ? (
              <>
                <div className="main-result">
                  {response.prediccion}
                </div>

                <div className="probabilities-list">
                  {Object.entries(response.probabilidades).map(([label, prob]) => (
                    <div key={label} style={{marginBottom: "10px"}}>
                      <div className="prob-row">
                        <span className="prob-label">{label}</span>
                        <div className="prob-bar-container">
                          <div 
                            className="prob-bar-fill" 
                            style={{width: `${prob}%`}}
                          ></div>
                        </div>
                        <span className="prob-value">{prob}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{textAlign: "center", color: "#666", padding: "40px"}}>
                <p>Los resultados aparecerán aquí</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
