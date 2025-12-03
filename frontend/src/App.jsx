import { useState } from "react";
import "./App.css";

export default function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const enviarImagen = async () => {
    const form = new FormData();
    form.append("image", image);

    setLoading(true);
    const res = await fetch("http://localhost:5000/predict", {  // API Flask
      method: "POST",
      body: form
    });

    const data = await res.json();
    setResponse(data);
    setLoading(false);
  };

  return (
    <div className="container">

      <h1 className="title">🌽 Detector de Enfermedades en Hojas de Maíz</h1>

      {/* SUBIDA DE IMAGEN */}
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e)=> {
          setImage(e.target.files[0]);
          setPreview(URL.createObjectURL(e.target.files[0]));
        }}
      />

      {/* PREVIEW DE LA IMAGEN */}
      {preview && (
        <div className="preview-box">
          <img src={preview} alt="Imagen subida" className="preview-img" />
        </div>
      )}

      <button onClick={enviarImagen} disabled={!image || loading} className="btn">
        {loading ? "Procesando..." : "Detectar Enfermedad"}
      </button>

      {/* RESULTADOS DEL BACKEND EN VIVO */}
      {response && (
        <div className="result-box">
          <h2>Resultado del Análisis</h2>
          <p><b>Enfermedad detectada:</b> {response.prediccion}</p>
          <p><b>Confianza del modelo:</b> {response.confianza}%</p>

          <h3>📌 Probabilidades por clase</h3>
          <pre className="json-box">{JSON.stringify(response.probabilidades, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
