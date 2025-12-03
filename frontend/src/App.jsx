import { useState } from "react";
import "./App.css";

export default function App() {
  const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://localhost:5000";
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

      {error && (
        <div className="result-box error">
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      )}

      {/* RESULTADOS DEL BACKEND EN VIVO */}
      {response && (
        <div className="result-box">
          <h2>Resultado del Análisis</h2>
          <p><b>Clase predicha:</b> {response.clase_predicha}</p>

          {response.vecinos && (
            <>
              <h3>Vecinos más cercanos</h3>
              <pre className="json-box">{JSON.stringify(response.vecinos, null, 2)}</pre>
            </>
          )}

          {response.distancias && (
            <>
              <h3>Distancias a vecinos</h3>
              <pre className="json-box">{JSON.stringify(response.distancias, null, 2)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
