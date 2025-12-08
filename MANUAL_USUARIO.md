# 📖 Manual de Usuario
## Sistema de Detección de Enfermedades en Hojas de Maíz

---

## 📋 Índice

1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Guía de Uso](#guía-de-uso)
4. [Interpretación de Resultados](#interpretación-de-resultados)
5. [Preguntas Frecuentes](#preguntas-frecuentes)
6. [Solución de Problemas](#solución-de-problemas)

---

## 🌽 Introducción

Este sistema utiliza modelos de machine learning para detectar enfermedades en hojas de maíz a través del análisis de imágenes, el cual está entrenado para identificar tres condiciones:

- **Sana**: Hoja sin signos de enfermedad
- **Roya Común**: Enfermedad fúngica caracterizada por pústulas color naranja
- **Tizón Foliar**: Enfermedad que causa lesiones necróticas en las hojas

### ¿Cómo funciona?

1. Subes una foto de una hoja de maíz
2. El sistema analiza la imagen con un modelo de Deep Learning
3. Recibes un diagnóstico con el nivel de confianza

---

## 💻 Requisitos del Sistema

### Para usar la aplicación web:

- **Navegador web moderno**: Chrome, Firefox, Edge o Safari
- **Conexión a internet** (si está en servidor remoto)
- **Cámara o imágenes** de hojas de maíz

### Requisitos de las imágenes:

✅ **Formato**: JPG, PNG, JPEG  
✅ **Contenido**: Foto clara de una hoja de maíz  
✅ **Iluminación**: Buena luz natural o artificial  
✅ **Enfoque**: Imagen nítida, no borrosa  

❌ **Evitar**: Imágenes oscuras, borrosas o con múltiples hojas superpuestas

---

## 🎯 Guía de Uso

### Paso 1: Acceder a la Aplicación

1. Abre tu navegador web
2. Navega a la dirección de la aplicación (ejemplo: `http://localhost:5173`)
3. Verifica que el sistema esté conectado (debe aparecer **🟢 Servidor conectado**)

![Pantalla principal](./docs/screenshots/pantalla_principal.png)

### Paso 2: Cargar una Imagen

Tienes dos opciones:

**Opción A: Desde archivo**
1. Click en el botón **"Seleccionar imagen"** o área de carga
2. Busca la imagen en tu dispositivo
3. Selecciona el archivo y haz click en "Abrir"

**Opción B: Tomar foto (desde móvil)**
1. Click en el botón de carga
2. Selecciona "Tomar foto"
3. Captura la imagen de la hoja
4. Confirma la foto

### Paso 3: Analizar la Imagen

1. Una vez cargada, verás una vista previa de la imagen
2. Verifica que la imagen se vea correctamente
3. Click en el botón **"Enviar"** o **"Analizar"**
4. Espera unos segundos mientras el sistema procesa la imagen

> ⏱️ **Tiempo de análisis**: Generalmente entre 1-5 segundos

### Paso 4: Ver Resultados

El sistema mostrará:

1. **Diagnóstico principal**: La enfermedad más probable
2. **Nivel de confianza**: Qué tan seguro está el modelo (0-100%)
3. **Probabilidades por clase**: Desglose para cada categoría

---

## 📊 Interpretación de Resultados

### Entendiendo el diagnóstico

El sistema te mostrará algo como esto:

```
Resultado: Roya Común
Confianza: 92.5%

Probabilidades:
- Roya Común:    92.5%
- Sana:           5.2%
- Tizón Foliar:   2.3%
```

### ¿Qué significa el nivel de confianza?

| Confianza | Interpretación | Recomendación |
|-----------|---------------|---------------|
| 90-100% | **Muy alta** | El diagnóstico es muy confiable |
| 70-89% | **Alta** | Diagnóstico probablemente correcto |
| 50-69% | **Media** | Revisar con experto agrónomo |
| 0-49% | **Baja** | Se recomienda tomar otra foto o consultar experto |

### Consejos para mejorar la precisión

✅ **Usa fotos claras y bien iluminadas**  
✅ **Enfoca una sola hoja por vez**  
✅ **Captura ambos lados de la hoja si es posible**  
✅ **Toma fotos de diferentes ángulos**  
✅ **Evita sombras fuertes o reflejos**  

---

## ❓ Preguntas Frecuentes

### ¿Puedo analizar varias hojas a la vez?

No, el sistema está diseñado para analizar una hoja por imagen. Para mejores resultados, toma una foto de cada hoja individualmente.

### ¿Funciona con otros cultivos?

No, el modelo está específicamente entrenado para hojas de maíz. No funcionará correctamente con otros cultivos.

### ¿Qué hago si el diagnóstico es incorrecto?

El sistema puede equivocarse ocasionalmente. Si crees que el diagnóstico es incorrecto:
1. Toma otra foto con mejor iluminación
2. Intenta desde otro ángulo
3. Consulta con un agrónomo profesional

### ¿Los datos de mis imágenes se guardan?

Esto depende de la configuración del servidor. Consulta con el administrador del sistema sobre la política de privacidad.

### ¿Necesito internet para usar el sistema?

- **Versión web**: Sí, necesitas conexión a internet
- **Versión local**: No, si tienes todo instalado localmente

---

## 🔧 Solución de Problemas

### Problema: Servidor desconectado (🔴)

**Solución**:
1. Verifica tu conexión a internet
2. Asegúrate de que el servidor backend esté ejecutándose
3. Recarga la página (F5)
4. Si persiste, contacta al administrador

### Problema: Error al cargar imagen

**Solución**:
1. Verifica que el formato sea JPG, PNG o JPEG
2. Asegúrate de que el archivo no esté corrupto
3. Verifica el tamaño del archivo (máximo 10MB recomendado)
4. Intenta con otra imagen

### Problema: Análisis muy lento

**Solución**:
1. Reduce el tamaño de la imagen antes de subirla
2. Ver ifica tu conexión a internet
3. Espera unos segundos más (primera predicción puede tardar más)
4. Si persiste, contacta al administrador

### Problema: Resultados inconsistentes

**Solución**:
1. Asegúrate de que la imagen sea clara y bien iluminada
2. Evita imágenes con múltiples hojas
3. Toma varias fotos y compara resultados
4. Si la confianza es baja (<70%), considera consultar un experto

---

## 📞 Soporte Técnico

Si experimentas problemas técnicos que no puedes resolver:

- **Email**: [tu-email@universidad.edu]
- **Teléfono**: [número de contacto]
- **Horario**: Lunes a Viernes, 9:00 AM - 5:00 PM

---

## 📚 Recursos Adicionales

### Aprender más sobre las enfermedades

- **Roya Común**: [Enlace a información agronómica]
- **Tizón Foliar**: [Enlace a información agronómica]
- **Manejo integrado de plagas**: [Enlace a guías]

### Mejores prácticas agrícolas

Consulta con tu agrónomo local sobre:
- Prevención de enfermedades
- Tratamientos apropiados
- Rotación de cultivos
- Manejo de nutrientes

---

## ⚠️ Advertencias Importantes

> **NOTA LEGAL**: Este sistema es una herramienta de apoyo para la toma de decisiones. NO reemplaza el criterio de un profesional agrónomo. Siempre consulta con expertos antes de aplicar tratamientos.

> **PRECISIÓN**: El modelo tiene una precisión aproximada del 85-95%, pero puede cometer errores. Utiliza el sistema como guía, no como diagnóstico definitivo.

> **USO RESPONSABLE**: No tomes decisiones importantes de tratamiento basándote únicamente en este sistema.

---

## 📝 Historial de Versiones

- **v1.0** (Diciembre 2025) - Lanzamiento inicial con 3 clases

---

**© 2025 - Universidad Señor de Sipán - Investigación II**
