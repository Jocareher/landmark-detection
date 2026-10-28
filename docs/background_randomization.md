# Randomizar el fondo blanco de SynBaby sin modificar la cara

## Hipótesis

El fondo blanco constante del entrenamiento sintético podría actuar como una pista de dominio que no existe en BabyLand o InfAnFace. Los GIF del normalizer sugieren que gran parte del cambio visual ocurre en el fondo, pero esa observación por sí sola no demuestra que el backbone use el blanco como *feature*: el fondo también ocupa muchos más píxeles que la cara. Este experimento cambia solo el fondo de entrenamiento para poner a prueba la hipótesis.

La idea general es **domain randomization**: variar una característica secundaria de los ejemplos sintéticos para que el modelo dependa menos de ella. Véase [Tobin et al., 2017](https://arxiv.org/abs/1703.06907). El efecto concreto aquí debe medirse; no se da por supuesto.

## Implementación

La augmentación es optativa y está **desactivada por defecto**. La configuración experimental está en [`configs/normalizer_background_randomization.yaml`](../configs/normalizer_background_randomization.yaml). Copia el entrenamiento actual de backbone + normalizer y cambia solamente el nombre de la corrida y esta augmentación.

Para cada imagen sintética de `train`:

1. Después de redimensionar, se detectan píxeles casi blancos y neutros (`min(R,G,B)>225`, cromaticidad pequeña). La selección se hace **antes** de las otras augmentaciones fotométricas, para reconocer el fondo blanco original.
2. Los landmarks visibles definen un **contorno convexo** que protege su interior. También se protege la silueta no blanca de la imagen y se añade un margen corto de 2 px para cubrir antialiasing sin dejar un marco blanco amplio.
3. Fuera de esa protección, un peso gradual sustituye los píxeles casi blancos por un color aleatorio con textura suave de baja frecuencia y un ruido fino leve. La nueva operación conserva exactamente los píxeles protegidos y no añade ni mueve landmarks.
4. El reemplazo se aplica **después** del blur, ruido y JPEG existentes. Así esos pasos no pueden propagar el nuevo fondo hacia la zona facial. Un 30 % de las imágenes conserva el fondo tal como llega a este paso.

Esto es distinto de poner solo ruido gaussiano sobre blanco: ese procedimiento suele conservar un fondo casi blanco y puede enseñar una textura artificial de alta frecuencia. Aquí se varían color, luminosidad y textura suave. Las otras augmentaciones fotométricas del entrenamiento actual **siguen modificando la cara** como ya lo hacían; la nueva operación no la modifica respecto a la imagen que recibe.

**Límite importante:** no hay máscara alfa o de segmentación en el dataset disponible. Por eso el contorno de landmarks combinado con el umbral de color es una protección conservadora, no una segmentación exacta de cara/cabello. Puede dejar un halo blanco estrecho o proteger partes no faciales; para un recorte perfecto haría falta una máscara del renderizador. La vista previa permite detectar si ese efecto es excesivo.

## Revisar la máscara antes de entrenar

Ejecutar en el entorno donde existe `synbaby72`:

```bash
python -m scripts.analysis.preview_background_randomization \
  --config configs/normalizer_background_randomization.yaml \
  --output outputs/background_preview.png \
  --count 24
```

El script genera una cuadrícula con imagen original y caja de landmarks de referencia, imagen aumentada, diferencia RGB y píxeles modificados. La protección real sigue el contorno, **no** el rectángulo dibujado. También crea `outputs/background_preview.csv` con la fracción de píxeles cambiados por imagen.

Revisar especialmente perfiles, cabello, orejas, cuello y ejemplos donde la cara toque el borde. Si casi no cambia ningún píxel, la protección es demasiado amplia o el umbral de blanco es demasiado estricto. Si un píxel facial cambia por la nueva operación, la aproximación no cumple el objetivo y se debe corregir antes de entrenar.

El dataset configurado vive en una ruta de HPC (`/home/jocareher/Downloads/landmark-detection/data/synbaby72`) que no está disponible en este equipo; por eso aquí se verificó el transform con imágenes de prueba, pero no se generó una vista previa de tus renders reales.

## Entrenar y comparar

```bash
python -m scripts.main --config configs/normalizer_background_randomization.yaml
```

Mantener el mismo split, semilla, inicialización, arquitectura, pérdidas y presupuesto de entrenamiento que en la corrida base. Comparar en **BabyLand, InfAnFace y validación sintética**:

- NME y Hausdorff antes de TTA, para medir si mejora la generalización del modelo entrenado.
- NME y Hausdorff después de la misma TTA, con idénticos hiperparámetros, para medir si cambia el comportamiento de adaptación.
- Diferencia original/espejo y errores por pose, porque en los análisis anteriores hubo asimetrías.
- Cambio medio absoluto del normalizer **por píxel de cara y por píxel de fondo**, usando una región facial conservadora. Comparar medias por región, no sumas: el fondo ocupa más área.

Si el modelo mejora en imágenes naturales sin perder demasiado en validación sintética, eso apoyaría la hipótesis de que el fondo fijo perjudicaba la transferencia. Si solo cambia el aspecto de los GIF pero no el NME, la evidencia no apoyaría esa explicación.
