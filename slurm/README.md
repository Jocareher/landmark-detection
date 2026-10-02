# Entrenamiento y TTA en SNOW / DTIC (UPF)

## Elegir la pérdida PCA

En los YAML de entrenamiento y TTA, `arguments.pca_loss_space` admite:

- `aligned`: pérdida original, MSE entre la predicción alineada por Procrustes
  y su reconstrucción PCA. No se aplica la inversa.
- `image`: MSE entre la predicción original y la reconstrucción PCA transformada
  de vuelta a la imagen, usando la inversa completa con gradientes (píxeles²).

Se usa una sola pérdida PCA y el mismo prior sirve para ambas. Los YAML activos
seleccionan `aligned`; configuraciones antiguas sin esta clave conservan `image`.
No se normaliza por tamaño de cara. El valor numérico de la pérdida y su peso
no son comparables entre espacios: no reutilices un peso/LR como si fueran equivalentes.
La elección para TTA es independiente de la usada al entrenar el checkpoint.

## Dos experimentos de normalización global

Ambos usan el fine-tuning habitual: normalizer, transition3/stage4 y heads.
`batch_all_norms` agrega BatchNorm a cada bloque oculto del normalizer; el
backbone y las heads conservan BatchNorm. `instance_all_norms` coloca
InstanceNorm en esos bloques y en las heads y reemplaza **todas** las BatchNorm
del HRNet por InstanceNorm. En este último también se entrenan los parámetros
afines de todas las InstanceNorm del backbone, aun donde las convoluciones
siguen congeladas. Los pesos convolucionales preentrenados y los parámetros
afines de BN se cargan por nombre; las medias/varianzas acumuladas de BN no
existen en InstanceNorm.

```bash
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization batch_all_norms \
  --pca-loss-space aligned --run-name train_bn_all_norms
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance_all_norms \
  --pca-loss-space aligned --run-name train_in_all_norms
```

Para TTA, usá **el checkpoint de cada entrenamiento**. En el primer caso,
`--batch-size` define cuántas imágenes comparten una adaptación y un Adam;
las BN seleccionadas usan estadísticas del batch. Pesos, buffers BN y Adam se
restauran al estado de SynBaby antes del siguiente batch. El último batch
puede ser más pequeño. En el segundo caso, cada episodio contiene una sola
imagen y solo se actualizan el normalizer y los parámetros afines de IN del
backbone y heads:

```bash
bash slurm/tta_landmarks.sh \
  --settings configs/upf_hpc.local.json \
  --checkpoint /ruta/train_bn_all_norms/checkpoints/full_model_best.pth \
  --dataset babyland --scope normalizer_all_norms --batch-size 4 \
  --pca-loss-space aligned --run-name tta_bn_all_norms_b4
bash slurm/tta_landmarks.sh \
  --settings configs/upf_hpc.local.json \
  --checkpoint /ruta/train_in_all_norms/checkpoints/full_model_best.pth \
  --dataset babyland --scope normalizer_all_instance_norms --batch-size 1 \
  --pca-loss-space aligned --run-name tta_in_all_norms_b1
```

Las rutas del ejemplo se reemplazan por los checkpoints impresos por el envío
de entrenamiento. `--dry-run` valida recursos/rutas sin enviar el job; para
medir memoria y tiempo reales, enviá después un trabajo con `--steps 1`.

```bash
# Local: la CLI sobrescribe el YAML.
python -m scripts.main --config configs/normalizer_experiments.yaml --pca-loss-space aligned
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml --pca-loss-space aligned

# HPC: la misma opción sirve para entrenamiento y TTA.
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance_early --pca-loss-space aligned
bash slurm/tta_landmarks.sh \
  --settings configs/upf_hpc.local.json --checkpoint /ruta/full_model_best.pth \
  --dataset babyland --scope normalizer --pca-loss-space aligned
```

Para volver a la variante en coordenadas de imagen, reemplazá `aligned` por
`image`. Usá nombres de ejecución y directorios diferentes para las comparaciones.
La selección se registra en la configuración resuelta, los checkpoints de
entrenamiento y el resumen de TTA.

## Experimento InstanceNorm temprano en local

Editá las rutas de datasets, pesos, PCA y salidas en
`configs/normalizer_experiments.yaml`. El preset activo usa normalizer sin
normalización interna, InstanceNorm al final de su último bloque oculto,
InstanceNorm después de `backbone.layer1` y heads con BatchNorm:

```bash
python -m scripts.main --config configs/normalizer_experiments.yaml
```

Al terminar, poné la ruta real de `full_model_best.pth` y las rutas de BabyLand
en `configs/pca_tta_evaluation.yaml`. Elegí un `output_dir` y
`wandb_run_name` distintos para cada alcance y ejecutá:

```bash
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml
```

Los valores de `pca_tta_adaptation_scope` disponibles para la comparación son
`normalizer`, `normalizer_stem`, `normalizer_stem_layer1` y
`normalizer_layer1_instance`. La arquitectura procede del checkpoint de
entrenamiento, no del YAML de evaluación.

Estos lanzadores corresponden a **SNOW**, descrito en la guía indicada por el
usuario, no a las particiones `std-gpu`/`high-gpu` de Correfoc. Se consultó la
documentación el 22-09-2026:

- [Colas y límites](https://guiesbibtic.upf.edu/recerca/hpc/cluster-queues):
  `short` hasta 2 horas, `medium` hasta 8 horas, `high` para trabajos más largos.
  La guía es inconsistente entre 14 días e ilimitado para `high`: el lanzador
  limita su selección automática a 14 días y comprueba el `MaxTime` real.
- [GPU y GRES](https://guiesbibtic.upf.edu/recerca/hpc/cuda-jobs): solicitar GPU
  mediante `--gres=gpu:TIPO:1`; el tipo debe ser el que registra Slurm.
- [Hardware](https://guiesbibtic.upf.edu/recerca/hpc/system-overview): L40S 48 GB,
  RTX 6000 24 GB, T4 y GTX 1080 Ti; almacenamiento compartido BeeGFS.
- [Conda](https://guiesbibtic.upf.edu/recerca/hpc/conda): crear el entorno en un nodo
  de cómputo, cargar el módulo e inicializar Conda antes de activarlo.

## 1. Entorno Linux a partir de `lmks`

Se inspeccionó `/Users/jocareher/anaconda3/envs/lmks`, no el Python del sistema.
Versiones encontradas: Python 3.11.15, torch 2.10.0, torchvision 0.25.0,
NumPy 2.4.3, pandas 3.0.3, Pillow 12.1.1, matplotlib 3.10.8, PyYAML 6.0.3,
tqdm 4.67.3, wandb 0.25.1 y torchinfo 1.8.0. La conversión Torch/NumPy funciona
en ese entorno. Los imports de entrenamiento/TTA no requieren OpenCV, sklearn,
seaborn ni scipy; scipy es opcional en otro script de análisis estadístico.
Rich es opcional y los logs Slurm ya utilizan la salida de texto normal.

No exportamos binarios macOS: `environments/lmks-hpc.yml` crea una base Linux y
`requirements-hpc.txt` conserva las versiones de las dependencias utilizadas.
Instalamos torch/torchvision desde el índice CUDA 12.6 publicado para estas
[versiones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/).
No se necesita torchaudio. Se requiere un nodo Linux compatible con esos wheels
y un driver NVIDIA compatible; el job ejecuta operaciones CUDA con backward para
verificarlo. Cargar un módulo CUDA no actualiza el driver del nodo.

Los jobs crean automáticamente el entorno configurado si no existe, instalan
PyTorch con CUDA y todas las dependencias y lo activan antes de ejecutar Python.
Si ya existe y supera la comprobación de imports, CUDA build, Torch/NumPy y
`pip check`, lo reutilizan sin reinstalar. Si está incompleto, intentan instalar
las dependencias. Un entorno existente con Python anterior a 3.11 requiere elegir
un nombre nuevo en `conda_env` para no modificar su versión de Python.

Un bloqueo compartido (`flock` en `$HOME/.cache/lmks-hpc`) evita que dos jobs creen
o reparen entornos a la vez. La primera instalación necesita acceso a Conda/pip
desde el nodo y consume parte del tiempo reservado; los errores quedan en los
logs Slurm y detienen el job antes del entrenamiento. No hace falta un paso manual.

Opcionalmente, `bash slurm/install_lmks.sh` permite preparar el entorno antes,
dentro de una asignación interactiva. Usa la misma lógica y se puede repetir.
El módulo predeterminado es `Miniconda3/4.9.2`; se puede cambiar en `conda_module`.

## 2. Ajustar rutas y recursos una sola vez

```bash
cp configs/upf_hpc.example.json configs/upf_hpc.local.json
sinfo -o '%P %l %G'
scontrol show partition high
```

Editá `configs/upf_hpc.local.json` con tus rutas **absolutas del HPC** y, si se
requiere, tu cuenta Slurm. Ubicá datasets, pesos y resultados en almacenamiento
compartido accesible desde los nodos. No copies los caches de muestras del Mac:
contienen rutas de otra máquina; cada job crea su propio cache y evita carreras
entre experimentos paralelos. Los datasets deben conservar sus imágenes/labels
y splits; no están incluidos en el clon de Git.

Para TTA también son obligatorias `paths.babyland_source_root` y
`paths.infanface_source_root` (solo se valida el dataset seleccionado). Apuntan
al directorio de **imágenes originales**, no al de recortes ni al de labels.
Si ya tenés un JSON local, agregá esas dos claves dentro de `paths`.
El lanzador pasa la seleccionada como `natural_source_root` al evaluador.

No hace falta editar los metadatos del detector: una ruta antigua como
`/Users/usuario/dataset/subject/image.jpg` se busca bajo la nueva raíz conservando
los directorios finales (`subject/image.jpg`) o como `image.jpg` si el directorio
es plano. Si varias alternativas existen, se detiene con un error para evitar
usar una imagen incorrecta. No se hace búsqueda recursiva por nombre. La raíz
explícita tiene prioridad y no vuelve a las rutas antiguas si falta una imagen.
Conservá la estructura y los nombres originales al copiar el dataset. La matriz
`transform_crop_to_orig` sigue viniendo de los metadatos y no se modifica.

`gpu_type: auto` prefiere un GRES identificable como L40S/Ada, seguido de RTX6000,
consultando `sinfo` en la partición seleccionada. No asume que `gpu:l40s:1` sea un
nombre válido. Si Slurm solo publica `turing` u otro nombre ambiguo, pide definir
`gpu_type`/`--gpu-type` explícitamente. El preflight exige por defecto **22 GiB**
para evitar caer sin querer en una T4 de 16 GB. Para probar T4, seleccioná su tipo
real, reducí el batch y ajustá el mínimo de memoria conscientemente. Ni 22 GiB
ni batch 32 garantizan que todo experimento quepa: medir el consumo real.

Recursos iniciales (estimaciones, NO tiempos medidos):

| Job | GPU | CPU | RAM | Tiempo | Cola automática |
|---|---|---|---|---|---|
| Entrenamiento | 1 | 8 | 32 GB | 24 h | high |
| TTA | 1 | 4 | 16 GB | 8 h | medium |

El código actual usa una sola GPU; pedir más no lo acelera. El lanzador usa
`--ntasks=1`, controla workers e impide superar el `MaxTime` real de la cola.
Se puede usar `--time HH:MM:SS` y `--partition ...`; la cola automática es la más
corta compatible con el tiempo solicitado. El tipo GPU debe existir en esa cola.
No fija `CUDA_VISIBLE_DEVICES`: respeta la asignación de Slurm.

## 3. Lanzar entrenamiento

Los wrappers se ejecutan con **bash**, no con `sbatch`: ellos preparan los logs,
congelan el código y llaman a `sbatch`. El login necesita Python 3.6+ para el
lanzador (solo biblioteca estándar, sin importar PyTorch ni YAML).

```bash
# Validación de rutas/colas/GRES, sin enviar nada:
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization layer --dry-run

# Ensayo de una época, para medir memoria y duración incluyendo validación/test:
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization layer \
  --epochs 1 --time 02:00:00

# Baseline: normalizer sin normalización, heads con BatchNorm.
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization baseline

# Experimentos completos independientes:
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization layer
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance
```

Para comparar las dos posiciones de InstanceNorm con PCA original:

```bash
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance_hidden \
  --pca-loss-space aligned --run-name train_in_hidden_layer1_pca_aligned
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance_early \
  --pca-loss-space aligned --run-name train_in_final_layer1_pca_aligned
```

Agregá `--dry-run` para validar sin enviar. Los presets resuelven:

| Preset | Normalizer | IN después de layer1 | Heads |
|---|---|---|---|
| `instance_hidden` | Conv → IN → ReLU en cada bloque oculto | Sí | BatchNorm |
| `instance_early` | IN solo en el último bloque oculto, antes de ReLU | Sí | BatchNorm |

La convolución RGB final queda sin normalización en ambos. La posición final
se corrigió a antes de ReLU; los checkpoints anteriores que la colocaban
después de ReLU conservan su topología al cargarlos desde sus metadatos.
Se heredan las demás pérdidas, augmentations y LR del YAML de normalizer.
En SynBaby se actualizan el normalizer, la
nueva InstanceNorm de layer1, transition3/stage4 y las heads. Los convolucionales
del stem/layer1 y sus estadísticas BatchNorm quedan congelados. Los antiguos
presets mantienen su arquitectura original y desactivan ambas InstanceNorm nuevas.
También evalúa BabyLand e InfAnFace al terminar, porque
`evaluate_babyland: true` y `evaluate_infanface: true` figuran en el YAML.
Podés poner cualquiera de las dos en `false` para omitirla. Cuando está activa,
el lanzador toma recortes, labels e imágenes originales del JSON HPC; las rutas
locales escritas en el YAML se sustituyen. El entrenamiento usa solo synbaby72.
Los jobs TTA siguen siendo independientes: hacen la adaptación por imagen y su
evaluación posterior. Reservá tiempo para estas evaluaciones al solicitar GPU.
La pérdida PCA usa `pca_loss_space` del YAML (`aligned` o `image`), salvo que
se pase `--pca-loss-space`. Los lanzadores no recalibran su peso automáticamente.

Cada envío imprime el job ID, un directorio único y la ruta futura del checkpoint.
No hay reanudación automática tras timeout: los checkpoints por época existentes
quedan en disco, pero el modo joint-finetune actual no acepta resume. Por eso
medí primero y solicitá un tiempo suficiente con margen, en vez de confiar en
requeue. Los jobs se envían con `--no-requeue`.

## 4. Lanzar TTA después del entrenamiento

Usá la ruta y el ID que imprimió el envío de entrenamiento:

```bash
bash slurm/tta_landmarks.sh \
  --settings configs/upf_hpc.local.json \
  --checkpoint /ruta/mostrada/checkpoints/full_model_best.pth \
  --afterok 123456 \
  --dataset babyland --scope normalizer
```

`--afterok` hace que el job espere a que el entrenamiento termine con éxito.
Si ya terminó, omití esa opción. Un fallo en la dependencia impide ejecutar TTA.
El checkpoint todavía puede no existir al enviar la dependencia; se verifica
nuevamente dentro del nodo antes de ejecutar la evaluación.

Para los tres modos (cada uno tiene su propio job/directorio/logs):

```bash
for scope in normalizer normalizer_head_norms normalizer_heads; do
  bash slurm/tta_landmarks.sh \
    --settings configs/upf_hpc.local.json \
    --checkpoint /ruta/mostrada/checkpoints/full_model_best.pth \
    --afterok 123456 --dataset babyland --scope "$scope"
done
```

Para comparar los cuatro alcances nuevos con el checkpoint `instance_early`:

```bash
for scope in normalizer normalizer_stem normalizer_stem_layer1 normalizer_layer1_instance; do
  bash slurm/tta_landmarks.sh \
    --settings configs/upf_hpc.local.json \
    --checkpoint /ruta/mostrada/checkpoints/full_model_best.pth \
    --dataset babyland --scope "$scope"
done
```

`normalizer_stem` actualiza conv1/bn1/conv2/bn2; `normalizer_stem_layer1`
agrega layer1 y la nueva InstanceNorm, sin tocar transition1; y
`normalizer_layer1_instance` ajusta solo los parámetros afines de esa
InstanceNorm además del normalizer. Los cuatro modos reinician los pesos por
imagen y mantienen las estadísticas de BatchNorm fijas durante TTA. Los alcances
con stem/layer1 necesitan más memoria y tiempo: medí primero con `--steps 1` y
ajustá `--time` y `--batch-size` según el resultado.

Repetí con `--dataset infanface` y/o el checkpoint de InstanceNorm. La arquitectura
se carga de los metadatos del checkpoint, nunca de `--normalization` en TTA.
Se admiten también `--checkpoint .../landmarker_best.pth` y
`--normalizer-checkpoint .../normalizer_best.pth` del mismo entrenamiento.
`--steps`, `--batch-size` y `--time` permiten ajustar cada envío.

## 5. Registro y ajuste del tiempo

Cada directorio contiene:

- `code/`: copia del código enviado, independiente de cambios posteriores al clon.
- `metadata/`: YAML base y resuelto, comandos, commit/diff, hashes de fuentes,
  paquetes instalados, GPU/driver, recursos Slurm, job ID y estado de ejecución.
- `logs/slurm-JOBID.out` y `.err`: texto sin animaciones de terminal (también se desactivan las barras tqdm).
- `logs/gpu.csv`: muestra cada 60 segundos la GPU asignada, si expone UUID.
- Los checkpoints, métricas, gráficos y diagnósticos habituales del experimento.

W&B queda **offline** por defecto: evita depender de conectividad/login durante
la ejecución y se puede sincronizar luego con `wandb sync /ruta/run/wandb/offline-*`.
Para online, configurá `wandb_mode` y autenticá W&B antes; no pongas tokens en JSON.
No se registra un volcado completo del entorno que pueda exponer credenciales.

```bash
squeue -u "$USER"
sacct -j 123456 --format=JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS,AllocTRES
```

Usá `sacct` como estado definitivo, especialmente ante timeout o SIGKILL, donde
Python podría no llegar a actualizar `execution.json`. Para entrenar, estimá
`épocas × (tiempo train + validación)` y sumá carga de datos, test/exportación y
un margen medido. Para TTA, usá `imágenes × tiempo por episodio` más exportación;
repetí la medición por scope/GPU, porque adaptar heads cambia el coste.
No se ha ejecutado ni medido este pipeline en SNOW desde la máquina local.

### Python antiguo en el nodo de acceso

El lanzador es compatible con Python 3.6+ y no necesita activar `lmks` en el
nodo de acceso. El entrenamiento y TTA siguen ejecutándose con Python 3.11 en
`lmks`, dentro del nodo asignado. Comprobá el Python del lanzador con
`python3 --version`. Si necesitás elegir otro ejecutable, usá
`LMKS_LAUNCH_PYTHON=/ruta/al/python bash slurm/train_hrnet_landmarks_template.sh ...`
(también funciona para `slurm/tta_landmarks.sh`).

El preset `--normalization baseline` cambia solo la normalización: normalizer
`none`, heads `batch`, backbone con su BatchNorm original. Conserva las pérdidas
(incluido el peso PCA del YAML), augmentations y política de fine-tuning de los
otros presets. El nombre de los resultados se toma de `wandb_run_name` del YAML.
TTA carga esta arquitectura desde el checkpoint, con los mismos comandos y scopes.

### Nombre de cada ejecución y actualizaciones del repositorio

Antes de enviar, editá `arguments.wandb_run_name` en
`configs/normalizer_experiments.yaml` para entrenamiento, o en
`configs/pca_tta_evaluation.yaml` para TTA. Por ejemplo,
`wandb_run_name: baseline_sin_norm` genera
`baseline_sin_norm_FECHA_ID/`. El nombre de Slurm usa ese prefijo y W&B usa el
nombre completo de la carpeta. Espacios y caracteres de ruta se sustituyen por
`_`. Usá un nombre simple en una línea con la indentación existente. Cambiar
`--normalization` no cambia automáticamente el nombre elegido en el YAML.
También podés pasar `--run-name NOMBRE` para sobrescribirlo solo en ese envío,
sin editar el YAML entre experimentos. La carpeta conserva el sufijo de fecha/ID.

Después de que el envío devuelve el job ID, `git pull` no modifica el código ni
el YAML de ese job: están copiados en su directorio, incluso mientras está en
cola. Los datasets, pesos de entrada y el entorno Conda siguen siendo compartidos;
no se deben mover o modificar mientras los jobs los usan. Conservá tus rutas HPC
en `configs/upf_hpc.local.json` (ignorado por Git) para evitar conflictos al actualizar.

### AdaIN con referencia fija de synbaby72

`--normalization adain` selecciona AdaIN en
`configs/normalizer_experiments.yaml` durante el entrenamiento. El normalizer y las tres heads usan AdaIN; el backbone
conserva BatchNorm. Se entrenan normalizer, transition3/stage4 y heads igual
que en los otros experimentos. Durante cada época, las capas AdaIN actualizan
una estimación de media y desviación por canal a partir del split de entrenamiento.
Validación y TTA no actualizan esa referencia. Al finalizar, se recupera el mejor
checkpoint, se recalculan los momentos con **todo** el split `synbaby72/train`
sin augmentations, y se guarda el checkpoint calibrado antes de la evaluación
final. Este recorrido adicional requiere tiempo de GPU: dejá margen en `train.time`.
La referencia y los parámetros afines quedan guardados en
`checkpoints/full_model_best.pth`; no se necesitan imágenes fuente durante TTA.
Antes de enviarlo, cambiá `wandb_run_name` en el YAML compartido, por ejemplo a
`train_adain_pca_image`; `--normalization` no cambia el nombre del run.

```bash
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization adain

bash slurm/tta_landmarks.sh \
  --settings configs/upf_hpc.local.json \
  --checkpoint /ruta/al/run/checkpoints/full_model_best.pth \
  --dataset babyland --scope normalizer
```

Para comparar los tres scopes de TTA, cambiá `--scope` a
`normalizer_head_norms` o `normalizer_heads`. El primero actualiza la escala y
el sesgo AdaIN de las heads; las medias y desviaciones de referencia siguen
fijas. Cada job recibe su propio directorio. El lanzador utiliza
`configs/pca_tta_evaluation.yaml` y sustituye las rutas HPC, el checkpoint y
el scope según los parámetros del comando. La arquitectura se lee del checkpoint.

En local, ajustá las rutas, `wandb_run_name` y ambos campos de normalización
a `adain` en `configs/normalizer_experiments.yaml`, luego ejecutá
`python -m scripts.main --config configs/normalizer_experiments.yaml`.
Para TTA local, ajustá el checkpoint, las rutas y el nombre de salida en
`configs/pca_tta_evaluation.yaml`, luego ejecutá
`python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml`.
