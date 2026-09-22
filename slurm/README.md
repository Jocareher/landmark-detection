# Entrenamiento y TTA en SNOW / DTIC (UPF)

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

En el login, desde la raíz del repositorio:

```bash
salloc --partition=short --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00
srun --pty bash
# Ya dentro del nodo de cómputo:
module avail Miniconda3
bash slurm/install_lmks.sh
exit
exit
```

El módulo predeterminado es `Miniconda3/4.9.2`, tal como figura en la guía.
Si `module avail` muestra otro, usá `LMKS_CONDA_MODULE=... bash slurm/install_lmks.sh`
y cambiá `conda_module` en los ajustes del siguiente paso. El instalador falla
si `lmks` ya existe: no modifica silenciosamente un entorno que esté en uso.
La instalación requiere acceso a los índices de Conda/pip desde el nodo.
Ningún job de entrenamiento/TTA instala paquetes ni consume tiempo GPU en ello.

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

# Experimentos completos independientes:
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization layer
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/upf_hpc.local.json --normalization instance
```

Se heredan pérdidas, augmentations y LR del YAML de normalizer. El job entrena el
normalizer completo, transition3/stage4 y las tres heads; congela el resto.
Activa evaluación sintética, pero deja BabyLand/InfAnFace para los jobs separados.
La pérdida PCA conserva el cambio a coordenadas de imagen y su peso del YAML;
los lanzadores no recalibran ese peso automáticamente.

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
